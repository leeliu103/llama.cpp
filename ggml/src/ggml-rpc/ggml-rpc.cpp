#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "mk-qwen3moe-tp.h"

#include <array>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <thread>

#ifndef _WIN32
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");
static const char * RPC_LOAD_TRACE = std::getenv("GGML_RPC_LOAD_TRACE");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

#define LOG_LOAD(...) \
    do { if (RPC_LOAD_TRACE) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_COMM_GET_ID,
    RPC_CMD_COMM_INIT,
    RPC_CMD_ALLREDUCE,
    RPC_CMD_GRAPH_COMPUTE_SEQ,
    RPC_CMD_SET_TENSOR_2D_HASH,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_MK_INIT,
    RPC_CMD_MK_DECODE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

// 2D (strided) tensor set used by the tensor-parallel meta backend for split weights.
// The whole tensor (all n_copies blocks of `size` bytes) is hashed as one unit so that
// large split weights hit the local file cache even though each individual block is small.
struct rpc_msg_set_tensor_2d_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;          // bytes per block
    uint64_t n_copies;
    uint64_t stride_tensor; // dst stride between blocks (tensor side)
    uint64_t hash;          // fnv_hash over the gathered n_copies*size contiguous bytes
};

struct rpc_msg_set_tensor_2d_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

struct rpc_msg_comm_get_id_rsp {
    uint8_t unique_id[128];
};

struct rpc_msg_comm_init_req {
    uint32_t nranks;
    uint32_t rank;
    uint32_t device;
    uint8_t  unique_id[128];
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_allreduce_req {
    rpc_tensor tensor;
};

struct rpc_msg_allreduce_rsp {
    uint8_t ok;
};

struct rpc_msg_graph_compute_seq_rsp {
    uint8_t ok;
};

// RPC_CMD_MK_INIT: this header, then n_tensors rpc_tensor in the order of rpc_mk_tensors()
struct rpc_msg_mk_init_hdr {
    uint32_t device;
    int32_t  rank;
    int32_t  n_ranks;
    int32_t  n_layers;
    int32_t  n_cells;
    int32_t  vocab_off;
    int32_t  emb_type;  // token_embd stays on the orchestrator's host: its type and shape; rows come per token
    int32_t  emb_ne0;
    int32_t  emb_ne1;
    uint32_t n_tensors;
};

struct rpc_msg_mk_init_rsp {
    uint8_t ok;
    char    err[255];
};

// RPC_CMD_MK_DECODE: this, then emb_bytes of the token's token_embd row; the response is int32 status, then the
// device's LM head rows of logits (float)
struct rpc_msg_mk_decode_req {
    int32_t  token;
    int32_t  pos;
    uint32_t emb_bytes;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    uint64_t    last_graph_uid;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

static uint64_t fnv_hash_init() {
    return 0xcbf29ce484222325ULL;
}

static uint64_t fnv_hash_update(uint64_t hash, const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    return fnv_hash_update(fnv_hash_init(), data, len);
}

static uint64_t fnv_hash_2d(const uint8_t * data, size_t size, size_t n_copies, size_t stride_data) {
    if (stride_data == size) {
        return fnv_hash(data, size * n_copies);
    }

    uint64_t hash = fnv_hash_init();
    for (size_t i = 0; i < n_copies; i++) {
        hash = fnv_hash_update(hash, data + i*stride_data, size);
    }
    return hash;
}

struct rpc_hash_2d_cache_key {
    uintptr_t data;
    size_t size;
    size_t n_copies;
    size_t stride_data;
    uint32_t type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::string name;
    uint64_t generation;

    bool operator==(const rpc_hash_2d_cache_key & other) const {
        return data == other.data
            && size == other.size
            && n_copies == other.n_copies
            && stride_data == other.stride_data
            && type == other.type
            && ne == other.ne
            && name == other.name
            && generation == other.generation;
    }
};

static void hash_combine(size_t & seed, size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct rpc_hash_2d_cache_key_hash {
    size_t operator()(const rpc_hash_2d_cache_key & key) const {
        size_t seed = 0;
        hash_combine(seed, std::hash<uintptr_t>{}(key.data));
        hash_combine(seed, std::hash<size_t>{}(key.size));
        hash_combine(seed, std::hash<size_t>{}(key.n_copies));
        hash_combine(seed, std::hash<size_t>{}(key.stride_data));
        hash_combine(seed, std::hash<uint32_t>{}(key.type));
        for (int64_t ne : key.ne) {
            hash_combine(seed, std::hash<int64_t>{}(ne));
        }
        hash_combine(seed, std::hash<std::string>{}(key.name));
        hash_combine(seed, std::hash<uint64_t>{}(key.generation));
        return seed;
    }
};

static std::mutex & rpc_hash_2d_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static uint64_t & rpc_hash_2d_cache_generation() {
    static uint64_t generation = 0;
    return generation;
}

static std::unordered_map<rpc_hash_2d_cache_key, uint64_t, rpc_hash_2d_cache_key_hash> & rpc_hash_2d_cache() {
    static std::unordered_map<rpc_hash_2d_cache_key, uint64_t, rpc_hash_2d_cache_key_hash> cache;
    return cache;
}

static void rpc_hash_2d_cache_clear() {
    std::lock_guard<std::mutex> lock(rpc_hash_2d_cache_mutex());
    rpc_hash_2d_cache().clear();
    rpc_hash_2d_cache_generation()++;
}

static rpc_hash_2d_cache_key rpc_hash_2d_make_cache_key(const ggml_tensor * tensor, const uint8_t * data,
        size_t size, size_t n_copies, size_t stride_data) {
    rpc_hash_2d_cache_key key = {};
    key.data = reinterpret_cast<uintptr_t>(data);
    key.size = size;
    key.n_copies = n_copies;
    key.stride_data = stride_data;
    key.type = tensor->type;
    key.name = tensor->name;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        key.ne[i] = tensor->ne[i];
    }
    {
        std::lock_guard<std::mutex> lock(rpc_hash_2d_cache_mutex());
        key.generation = rpc_hash_2d_cache_generation();
    }
    return key;
}

static uint64_t fnv_hash_2d_cached(const ggml_tensor * tensor, const uint8_t * data, size_t size, size_t n_copies,
        size_t stride_data, bool * cache_hit) {
    rpc_hash_2d_cache_key key = rpc_hash_2d_make_cache_key(tensor, data, size, n_copies, stride_data);
    {
        std::lock_guard<std::mutex> lock(rpc_hash_2d_cache_mutex());
        auto it = rpc_hash_2d_cache().find(key);
        if (it != rpc_hash_2d_cache().end()) {
            *cache_hit = true;
            return it->second;
        }
    }

    uint64_t hash = fnv_hash_2d(data, size, n_copies, stride_data);
    {
        std::lock_guard<std::mutex> lock(rpc_hash_2d_cache_mutex());
        rpc_hash_2d_cache().emplace(std::move(key), hash);
    }
    *cache_hit = false;
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        const int64_t t_hash0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        request.hash = fnv_hash((const uint8_t*)data, size);
        const int64_t t_hash1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        rpc_msg_set_tensor_hash_rsp response;
        const int64_t t_rpc0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        const int64_t t_rpc1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        RPC_STATUS_ASSERT(status);
        LOG_LOAD("[rpc-load] client set_tensor_hash tensor=%s bytes=%zu wire_req=%zu wire_rsp=%zu hash_us=%" PRId64 " rpc_us=%" PRId64 " result=%u\n",
                 tensor->name, size, sizeof(request), sizeof(response), t_hash1 - t_hash0, t_rpc1 - t_rpc0, response.result);
        if (response.result) {
            // the server has the same data, no need to send it
            LOG_LOAD("[rpc-load] client set_tensor done tensor=%s bytes=%zu source=cache total_us=%" PRId64 "\n",
                     tensor->name, size, ggml_time_us() - t_total0);
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    const int64_t t_pack0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    const int64_t t_pack1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    const int64_t t_send0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    const int64_t t_send1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    RPC_STATUS_ASSERT(status);
    LOG_LOAD("[rpc-load] client set_tensor done tensor=%s bytes=%zu input=%zu source=upload pack_us=%" PRId64 " send_us=%" PRId64 " total_us=%" PRId64 "\n",
             tensor->name, size, input.size(), t_pack1 - t_pack0, t_send1 - t_send0, ggml_time_us() - t_total0);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    const size_t total = n_copies * size;
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;

    if (total > HASH_THRESHOLD) {
        rpc_msg_set_tensor_2d_hash_req request;
        request.tensor        = rpc_tensor;
        request.offset        = offset;
        request.size          = size;
        request.n_copies      = n_copies;
        request.stride_tensor = stride_tensor;
        const int64_t t_hash0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        bool hash_cache_hit   = false;
        request.hash          = fnv_hash_2d_cached(tensor, (const uint8_t *) data, size, n_copies, stride_data, &hash_cache_hit);
        const int64_t t_hash1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        rpc_msg_set_tensor_2d_hash_rsp response;
        const int64_t t_rpc0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_2D_HASH, &request, sizeof(request), &response, sizeof(response));
        const int64_t t_rpc1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        RPC_STATUS_ASSERT(status);
        LOG_LOAD("[rpc-load] client set_tensor_2d_hash tensor=%s total=%zu row=%zu n=%zu stride_tensor=%zu stride_data=%zu wire_req=%zu wire_rsp=%zu gather_us=0 hash_us=%" PRId64 " hash_cache=%u rpc_us=%" PRId64 " result=%u\n",
                 tensor->name, total, size, n_copies, stride_tensor, stride_data,
                 sizeof(request), sizeof(response), t_hash1 - t_hash0, hash_cache_hit ? 1 : 0, t_rpc1 - t_rpc0, response.result);
        if (response.result) {
            // the server has the same data cached, no need to send it
            LOG_LOAD("[rpc-load] client set_tensor_2d done tensor=%s total=%zu source=cache total_us=%" PRId64 "\n",
                     tensor->name, total, ggml_time_us() - t_total0);
            return;
        }
    }

    // Gather the n_copies blocks only when the data must be uploaded.
    const uint8_t * gathered = (const uint8_t *) data;
    std::vector<uint8_t> gather_buf;
    const int64_t t_gather0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    if (stride_data != size) {
        gather_buf.resize(total);
        for (size_t i = 0; i < n_copies; i++) {
            memcpy(gather_buf.data() + i*size, (const uint8_t*)data + i*stride_data, size);
        }
        gathered = gather_buf.data();
    }
    const int64_t t_gather1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;

    // input format: | rpc_tensor | offset | size | n_copies | stride_tensor | gathered data (total bytes) |
    const size_t header = sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    const int64_t t_pack0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    std::vector<uint8_t> input(header + total, 0);
    uint8_t * p = input.data();
    memcpy(p, &rpc_tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    const uint64_t u_offset = offset, u_size = size, u_ncopies = n_copies, u_stride = stride_tensor;
    memcpy(p, &u_offset, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, &u_size, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, &u_ncopies, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, &u_stride, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, gathered, total);
    const int64_t t_pack1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    const int64_t t_send0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_2D, input.data(), input.size());
    const int64_t t_send1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    RPC_STATUS_ASSERT(status);
    LOG_LOAD("[rpc-load] client set_tensor_2d done tensor=%s total=%zu input=%zu source=upload gather_us=%" PRId64 " pack_us=%" PRId64 " send_us=%" PRId64 " total_us=%" PRId64 "\n",
             tensor->name, total, input.size(), t_gather1 - t_gather0, t_pack1 - t_pack0,
             t_send1 - t_send0, ggml_time_us() - t_total0);
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        rpc_hash_2d_cache_clear();
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation

struct rpc_cached_file {
    std::vector<uint8_t> buffer;
    const uint8_t * ptr = nullptr;
    size_t len = 0;

#ifndef _WIN32
    void * mapping = MAP_FAILED;
    size_t mapping_len = 0;
#endif

    rpc_cached_file() = default;
    rpc_cached_file(const rpc_cached_file &) = delete;
    rpc_cached_file & operator=(const rpc_cached_file &) = delete;

    ~rpc_cached_file() {
#ifndef _WIN32
        if (mapping != MAP_FAILED) {
            munmap(mapping, mapping_len);
        }
#endif
    }

    const uint8_t * data() const {
        return ptr;
    }

    size_t size() const {
        return len;
    }

#ifndef _WIN32
    void set_mapping(void * data, size_t size) {
        mapping = data;
        mapping_len = size;
        ptr = (const uint8_t *) data;
        len = size;
    }
#endif

    void set_buffer(std::vector<uint8_t> data) {
        buffer = std::move(data);
        ptr = buffer.data();
        len = buffer.size();
    }
};

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        uid_graphs.resize(backends.size());
        uid_ctxs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_2d_hash(const rpc_msg_set_tensor_2d_hash_req & request, rpc_msg_set_tensor_2d_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    bool comm_get_id(rpc_msg_comm_get_id_rsp & response);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool allreduce(const rpc_msg_allreduce_req & request, rpc_msg_allreduce_rsp & response);
    bool graph_compute_seq(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_rsp & response);
    bool mk_init(const std::vector<uint8_t> & input, rpc_msg_mk_init_rsp & response);
    bool mk_decode(const std::vector<uint8_t> & input, std::vector<uint8_t> & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, rpc_cached_file & file);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // resolve the HIP/CUDA NCCL exports from a backend's reg (returns false if unavailable)
    bool resolve_nccl_procs(ggml_backend_t backend);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
    // graphs cached by uid (per device) for batched decode (RPC_CMD_GRAPH_COMPUTE_SEQ)
    std::vector<std::unordered_map<uint64_t, stored_graph>> uid_graphs;
    // owning contexts for the cached graphs above (kept alive across tokens)
    std::vector<std::unordered_map<uint64_t, ggml_context_ptr>> uid_ctxs;

    // cross-process RCCL/NCCL collective state (one comm per server, single GPU)
    void *        nccl_comm   = nullptr;
    ggml_backend_t nccl_backend = nullptr;
    bool   (*nccl_get_unique_id)(void *)                                   = nullptr;
    void * (*nccl_comm_init_rank)(ggml_backend_t, int, int, const void *)  = nullptr;
    bool   (*nccl_allreduce)(ggml_backend_t, void *, struct ggml_tensor *) = nullptr;
    void   (*nccl_comm_free)(void *)                                       = nullptr;

    // llama.cpp's tensor-split decode kernel for this device (RPC_CMD_MK_*): the library is GGML_RPC_MK_LIB
    // (libmk-qwen3moe-tp.so), its own exchange between the servers GGML_RPC_MK_PEERS ("odl:host:port,...")
    void *         mk         = nullptr;
    ggml_backend_t mk_backend = nullptr;
    int32_t        mk_rows    = 0;
    void * (*mk_create_fn)(const mk_tp_model *, char *, size_t) = nullptr;
    int    (*mk_decode_fn)(void *, int32_t, int32_t, const void *, float *) = nullptr;
    void   (*mk_free_fn)(void *)                                = nullptr;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        const int64_t t_cache0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
        LOG_LOAD("[rpc-load] server set_tensor cache_save tensor=%s hash=%016" PRIx64 " bytes=%zu cache_us=%" PRId64 "\n",
                 in_tensor->name, hash, size, ggml_time_us() - t_cache0);
    }
    const int64_t t_set0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    ggml_backend_tensor_set(tensor, data, offset, size);
    const int64_t t_set1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    LOG_LOAD("[rpc-load] server set_tensor done tensor=%s bytes=%zu input=%zu tensor_set_us=%" PRId64 " total_us=%" PRId64 "\n",
             in_tensor->name, size, input.size(), t_set1 - t_set0, ggml_time_us() - t_total0);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, rpc_cached_file & file) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    const int64_t t0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;

#ifndef _WIN32
    int fd = open(cache_file.c_str(), O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void * mapping = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (mapping != MAP_FAILED) {
                file.set_mapping(mapping, (size_t) st.st_size);
                LOG_LOAD("[rpc-load] server cache_read hash=%016" PRIx64 " bytes=%zu source=mmap read_us=%" PRId64 "\n",
                         hash, file.size(), ggml_time_us() - t0);
                return true;
            }
        } else {
            close(fd);
        }
    }
#endif

    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    ifs.read((char *) data.data(), size);
    file.set_buffer(std::move(data));
    LOG_LOAD("[rpc-load] server cache_read hash=%016" PRIx64 " bytes=%zu source=read read_us=%" PRId64 "\n",
             hash, size, ggml_time_us() - t0);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    rpc_cached_file cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        LOG_LOAD("[rpc-load] server set_tensor_hash miss tensor=%s hash=%016" PRIx64 " total_us=%" PRId64 "\n",
                 request.tensor.name, request.hash, ggml_time_us() - t_total0);
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    const int64_t t_set0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    const int64_t t_set1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    response.result = 1;
    LOG_LOAD("[rpc-load] server set_tensor_hash hit tensor=%s hash=%016" PRIx64 " bytes=%zu tensor_set_us=%" PRId64 " total_us=%" PRId64 "\n",
             request.tensor.name, request.hash, size, t_set1 - t_set0, ggml_time_us() - t_total0);
    return true;
}

bool rpc_server::set_tensor_2d_hash(const rpc_msg_set_tensor_2d_hash_req & request, rpc_msg_set_tensor_2d_hash_rsp & response)
{
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    response.result = 0;
    rpc_cached_file cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        LOG_LOAD("[rpc-load] server set_tensor_2d_hash miss tensor=%s hash=%016" PRIx64 " total_us=%" PRId64 "\n",
                 request.tensor.name, request.hash, ggml_time_us() - t_total0);
        return true;
    }
    const size_t size          = request.size;
    const size_t n_copies      = request.n_copies;
    const size_t stride_tensor = request.stride_tensor;
    if (size == 0 || n_copies == 0 || cached_file.size() != size * n_copies) {
        GGML_LOG_ERROR("[%s] cached file size mismatch (have %zu, want %zu*%zu)\n",
                       __func__, cached_file.size(), size, n_copies);
        return true; // treat as miss so the client re-sends
    }
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    // sanitize: the last written byte must stay within the buffer
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        const uint64_t last = request.tensor.data + request.offset + (n_copies-1)*stride_tensor + size;
        if (request.tensor.data + request.offset < p0 || last > p1) {
            GGML_LOG_ERROR("[%s] 2D tensor region out of buffer bounds\n", __func__);
            return false;
        }
    }
    const int64_t t_set0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    if (stride_tensor == size) {
        ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, cached_file.size());
    } else {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, cached_file.data() + i*size, request.offset + i*stride_tensor, size);
        }
    }
    const int64_t t_set1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    response.result = 1;
    LOG_LOAD("[rpc-load] server set_tensor_2d_hash hit tensor=%s hash=%016" PRIx64 " total=%zu row=%zu n=%zu stride_tensor=%zu contiguous=%d tensor_set_us=%" PRId64 " total_us=%" PRId64 "\n",
             request.tensor.name, request.hash, cached_file.size(), size, n_copies, stride_tensor, stride_tensor == size,
             t_set1 - t_set0, ggml_time_us() - t_total0);
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    const int64_t t_total0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    // format: | rpc_tensor | offset | size | n_copies | stride_tensor | gathered data (n_copies*size) |
    const size_t header = sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    if (input.size() < header) {
        return false;
    }
    const uint8_t * p = input.data();
    const rpc_tensor * in_tensor = (const rpc_tensor *)p; p += sizeof(rpc_tensor);
    uint64_t offset, size, n_copies, stride_tensor;
    memcpy(&offset, p, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(&size, p, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(&n_copies, p, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(&stride_tensor, p, sizeof(uint64_t)); p += sizeof(uint64_t);
    const size_t total = size * n_copies;
    if (size == 0 || n_copies == 0 || input.size() != header + total) {
        GGML_LOG_ERROR("[%s] malformed 2D set_tensor (input %zu, want %zu)\n", __func__, input.size(), header + total);
        return false;
    }
    const uint8_t * data = p;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    // sanitize: the last written byte must stay within the buffer
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        const uint64_t last = in_tensor->data + offset + (n_copies-1)*stride_tensor + size;
        if (in_tensor->data + offset < p0 || last > p1) {
            GGML_LOG_ERROR("[%s] 2D tensor region out of buffer bounds\n", __func__);
            return false;
        }
    }

    // Cache the whole tensor under its hash so future loads hit set_tensor_2d_hash.
    if (cache_dir && total > HASH_THRESHOLD) {
        const int64_t t_cache0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
        uint64_t hash = fnv_hash(data, total);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, total);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
        LOG_LOAD("[rpc-load] server set_tensor_2d cache_save tensor=%s hash=%016" PRIx64 " total=%zu cache_us=%" PRId64 "\n",
                 in_tensor->name, hash, total, ggml_time_us() - t_cache0);
    }

    const int64_t t_set0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    if (stride_tensor == size) {
        ggml_backend_tensor_set(tensor, data, offset, total);
    } else {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, data + i*size, offset + i*stride_tensor, size);
        }
    }
    const int64_t t_set1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
    LOG_LOAD("[rpc-load] server set_tensor_2d done tensor=%s total=%zu row=%zu n=%zu stride_tensor=%" PRIu64 " contiguous=%d input=%zu tensor_set_us=%" PRId64 " total_us=%" PRId64 "\n",
             in_tensor->name, total, size, n_copies, stride_tensor, stride_tensor == size, input.size(), t_set1 - t_set0, ggml_time_us() - t_total0);
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    stored_graphs[device].graph = graph;
    return true;
}

// Batched per-token execution for tensor-parallel decode. Runs an ordered sequence of
// (subgraph compute -> optional allreduce) steps autonomously, enqueueing everything on the
// backend stream and synchronizing exactly once at the end. This collapses ~180 per-layer RPC
// round-trips + host syncs into a single command, and keeps the RCCL ranks in lockstep
// structurally (every server runs the same fixed sequence; the k-th allreduce pairs by enqueue
// order) rather than relying on orchestrator pacing.
//
// Wire format (after device):
//   device(4) | n_steps(4) |
//   per step: uid(8) | has_graph(1) | [ gsize(4) | graph_blob[gsize] if has_graph ]
//             has_ar(1) | [ ar_tensor(rpc_tensor) if has_ar ]
// Graphs are cached by uid. The first decode token sends has_graph=1 (server builds and caches
// the graph under uid in a persistent context). Subsequent tokens send has_graph=0 and the server
// reuses the cached graph (no rebuild, HIP graph reused) - this is what makes batching a win.
// ar_tensor's data pointers are stable across tokens too, but it is cheap to re-send each time.
// Response: ok=1 success; ok=2 means a referenced uid was not cached -> orchestrator should run a
// normal warm-up pass for this token and retry batched next token.
bool rpc_server::graph_compute_seq(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_rsp & response) {
    response.ok = 0;
    const uint8_t * p   = input.data();
    const uint8_t * end = input.data() + input.size();
    if (end - p < (ptrdiff_t)(2*sizeof(uint32_t))) {
        return false;
    }
    uint32_t device;
    memcpy(&device, p, sizeof(device)); p += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_steps;
    memcpy(&n_steps, p, sizeof(n_steps)); p += sizeof(n_steps);

    auto & cache = uid_graphs[device];
    // transient contexts for allreduce tensors (graphs live in the persistent cache)
    std::vector<ggml_context_ptr> ar_ctxs;

    for (uint32_t s = 0; s < n_steps; s++) {
        if (end - p < (ptrdiff_t)(sizeof(uint64_t) + 1)) {
            return false;
        }
        uint64_t uid;
        memcpy(&uid, p, sizeof(uid)); p += sizeof(uid);
        uint8_t has_graph = *p; p += 1;

        ggml_cgraph * graph = nullptr;
        if (has_graph) {
            if (end - p < (ptrdiff_t)sizeof(uint32_t)) {
                return false;
            }
            uint32_t gsize;
            memcpy(&gsize, p, sizeof(gsize)); p += sizeof(gsize);
            if (end - p < (ptrdiff_t)gsize || gsize < 2*sizeof(uint32_t)) {
                return false;
            }
            const uint8_t * gp = p; p += gsize;
            gp += sizeof(uint32_t); // skip embedded device
            uint32_t n_nodes;
            memcpy(&n_nodes, gp, sizeof(n_nodes)); gp += sizeof(n_nodes);
            const uint64_t * nodes = (const uint64_t *)gp;
            gp += (size_t)n_nodes*sizeof(uint64_t);
            uint32_t n_tensors;
            memcpy(&n_tensors, gp, sizeof(n_tensors)); gp += sizeof(n_tensors);
            const rpc_tensor * tensors = (const rpc_tensor *)gp;

            // build into a persistent per-uid context so the graph survives across tokens
            stored_graph & sg = cache[uid];
            size_t buf_size = ggml_tensor_overhead()*((size_t)n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
            sg.buffer.assign(buf_size, 0);
            ggml_init_params params = { buf_size, sg.buffer.data(), true };
            // NOTE: keep the context alive for the lifetime of the cached graph.
            // We stash it in a static-like holder via a leak-free owner list.
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                return false;
            }
            uid_ctxs[device][uid].reset(ctx);
            ggml_cgraph * g = ggml_new_graph_custom(ctx, n_nodes, false);
            g->n_nodes = n_nodes;
            std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
            tensor_ptrs.reserve(n_tensors);
            for (uint32_t i = 0; i < n_tensors; i++) {
                tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
            }
            std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
            tensor_map.reserve(n_nodes);
            for (uint32_t i = 0; i < n_nodes; i++) {
                int64_t id;
                memcpy(&id, &nodes[i], sizeof(id));
                g->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
                if (g->nodes[i] == nullptr && id != 0) {
                    GGML_LOG_ERROR("[%s] failed to create node %u (step %u)\n", __func__, i, s);
                    return false;
                }
            }
            sg.graph = g;
            graph = g;
        } else {
            auto it = cache.find(uid);
            if (it == cache.end() || it->second.graph == nullptr) {
                // not cached: ask orchestrator to warm up this token via the normal path
                response.ok = 2;
                return true;
            }
            graph = it->second.graph;
        }

        ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
        if (status != GGML_STATUS_SUCCESS) {
            return false;
        }

        if (end - p < (ptrdiff_t)1) {
            return false;
        }
        uint8_t has_ar = *p; p += 1;
        if (has_ar) {
            if (nccl_comm == nullptr || nccl_backend == nullptr) {
                return false;
            }
            if (end - p < (ptrdiff_t)sizeof(rpc_tensor)) {
                return false;
            }
            const rpc_tensor * ar_rt = (const rpc_tensor *)p;
            p += sizeof(rpc_tensor);
            ggml_init_params arp = { ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr arctx { ggml_init(arp) };
            ggml_tensor * ar_tensor = deserialize_tensor(arctx.get(), ar_rt);
            if (ar_tensor == nullptr) {
                return false;
            }
            ar_ctxs.push_back(std::move(arctx));
            if (!nccl_allreduce(nccl_backend, nccl_comm, ar_tensor)) {
                return false;
            }
        }
    }

    // single host sync for the whole token
    ggml_backend_synchronize(backends[device]);
    response.ok = 1;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

bool rpc_server::resolve_nccl_procs(ggml_backend_t backend) {
    if (nccl_get_unique_id != nullptr) {
        return true;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (reg == nullptr) {
        return false;
    }
    nccl_get_unique_id  = (bool   (*)(void *))                                   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nccl_get_unique_id");
    nccl_comm_init_rank = (void * (*)(ggml_backend_t, int, int, const void *))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nccl_comm_init_rank");
    nccl_allreduce      = (bool   (*)(ggml_backend_t, void *, struct ggml_tensor *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nccl_allreduce");
    nccl_comm_free      = (void   (*)(void *))                                   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nccl_comm_free");
    if (!nccl_get_unique_id || !nccl_comm_init_rank || !nccl_allreduce || !nccl_comm_free) {
        nccl_get_unique_id = nullptr;
        GGML_LOG_ERROR("[%s] backend does not export NCCL collective procs (build with GGML_HIP_RCCL/GGML_CUDA_NCCL)\n", __func__);
        return false;
    }
    return true;
}

bool rpc_server::comm_get_id(rpc_msg_comm_get_id_rsp & response) {
    if (backends.empty() || !resolve_nccl_procs(backends[0])) {
        return false;
    }
    memset(response.unique_id, 0, sizeof(response.unique_id));
    return nccl_get_unique_id(response.unique_id);
}

bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size() || !resolve_nccl_procs(backends[dev_id])) {
        return true;
    }
    if (nccl_comm != nullptr) {
        nccl_comm_free(nccl_comm);
        nccl_comm = nullptr;
    }
    nccl_backend = backends[dev_id];
    nccl_comm = nccl_comm_init_rank(nccl_backend, (int) request.nranks, (int) request.rank, request.unique_id);
    response.ok = (nccl_comm != nullptr) ? 1 : 0;
    LOG_DBG("[%s] rank %u/%u device %u -> %s\n", __func__, request.rank, request.nranks, dev_id, response.ok ? "ok" : "FAILED");
    return true;
}

bool rpc_server::allreduce(const rpc_msg_allreduce_req & request, rpc_msg_allreduce_rsp & response) {
    response.ok = 0;
    if (nccl_comm == nullptr || nccl_backend == nullptr) {
        return true;
    }
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    ggml_tensor * tensor = deserialize_tensor(ctx_ptr.get(), &request.tensor);
    if (tensor == nullptr) {
        return true;
    }
    bool rc = nccl_allreduce(nccl_backend, nccl_comm, tensor);
    // nccl_allreduce only enqueues; sync here so the ack means "collective complete".
    ggml_backend_synchronize(nccl_backend);
    response.ok = rc ? 1 : 0;
    return true;
}

// per layer q k v o gate up down router attn_norm ffn_norm q_norm k_norm k_cache v_cache, then output out_norm
static constexpr uint32_t RPC_MK_PER_LAYER = 14;

bool rpc_server::mk_init(const std::vector<uint8_t> & input, rpc_msg_mk_init_rsp & response) {
    memset(&response, 0, sizeof(response));
    auto fail = [&](const char * why) {
        snprintf(response.err, sizeof(response.err), "%s", why);
        return true;
    };
    if (input.size() < sizeof(rpc_msg_mk_init_hdr)) {
        return false;
    }
    rpc_msg_mk_init_hdr hdr;
    memcpy(&hdr, input.data(), sizeof(hdr));
    if (hdr.n_layers <= 0 || hdr.n_tensors != (uint32_t) hdr.n_layers * RPC_MK_PER_LAYER + 2 ||
        input.size() != sizeof(hdr) + (size_t) hdr.n_tensors * sizeof(rpc_tensor) || hdr.device >= backends.size()) {
        return false;
    }
#ifndef _WIN32
    if (mk_create_fn == nullptr) {
        const char * lib = std::getenv("GGML_RPC_MK_LIB");
        void * h = lib ? dlopen(lib, RTLD_NOW | RTLD_LOCAL) : nullptr;
        if (h == nullptr) {
            return fail(lib ? dlerror() : "GGML_RPC_MK_LIB is not set");
        }
        mk_create_fn = (void * (*)(const mk_tp_model *, char *, size_t)) dlsym(h, "mk_tp_create");
        mk_decode_fn = (int (*)(void *, int32_t, int32_t, const void *, float *)) dlsym(h, "mk_tp_decode");
        mk_free_fn   = (void (*)(void *)) dlsym(h, "mk_tp_free");
        if (!mk_create_fn || !mk_decode_fn || !mk_free_fn) {
            mk_create_fn = nullptr;
            return fail("GGML_RPC_MK_LIB does not export mk_tp_create / mk_tp_decode / mk_tp_free");
        }
    }
#else
    return fail("not supported on Windows yet");
#endif
    if (mk != nullptr) {
        mk_free_fn(mk);
        mk = nullptr;
    }
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * hdr.n_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    const rpc_tensor * rt = (const rpc_tensor *) (input.data() + sizeof(hdr));
    std::vector<mk_tp_tensor> t(hdr.n_tensors);
    for (uint32_t i = 0; i < hdr.n_tensors; i++) {
        ggml_tensor * x = deserialize_tensor(ctx_ptr.get(), &rt[i]);
        if (x == nullptr || x->buffer == nullptr || x->data == nullptr) {
            return fail("a tensor is not in a buffer of this connection");
        }
        t[i] = { x->data, (int32_t) x->type, { x->ne[0], x->ne[1], x->ne[2] } };
    }
    std::vector<mk_tp_layer> layers(hdr.n_layers);
    for (int32_t il = 0; il < hdr.n_layers; il++) {
        const mk_tp_tensor * l = &t[(size_t) il * RPC_MK_PER_LAYER];
        layers[il] = { l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11],
                       const_cast<void *>(l[12].data), const_cast<void *>(l[13].data) };
    }
    const mk_tp_tensor * g = &t[(size_t) hdr.n_layers * RPC_MK_PER_LAYER];
    mk_tp_model m = {};
    m.rank      = hdr.rank;
    m.n_ranks   = hdr.n_ranks;
    m.n_layers  = hdr.n_layers;
    m.n_cells   = hdr.n_cells;
    m.vocab_off = hdr.vocab_off;
    m.layers    = layers.data();
    m.tok_embd  = { nullptr, hdr.emb_type, { hdr.emb_ne0, hdr.emb_ne1, 1 } };
    m.output    = g[0];
    m.out_norm  = g[1];
    m.peers     = std::getenv("GGML_RPC_MK_PEERS");
    if (m.peers == nullptr) {
        return fail("GGML_RPC_MK_PEERS is not set");
    }
    mk_backend = backends[hdr.device];
    // the weights were written through this backend: let that finish before copying them
    ggml_backend_synchronize(mk_backend);
    char err[256] = "";
    mk = mk_create_fn(&m, err, sizeof(err));
    if (mk == nullptr) {
        return fail(err[0] ? err : "mk_tp_create failed");
    }
    mk_rows     = (int32_t) m.output.ne[1];
    response.ok = 1;
    return true;
}

bool rpc_server::mk_decode(const std::vector<uint8_t> & input, std::vector<uint8_t> & response) {
    response.assign(sizeof(int32_t) + (size_t) mk_rows * sizeof(float), 0);
    int32_t status = 1;
    rpc_msg_mk_decode_req request;
    if (input.size() >= sizeof(request)) {
        memcpy(&request, input.data(), sizeof(request));
    }
    if (mk != nullptr && input.size() == sizeof(request) + request.emb_bytes) {
        // the KV cache is written by this backend's graphs (prompt processing)
        ggml_backend_synchronize(mk_backend);
        status = mk_decode_fn(mk, request.token, request.pos, request.emb_bytes ? input.data() + sizeof(request) : nullptr,
                              (float *) (response.data() + sizeof(int32_t)));
    }
    memcpy(response.data(), &status, sizeof(status));
    return true;
}

rpc_server::~rpc_server() {
    if (mk != nullptr && mk_free_fn != nullptr) {
        mk_free_fn(mk);
    }
    if (nccl_comm != nullptr && nccl_comm_free != nullptr) {
        nccl_comm_free(nccl_comm);
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                const int64_t t_recv0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
                if (!recv_msg(sock, input)) {
                    return;
                }
                const int64_t t_recv1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
                LOG_LOAD("[rpc-load] server recv cmd=SET_TENSOR input=%zu recv_us=%" PRId64 "\n",
                         input.size(), t_recv1 - t_recv0);
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                const int64_t t_recv0 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
                if (!recv_msg(sock, input)) {
                    return;
                }
                const int64_t t_recv1 = RPC_LOAD_TRACE ? ggml_time_us() : 0;
                LOG_LOAD("[rpc-load] server recv cmd=SET_TENSOR_2D input=%zu recv_us=%" PRId64 "\n",
                         input.size(), t_recv1 - t_recv0);
                if (!server.set_tensor_2d(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D_HASH: {
                rpc_msg_set_tensor_2d_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_2d_hash_rsp response;
                if (!server.set_tensor_2d_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_GET_ID: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_comm_get_id_rsp response;
                if (!server.comm_get_id(response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLREDUCE: {
                rpc_msg_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_allreduce_rsp response;
                if (!server.allreduce(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE_SEQ: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                rpc_msg_graph_compute_seq_rsp response;
                if (!server.graph_compute_seq(input, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_MK_INIT: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                rpc_msg_mk_init_rsp response;
                if (!server.mk_init(input, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_MK_DECODE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                std::vector<uint8_t> response;
                server.mk_decode(input, response);
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// ---------------------------------------------------------------------------
// Orchestrator-side multi-node collective.  The meta backend (tensor parallel)
// calls these through the RPC reg's proc-address table.  Each "rank" is one RPC
// server holding one GPU slice; the actual RCCL allreduce runs on the servers
// over the configured net plugin (e.g. OdinLink/USB4).  Gated by GGML_RPC_RCCL
// so default RPC behaviour (butterfly-over-TCP) is unchanged.
// ---------------------------------------------------------------------------
struct rpc_comm_ctx {
    std::vector<std::shared_ptr<socket_t>> socks;
    std::vector<uint32_t>                  devices; // remote device id per rank, for graph serialization
    std::unordered_set<uint64_t>           warmed_uids; // graph uids already cached on the servers
};

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2) {
        return nullptr;
    }
    auto * cc = new rpc_comm_ctx;
    cc->socks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            GGML_LOG_ERROR("[RPC comm] backend %zu is not an RPC backend\n", i);
            delete cc;
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        auto sock = get_socket(rpc_ctx->endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[RPC comm] failed to get socket for %s\n", rpc_ctx->endpoint.c_str());
            delete cc;
            return nullptr;
        }
        cc->socks.push_back(sock);
        cc->devices.push_back(rpc_ctx->device);
    }

    // rank 0 generates the unique id, all ranks init concurrently
    rpc_msg_comm_get_id_rsp id_rsp;
    if (!send_rpc_cmd(cc->socks[0], RPC_CMD_COMM_GET_ID, nullptr, 0, &id_rsp, sizeof(id_rsp))) {
        GGML_LOG_ERROR("[RPC comm] COMM_GET_ID failed\n");
        delete cc;
        return nullptr;
    }

    const size_t n = cc->socks.size();
    std::vector<uint8_t> ok(n, 0);
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (size_t i = 0; i < n; i++) {
        threads.emplace_back([&, i]() {
            rpc_msg_comm_init_req req = {};
            req.nranks = (uint32_t) n;
            req.rank   = (uint32_t) i;
            ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
            req.device = rpc_ctx->device;
            memcpy(req.unique_id, id_rsp.unique_id, sizeof(req.unique_id));
            rpc_msg_comm_init_rsp rsp = {};
            if (send_rpc_cmd(cc->socks[i], RPC_CMD_COMM_INIT, &req, sizeof(req), &rsp, sizeof(rsp))) {
                ok[i] = rsp.ok;
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    for (size_t i = 0; i < n; i++) {
        if (!ok[i]) {
            GGML_LOG_ERROR("[RPC comm] COMM_INIT failed on rank %zu\n", i);
            delete cc;
            return nullptr;
        }
    }
    GGML_LOG_INFO("[RPC comm] RCCL communicator initialized across %zu ranks\n", n);
    return cc;
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx, struct ggml_tensor ** tensors) {
    if (comm_ctx == nullptr) {
        return false;
    }
    auto * cc = static_cast<rpc_comm_ctx *>(comm_ctx);
    const size_t n = cc->socks.size();

    // Send ALLREDUCE concurrently to all servers, then wait for every ack. The wait is
    // load-bearing: it keeps the 4 servers in lockstep so the k-th ncclAllReduce on each
    // rank pairs up. Pure fire-and-forget lets the orchestrator race ahead until a socket's
    // TCP buffer fills, which desyncs the ranks and deadlocks the blocking collective.
    std::vector<uint8_t> ok(n, 0);
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (size_t i = 0; i < n; i++) {
        threads.emplace_back([&, i]() {
            rpc_msg_allreduce_req req = {};
            req.tensor = serialize_tensor(tensors[i]);
            rpc_msg_allreduce_rsp rsp = {};
            if (send_rpc_cmd(cc->socks[i], RPC_CMD_ALLREDUCE, &req, sizeof(req), &rsp, sizeof(rsp))) {
                ok[i] = rsp.ok;
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    for (size_t i = 0; i < n; i++) {
        if (!ok[i]) {
            return false;
        }
    }
    return true;
}

// Batched per-token dispatch: build one GRAPH_COMPUTE_SEQ message per server containing the
// full ordered sequence of (subgraph, optional allreduce), send concurrently, wait for all acks.
// Each server runs its sequence autonomously (one host sync, one ack), so the orchestrator pays
// a single round-trip per token instead of one per subgraph, and the RCCL ranks stay in lockstep
// because every server executes the same fixed sequence.
// cgraphs/ar_nodes are [n_backends * n_subgraphs] row-major by backend.
static bool ggml_backend_rpc_comm_exec_sequence(void * comm_ctx, size_t n_backends, size_t n_subgraphs,
                                                struct ggml_cgraph ** cgraphs, struct ggml_tensor ** ar_nodes) {
    if (comm_ctx == nullptr) {
        return false;
    }
    auto * cc = static_cast<rpc_comm_ctx *>(comm_ctx);
    if (cc->socks.size() != n_backends) {
        return false;
    }

    // Serialize one SEQ blob per backend. Per step: uid(8) | has_graph(1) | [gsize+blob] | has_ar(1) | [ar]
    std::vector<std::vector<uint8_t>> msgs(n_backends);
    for (size_t j = 0; j < n_backends; j++) {
        std::vector<uint8_t> & buf = msgs[j];
        uint32_t device  = cc->devices[j];
        uint32_t n_steps = (uint32_t) n_subgraphs;
        buf.insert(buf.end(), (uint8_t*)&device,  (uint8_t*)&device  + sizeof(device));
        buf.insert(buf.end(), (uint8_t*)&n_steps, (uint8_t*)&n_steps + sizeof(n_steps));
        for (size_t i = 0; i < n_subgraphs; i++) {
            ggml_cgraph * cg = cgraphs[j*n_subgraphs + i];
            ggml_tensor * ar = ar_nodes[j*n_subgraphs + i];
            uint64_t uid = cg->uid;
            uint8_t has_graph = cc->warmed_uids.count(uid) ? 0 : 1;
            uint8_t has_ar = ar != nullptr ? 1 : 0;
            buf.insert(buf.end(), (uint8_t*)&uid, (uint8_t*)&uid + sizeof(uid));
            buf.push_back(has_graph);
            if (has_graph) {
                std::vector<uint8_t> gblob;
                serialize_graph(device, cg, gblob);
                uint32_t gsize = (uint32_t) gblob.size();
                buf.insert(buf.end(), (uint8_t*)&gsize, (uint8_t*)&gsize + sizeof(gsize));
                buf.insert(buf.end(), gblob.begin(), gblob.end());
            }
            buf.push_back(has_ar);
            if (has_ar) {
                rpc_tensor art = serialize_tensor(ar);
                buf.insert(buf.end(), (uint8_t*)&art, (uint8_t*)&art + sizeof(art));
            }
        }
    }

    // Send concurrently and wait for all acks (one round-trip per token, total).
    std::vector<uint8_t> ok(n_backends, 0);
    std::vector<std::thread> threads;
    threads.reserve(n_backends);
    for (size_t j = 0; j < n_backends; j++) {
        threads.emplace_back([&, j]() {
            rpc_msg_graph_compute_seq_rsp rsp = {};
            if (send_rpc_cmd(cc->socks[j], RPC_CMD_GRAPH_COMPUTE_SEQ, msgs[j].data(), msgs[j].size(), &rsp, sizeof(rsp))) {
                ok[j] = rsp.ok;
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    for (size_t j = 0; j < n_backends; j++) {
        if (ok[j] != 1) {
            // ok==2 (uid not cached on a server) or 0 (send failed): fall back to per-subgraph path.
            // Do not mark uids warmed so the next attempt re-sends the graph blobs.
            return false;
        }
    }
    // all servers ran the full sequence; their graph caches now hold every (rank, subgraph) uid
    for (size_t k = 0; k < n_backends * n_subgraphs; k++) {
        cc->warmed_uids.insert(cgraphs[k]->uid);
    }
    return true;
}

static void ggml_backend_rpc_comm_free(void * comm_ctx) {
    delete static_cast<rpc_comm_ctx *>(comm_ctx);
}

// llama.cpp's tensor-split decode kernel on the servers (RPC_CMD_MK_*). Every server is called at once: they
// connect to each other while creating the kernel and exchange partial sums while decoding.
// tensors: [n_backends][count], each backend's own slices in the order of RPC_MK_PER_LAYER;
// hdr: [n_backends][8] = rank, n_ranks, n_layers, n_cells, vocab_off, emb_type, emb_ne0, emb_ne1.
static bool ggml_backend_rpc_mk_init(ggml_backend_t * backends, size_t n_backends, struct ggml_tensor * const * tensors,
                                     size_t count, const int32_t * hdr, char * err, size_t err_len) {
    std::vector<rpc_msg_mk_init_rsp> rsp(n_backends);
    std::vector<uint8_t> sent(n_backends, 0);
    std::vector<std::thread> threads;
    for (size_t j = 0; j < n_backends; j++) {
        if (!ggml_backend_is_rpc(backends[j])) {
            snprintf(err, err_len, "device %zu is not an RPC device", j);
            return false;
        }
    }
    for (size_t j = 0; j < n_backends; j++) {
        threads.emplace_back([&, j]() {
            ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *) backends[j]->context;
            auto sock = get_socket(ctx->endpoint);
            if (sock == nullptr) {
                return;
            }
            const int32_t * hj = hdr + 8*j;
            rpc_msg_mk_init_hdr h = { ctx->device, hj[0], hj[1], hj[2], hj[3], hj[4], hj[5], hj[6], hj[7], (uint32_t) count };
            std::vector<uint8_t> input(sizeof(h) + count * sizeof(rpc_tensor));
            memcpy(input.data(), &h, sizeof(h));
            for (size_t i = 0; i < count; i++) {
                rpc_tensor t = serialize_tensor(tensors[j*count + i]);
                memcpy(input.data() + sizeof(h) + i * sizeof(rpc_tensor), &t, sizeof(t));
            }
            sent[j] = send_rpc_cmd(sock, RPC_CMD_MK_INIT, input.data(), input.size(), &rsp[j], sizeof(rsp[j]));
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    for (size_t j = 0; j < n_backends; j++) {
        if (!sent[j] || !rsp[j].ok) {
            snprintf(err, err_len, "device %zu: %s", j, sent[j] ? rsp[j].err : "no response");
            return false;
        }
    }
    return true;
}

// One token on every server at once, with its token_embd row; logits[j] receives rows[j] floats (backend j's
// LM head rows).
static bool ggml_backend_rpc_mk_decode(ggml_backend_t * backends, size_t n_backends, int32_t token, int32_t pos,
                                       const void * emb_row, size_t emb_bytes, float * const * logits, const int32_t * rows) {
    std::vector<uint8_t> input(sizeof(rpc_msg_mk_decode_req) + emb_bytes);
    const rpc_msg_mk_decode_req req = { token, pos, (uint32_t) emb_bytes };
    memcpy(input.data(), &req, sizeof(req));
    if (emb_bytes) {
        memcpy(input.data() + sizeof(req), emb_row, emb_bytes);
    }
    std::vector<uint8_t> ok(n_backends, 0);
    std::vector<std::thread> threads;
    for (size_t j = 0; j < n_backends; j++) {
        threads.emplace_back([&, j]() {
            ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *) backends[j]->context;
            auto sock = get_socket(ctx->endpoint);
            if (sock == nullptr) {
                return;
            }
            std::vector<uint8_t> out(sizeof(int32_t) + (size_t) rows[j] * sizeof(float));
            if (!send_rpc_cmd(sock, RPC_CMD_MK_DECODE, input.data(), input.size(), out.data(), out.size())) {
                return;
            }
            int32_t status;
            memcpy(&status, out.data(), sizeof(status));
            if (status == 0 && logits[j]) {
                memcpy(logits[j], out.data() + sizeof(int32_t), (size_t) rows[j] * sizeof(float));
            }
            ok[j] = status == 0;
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    return std::all_of(ok.begin(), ok.end(), [](uint8_t b) { return b != 0; });
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_mk_init") == 0) {
        return (void *)ggml_backend_rpc_mk_init;
    }
    if (std::strcmp(name, "ggml_backend_rpc_mk_decode") == 0) {
        return (void *)ggml_backend_rpc_mk_decode;
    }
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    // multi-node RCCL collective for tensor parallelism, opt-in via env
    static const bool rpc_rccl = []() {
        const char * e = std::getenv("GGML_RPC_RCCL");
        return e && std::atoi(e) != 0;
    }();
    if (rpc_rccl) {
        if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
            return (void *)ggml_backend_rpc_comm_init;
        }
        if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
            return (void *)ggml_backend_rpc_comm_allreduce_tensor;
        }
        if (std::strcmp(name, "ggml_backend_comm_exec_sequence") == 0) {
            return (void *)ggml_backend_rpc_comm_exec_sequence;
        }
        if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
            return (void *)ggml_backend_rpc_comm_free;
        }
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
