#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-rpc-graph-metadata.h"
#include "ggml-rpc-execution-scope.h"
#include "ggml-rpc-sequence-timing.h"
#include "ggml-rpc-execution-caps.h"
#include "transport.h"

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
    RPC_CMD_GRAPH_COMPUTE_SEQ_EXT,
    RPC_CMD_QUERY_EXECUTION_CAPS,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");
static_assert(RPC_CMD_GRAPH_COMPUTE_SEQ_EXT == 23 && RPC_CMD_QUERY_EXECUTION_CAPS == 24, "append-only RPC commands");
static_assert(GGML_RPC_EXECUTION_EXTENSION_PROTOCOL_VERSION == RPC_EXECUTION_CAPS_PROTOCOL, "execution packet version remains 2");

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

static_assert(RPC_CONN_CAPS_SIZE == 24, "transport QPN/PSN/GID capability bytes stay intact");
static_assert(sizeof(rpc_msg_hello_req) == 24 && sizeof(rpc_msg_hello_rsp) == 28, "HELLO wire layout unchanged");

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

struct rpc_msg_graph_compute_seq_ext_header {
    uint32_t version;
    uint32_t plan;
};

struct rpc_msg_graph_compute_seq_ext_rsp {
    uint8_t ok;
    uint64_t custom_stages;
};

static_assert(sizeof(rpc_msg_graph_compute_seq_ext_rsp) == 9, "extension response wire size changed");

struct rpc_msg_graph_metadata_header {
    uint64_t fingerprint;
    uint64_t allreduce_id;
    uint32_t n_counts; // zero reuses validated metadata for this graph UID
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
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size,
                         rpc_sequence_timing::scope * timing = nullptr) {
    if (timing) timing->next(rpc_sequence_timing::send);
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    if (timing) timing->next(rpc_sequence_timing::response_wait);
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
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, rpc_execution_capabilities & caps) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};
    sock->get_caps(request.conn_caps);
    if (!send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response))) {
        GGML_LOG_ERROR("RPC HELLO failed or returned a malformed frame\n");
        return false;
    }
    if (!rpc_execution_peer_version_supported(response.major, response.minor,
                                               RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION)) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n", response.major, response.minor, response.patch);
        return false;
    }
    sock->update_caps(response.conn_caps); // Complete transport activation before the framed optional query.
    caps = rpc_execution_caps_from_legacy(response.major, response.minor, response.patch, response.padding);
    if (!rpc_execution_caps_discover(caps, [&](const rpc_msg_execution_caps_req & req, rpc_msg_execution_caps_rsp & rsp) {
            return send_rpc_cmd(sock, RPC_CMD_QUERY_EXECUTION_CAPS, &req, sizeof(req), &rsp, sizeof(rsp));
        })) {
        GGML_LOG_ERROR("RPC execution capability query failed or returned invalid schema; closing connection\n");
        return false;
    }
    if (caps.query == rpc_execution_query_state::queried_negative) {
        GGML_LOG_WARN("RPC peer %u.%u.%u declined execution capability schema %u (status %u); stock operation remains available\n",
                      unsigned(caps.major), unsigned(caps.minor), unsigned(caps.patch), RPC_EXECUTION_CAPS_SCHEMA, caps.query_status);
    }
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint, rpc_execution_capabilities * extension_caps = nullptr) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    struct cached_socket {
        std::weak_ptr<socket_t> socket;
        rpc_execution_capabilities extension_caps;
    };
    static std::unordered_map<std::string, cached_socket> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.socket.lock()) {
            if (extension_caps != nullptr) {
                *extension_caps = it->second.extension_caps;
            }
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
    rpc_execution_capabilities caps;
    if (!negotiate_hello(sock, caps)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = {sock, caps};
    if (extension_caps != nullptr) {
        *extension_caps = caps;
    }
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
        if (rpc_sequence_timing::enabled()) sequence_timing = std::make_unique<rpc_sequence_timing::collector>();
    }
    ~rpc_server();

    rpc_execution_capabilities execution_capabilities();
    void hello(rpc_msg_hello_rsp & response);
    void query_execution_caps(const rpc_msg_execution_caps_req & request, rpc_msg_execution_caps_rsp & response);
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
    bool graph_compute_seq(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_rsp & response,
                           const char * plan = nullptr, uint64_t * custom_stages = nullptr, size_t offset = 0,
                           rpc_sequence_timing::scope * timing = nullptr);
    bool graph_compute_seq_ext(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_ext_rsp & response,
                               rpc_sequence_timing::scope * timing = nullptr);
    rpc_sequence_timing::collector * sequence_timer() { return sequence_timing.get(); }

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph = nullptr;
        std::vector<uint64_t> node_ids;
        std::unique_ptr<rpc_graph_metadata> extension_metadata;
        uint64_t extension_metadata_hash = 0;
        uint64_t extension_allreduce_id = 0;
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
    uint32_t rank = 0;
    bool decode_graph_dumped = false;
    std::unique_ptr<rpc_sequence_timing::collector> sequence_timing;
};

rpc_execution_capabilities rpc_server::execution_capabilities() {
    rpc_execution_capabilities caps;
    caps.execution_protocol = RPC_EXECUTION_CAPS_PROTOCOL;
    caps.features = backends.empty() ? 0 : RPC_EXECUTION_KNOWN_FEATURES;
    caps.plans = backends.empty() ? 0 : RPC_EXECUTION_KNOWN_PLANS;
    for (auto backend : backends) {
        auto dev = backend ? ggml_backend_get_device(backend) : nullptr;
        auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg || !ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_begin") ||
            !ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_end") ||
            !ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_cancel_v1")) {
            caps.features = caps.plans = 0;
            break;
        }
        auto supports = (ggml_backend_execution_extension_supports_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_supports");
        if (!supports) {
            caps.features = caps.plans = 0;
            break;
        }
        const bool lifecycle = ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_prepare_v1") &&
                               ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_seal_v1");
        if (!lifecycle) caps.features &= ~RPC_EXECUTION_FEATURE_PREPARE_SEAL;
        for (uint32_t plan = 1; plan <= RPC_EXECUTION_PLAN_MAX; ++plan) {
            if ((!lifecycle && rpc_execution_plan_requires_lifecycle(plan)) || !supports(backend, rpc_execution_plan_name(plan))) {
                caps.plans &= ~rpc_execution_plan_bit(plan);
            }
        }
    }
    return caps;
}

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    const auto caps = execution_capabilities();
    response.padding = rpc_execution_caps_legacy_projection(caps.features, caps.plans);
    // conn_caps remains owned and populated entirely by the transport handshake.
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

void rpc_server::query_execution_caps(const rpc_msg_execution_caps_req & request, rpc_msg_execution_caps_rsp & response) {
    response = {};
    response.magic = RPC_EXECUTION_CAPS_MAGIC;
    response.schema_version = RPC_EXECUTION_CAPS_SCHEMA;
    if (request.schema_version != RPC_EXECUTION_CAPS_SCHEMA) {
        response.status = RPC_EXECUTION_CAPS_UNSUPPORTED_SCHEMA;
        return;
    }
    if (request.reserved != 0) {
        response.status = RPC_EXECUTION_CAPS_BAD_REQUEST;
        return;
    }
    const auto caps = execution_capabilities();
    response.status = RPC_EXECUTION_CAPS_OK;
    response.execution_protocol = caps.execution_protocol;
    response.feature_flags = caps.features;
    response.supported_plans = caps.plans;
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

static rpc_execution_extension_hooks rpc_execution_extension_get_hooks(ggml_backend_t backend) {
    rpc_execution_extension_hooks hooks;
    auto dev = backend ? ggml_backend_get_device(backend) : nullptr;
    auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return hooks;
    hooks.begin = (ggml_backend_execution_extension_begin_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_begin");
    hooks.end = (ggml_backend_execution_extension_end_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_end");
    hooks.prepare = (ggml_backend_execution_extension_prepare_v1_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_prepare_v1");
    hooks.seal = (ggml_backend_execution_extension_seal_v1_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_seal_v1");
    hooks.cancel = (ggml_backend_execution_extension_cancel_v1_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_execution_extension_cancel_v1");
    return hooks;
}

static std::vector<const ggml_tensor *> rpc_graph_tensors(const ggml_cgraph * graph) {
    std::vector<const ggml_tensor *> tensors;
    std::unordered_set<const ggml_tensor *> seen;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] != nullptr && seen.insert(graph->nodes[i]).second) {
            tensors.push_back(graph->nodes[i]);
        }
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * tensor = tensors[i];
        for (const auto * src : tensor->src) {
            if (src != nullptr && seen.insert(src).second) {
                tensors.push_back(src);
            }
        }
        if (tensor->view_src != nullptr && seen.insert(tensor->view_src).second) {
            tensors.push_back(tensor->view_src);
        }
    }
    return tensors;
}

static bool rpc_graph_is_decode(const ggml_cgraph * graph) {
    for (const auto * tensor : rpc_graph_tensors(graph)) {
        if (strcmp(tensor->name, "inp_tokens") == 0 && ggml_nelements(tensor) == 1) {
            return true;
        }
        if (strncmp(tensor->name, "attn_norm-", 10) == 0 &&
            tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            return true;
        }
    }
    return false;
}

static void rpc_dump_graph(uint32_t rank, uint32_t step, uint64_t uid, const ggml_cgraph * graph) {
    fprintf(stderr, "[RPC decode graph] rank=%u step=%u uid=%" PRIu64 " nodes=%d\n", rank, step, uid, graph->n_nodes);
    for (const auto * t : rpc_graph_tensors(graph)) {
        fprintf(stderr, "[RPC decode tensor] id=%p data=%p op=%s name=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                "] nb=[%zu,%zu,%zu,%zu] view=%p view_offs=%zu src=[", (const void *) t,
                t->data, ggml_op_name(t->op), t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3], (const void *) t->view_src, t->view_offs);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            fprintf(stderr, "%s%p", i == 0 ? "" : ",", (const void *) t->src[i]);
        }
        fprintf(stderr, "]\n");
    }
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
bool rpc_server::graph_compute_seq(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_rsp & response,
                                   const char * plan, uint64_t * custom_stages, size_t offset,
                                   rpc_sequence_timing::scope * timing) {
    if (timing) {
        timing->next(rpc_sequence_timing::setup);
        timing->data().plan = rpc_execution_plan_id(plan);
    }
    response.ok = 0;
    if (offset > input.size()) {
        return false;
    }
    const uint8_t * p   = input.data() + offset;
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
    if (timing) timing->data().sequence_graphs = n_steps;

    rpc_execution_extension_scope extension_scope;
    if (plan != nullptr && !extension_scope.begin(backends[device], plan,
            rpc_execution_extension_get_hooks(backends[device]),
            rpc_execution_plan_requires_lifecycle(rpc_execution_plan_id(plan)))) {
        GGML_LOG_ERROR("[%s] rank %u rejected execution plan %s\n", __func__, rank, plan);
        return false;
    }

    static const bool dump_enabled = [] {
        const char * value = std::getenv("GGML_RPC_DUMP_DECODE_GRAPH");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    bool dump_sequence = false;

    auto & cache = uid_graphs[device];
    // transient contexts for allreduce tensors (graphs live in the persistent cache)
    std::vector<ggml_context_ptr> ar_ctxs;
    struct execution_step {
        ggml_cgraph * graph;
        ggml_tensor * allreduce;
    };
    std::vector<execution_step> planned_steps;
    std::unordered_set<uint64_t> planned_uids;
    if (plan != nullptr) {
        constexpr size_t min_step_bytes = sizeof(uint64_t) + 2 + sizeof(rpc_msg_graph_metadata_header);
        if (n_steps == 0 || n_steps > size_t(end - p) / min_step_bytes) return false;
        planned_steps.reserve(n_steps);
    }

    if (timing) timing->next(plan ? rpc_sequence_timing::parse : rpc_sequence_timing::stock_parse_enqueue);
    for (uint32_t s = 0; s < n_steps; s++) {
        if (end - p < (ptrdiff_t)(sizeof(uint64_t) + 1)) {
            return false;
        }
        uint64_t uid;
        memcpy(&uid, p, sizeof(uid)); p += sizeof(uid);
        if (plan != nullptr && (uid == 0 || !planned_uids.insert(uid).second)) return false;
        uint8_t has_graph = *p; p += 1;
        if (timing) {
            ++timing->data().graphs_seen;
            timing->data().graph_payloads += has_graph != 0;
        }

        ggml_cgraph * graph = nullptr;
        stored_graph * graph_record = nullptr;
        if (has_graph) {
            if (end - p < (ptrdiff_t)sizeof(uint32_t)) {
                return false;
            }
            uint32_t gsize;
            memcpy(&gsize, p, sizeof(gsize)); p += sizeof(gsize);
            if (end - p < (ptrdiff_t)gsize || gsize < 3*sizeof(uint32_t)) {
                return false;
            }
            const uint8_t * gp = p; p += gsize;
            gp += sizeof(uint32_t); // skip embedded device
            uint32_t n_nodes;
            memcpy(&n_nodes, gp, sizeof(n_nodes)); gp += sizeof(n_nodes);
            if (n_nodes > (gsize - 3*sizeof(uint32_t))/sizeof(uint64_t)) {
                return false;
            }
            const uint64_t * nodes = (const uint64_t *)gp;
            gp += (size_t)n_nodes*sizeof(uint64_t);
            uint32_t n_tensors;
            memcpy(&n_tensors, gp, sizeof(n_tensors)); gp += sizeof(n_tensors);
            const size_t graph_header_size = 3*sizeof(uint32_t) + (size_t) n_nodes*sizeof(uint64_t);
            if ((size_t) n_tensors*sizeof(rpc_tensor) != gsize - graph_header_size) {
                return false;
            }
            const rpc_tensor * tensors = (const rpc_tensor *)gp;

            // build into a persistent per-uid context so the graph survives across tokens
            stored_graph & sg = cache[uid];
            sg.extension_metadata.reset();
            sg.node_ids.resize(n_nodes);
            memcpy(sg.node_ids.data(), nodes, size_t(n_nodes) * sizeof(uint64_t));
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
            graph_record = &sg;
        } else {
            auto it = cache.find(uid);
            if (it == cache.end() || it->second.graph == nullptr) {
                // not cached: ask orchestrator to warm up this token via the normal path
                response.ok = 2;
                return true;
            }
            graph = it->second.graph;
            graph_record = &it->second;
        }

        rpc_msg_graph_metadata_header metadata_header{};
        if (plan != nullptr) {
            if (end - p < ptrdiff_t(sizeof(metadata_header))) return false;
            memcpy(&metadata_header, p, sizeof(metadata_header));
            p += sizeof(metadata_header);
            if (graph_record->node_ids.size() != size_t(graph->n_nodes)) return false;
            if (metadata_header.n_counts) {
                if (timing) ++timing->data().metadata_payloads;
                if (metadata_header.n_counts != uint32_t(graph->n_nodes) ||
                    size_t(metadata_header.n_counts) > size_t(end - p) / sizeof(int32_t)) return false;
                std::vector<int32_t> counts(metadata_header.n_counts);
                memcpy(counts.data(), p, counts.size() * sizeof(int32_t));
                p += counts.size() * sizeof(int32_t);
                if (rpc_graph_metadata_hash(graph_record->node_ids.data(), counts.data(), counts.size(),
                                            metadata_header.allreduce_id) != metadata_header.fingerprint) return false;
                const ggml_tensor * metadata_allreduce = nullptr;
                if (metadata_header.allreduce_id) {
                    const auto found = std::find(graph_record->node_ids.begin(), graph_record->node_ids.end(),
                                                 metadata_header.allreduce_id);
                    if (found == graph_record->node_ids.end()) return false;
                    metadata_allreduce = graph->nodes[found - graph_record->node_ids.begin()];
                }
                auto metadata = std::make_unique<rpc_graph_metadata>();
                if (!metadata->initialize(graph, counts.data(), counts.size(), metadata_allreduce)) return false;
                graph_record->extension_metadata = std::move(metadata);
                graph_record->extension_metadata_hash = metadata_header.fingerprint;
                graph_record->extension_allreduce_id = metadata_header.allreduce_id;
            } else if (has_graph || !graph_record->extension_metadata ||
                       graph_record->extension_metadata_hash != metadata_header.fingerprint ||
                       graph_record->extension_allreduce_id != metadata_header.allreduce_id) {
                return false;
            }
            graph = graph_record->extension_metadata->view();
            if (!graph) return false;
        }

        if (dump_enabled && !decode_graph_dumped && n_steps > 1 && s == 0) {
            dump_sequence = rpc_graph_is_decode(graph);
            if (dump_sequence) {
                fprintf(stderr, "[RPC decode graph] rank=%u steps=%u plan=%s\n", rank, n_steps, plan ? plan : "stock");
            }
        }
        if (dump_sequence) {
            rpc_dump_graph(rank, s, uid, graph);
        }

        if (plan == nullptr) {
            const ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
            if (status != GGML_STATUS_SUCCESS) {
                return false;
            }
        }

        ggml_tensor * ar_tensor = nullptr;
        if (end - p < (ptrdiff_t)1) {
            return false;
        }
        uint8_t has_ar = *p; p += 1;
        if (timing) timing->data().allreduces_seen += has_ar != 0;
        if (plan != nullptr && bool(has_ar) != (metadata_header.allreduce_id != 0)) return false;
        if (has_ar) {
            if (nccl_comm == nullptr || nccl_backend == nullptr) {
                return false;
            }
            if (end - p < (ptrdiff_t)sizeof(rpc_tensor)) {
                return false;
            }
            const rpc_tensor * ar_rt = (const rpc_tensor *)p;
            p += sizeof(rpc_tensor);
            if (plan != nullptr && ar_rt->id != metadata_header.allreduce_id) return false;
            ggml_init_params arp = { ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr arctx { ggml_init(arp) };
            ar_tensor = deserialize_tensor(arctx.get(), ar_rt);
            if (ar_tensor == nullptr) {
                return false;
            }
            ar_ctxs.push_back(std::move(arctx));
            if (plan == nullptr && !nccl_allreduce(nccl_backend, nccl_comm, ar_tensor)) {
                return false;
            }
        }
        if (plan != nullptr) {
            planned_steps.push_back({graph, ar_tensor});
        }
    }

    if (plan != nullptr) {
        if (p != end) {
            return false;
        }
        if (timing) timing->next(rpc_sequence_timing::prepare);
        // Validate and bind the complete local sequence before its first submission.
        std::vector<ggml_cgraph *> graphs;
        std::vector<ggml_tensor *> allreduce_operands;
        graphs.reserve(planned_steps.size());
        allreduce_operands.reserve(planned_steps.size());
        for (const auto & step : planned_steps) {
            graphs.push_back(step.graph);
            allreduce_operands.push_back(step.allreduce);
        }
        if (!extension_scope.prepare(graphs.size(), graphs.data(), allreduce_operands.data())) {
            GGML_LOG_ERROR("[%s] rank %u rejected preparation for plan %s\n", __func__, rank, plan);
            return false;
        }
        if (timing) timing->next(rpc_sequence_timing::enqueue);
        for (const auto & step : planned_steps) {
            if (ggml_backend_graph_compute_async(backends[device], step.graph) != GGML_STATUS_SUCCESS ||
                (step.allreduce != nullptr && !nccl_allreduce(nccl_backend, nccl_comm, step.allreduce))) {
                return false;
            }
        }
    }

    // Seal may queue a final status copy. Even if that enqueue fails, retain
    // the existing token synchronization after the complete collective sequence.
    if (timing) timing->next(rpc_sequence_timing::complete);
    if (!extension_scope.complete([&]() {
            if (timing) timing->measure(rpc_sequence_timing::sync, [&]() { ggml_backend_synchronize(backends[device]); });
            else ggml_backend_synchronize(backends[device]);
        }, custom_stages)) {
        // The token completed synchronization; report failure without closing
        // this connection while the other ranks complete their acknowledgements.
        return true;
    }
    if (dump_sequence) {
        decode_graph_dumped = true;
        fflush(stderr);
    }
    response.ok = 1;
    return true;
}

bool rpc_server::graph_compute_seq_ext(const std::vector<uint8_t> & input, rpc_msg_graph_compute_seq_ext_rsp & response,
                                      rpc_sequence_timing::scope * timing) {
    if (timing) timing->next(rpc_sequence_timing::setup);
    response = {};
    if (input.size() < sizeof(rpc_msg_graph_compute_seq_ext_header)) {
        return false;
    }
    rpc_msg_graph_compute_seq_ext_header header;
    memcpy(&header, input.data(), sizeof(header));
    if (timing) timing->data().plan = header.plan;
    const char * plan_name = rpc_execution_plan_name(header.plan);
    if (header.version != GGML_RPC_EXECUTION_EXTENSION_PROTOCOL_VERSION || plan_name == nullptr) {
        return false;
    }

    rpc_msg_graph_compute_seq_rsp legacy_response = {};
    uint64_t custom_stages = 0;
    const bool ok = graph_compute_seq(input, legacy_response, plan_name,
                                      &custom_stages, sizeof(header), timing);
    if (ok && legacy_response.ok == 1) {
        response.ok = 1;
        response.custom_stages = custom_stages;
    } else {
        GGML_LOG_ERROR("[%s] rank %u execution failed; stock retry is disabled\n", __func__, rank);
    }
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
    rank = request.rank;
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

rpc_server::~rpc_server() {
    if (sequence_timing) sequence_timing->emit("worker", rank);
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
            case RPC_CMD_QUERY_EXECUTION_CAPS: {
                rpc_msg_execution_caps_req request{};
                if (!recv_msg(sock, &request, sizeof(request))) return;
                rpc_msg_execution_caps_rsp response{};
                server.query_execution_caps(request, response);
                if (!send_msg(sock, &response, sizeof(response))) return;
                break;
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
                rpc_sequence_timing::sample sample;
                auto * collector = server.sequence_timer();
                rpc_sequence_timing::scope scope(sample, collector, collector != nullptr);
                auto * timing = collector ? &scope : nullptr;
                if (timing) timing->next(rpc_sequence_timing::receive);
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (timing) sample.bytes = input.size();
                rpc_msg_graph_compute_seq_rsp response;
                if (!server.graph_compute_seq(input, response, nullptr, nullptr, 0, timing)) {
                    return;
                }
                if (timing) timing->next(rpc_sequence_timing::reply);
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                if (timing) sample.result = response.ok == 1 ? rpc_sequence_timing::success :
                    response.ok == 2 ? rpc_sequence_timing::cache_miss : rpc_sequence_timing::failed;
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE_SEQ_EXT: {
                rpc_sequence_timing::sample sample;
                sample.plan = UINT32_MAX; // unknown until the extension header is validated
                auto * collector = server.sequence_timer();
                rpc_sequence_timing::scope scope(sample, collector, collector != nullptr);
                auto * timing = collector ? &scope : nullptr;
                if (timing) timing->next(rpc_sequence_timing::receive);
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (timing) sample.bytes = input.size();
                rpc_msg_graph_compute_seq_ext_rsp response = {};
                if (!server.graph_compute_seq_ext(input, response, timing)) {
                    return;
                }
                if (timing) timing->next(rpc_sequence_timing::reply);
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                if (timing) sample.result = response.ok == 1 ? rpc_sequence_timing::success : rpc_sequence_timing::failed;
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
    rpc_comm_ctx() {
        if (rpc_sequence_timing::enabled()) sequence_timing = std::make_unique<rpc_sequence_timing::collector>();
    }
    ~rpc_comm_ctx() {
        if (sequence_timing) sequence_timing->emit("client", -1);
        for (size_t j = 0; j < rank_timing.size(); ++j) rank_timing[j]->emit("client_rank", int64_t(j));
    }
    std::vector<std::shared_ptr<socket_t>> socks;
    std::vector<uint32_t>                  devices; // remote device id per rank, for graph serialization
    std::unordered_set<uint64_t>           warmed_uids; // graph uids already cached on the servers
    std::unordered_map<uint64_t, uint64_t>  warmed_metadata;
    uint64_t extension_features = RPC_EXECUTION_KNOWN_FEATURES;
    uint64_t extension_plans = RPC_EXECUTION_KNOWN_PLANS;
    std::vector<rpc_execution_capabilities> rank_capabilities;
    bool extension_counts_reported = false;
    std::unique_ptr<rpc_sequence_timing::collector> sequence_timing;
    std::vector<std::unique_ptr<rpc_sequence_timing::collector>> rank_timing;
    std::vector<rpc_sequence_timing::sample> rank_samples;
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
        rpc_execution_capabilities extension_caps;
        auto sock = get_socket(rpc_ctx->endpoint, &extension_caps);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[RPC comm] failed to get socket for %s\n", rpc_ctx->endpoint.c_str());
            delete cc;
            return nullptr;
        }
        cc->socks.push_back(sock);
        cc->devices.push_back(rpc_ctx->device);
        cc->extension_features &= extension_caps.features;
        cc->extension_plans &= extension_caps.plans;
        cc->rank_capabilities.push_back(extension_caps);
        GGML_LOG_DEBUG("[RPC comm] rank %zu execution caps: peer=%u.%u.%u query=%u features=0x%" PRIx64 " plans=0x%" PRIx64 "\n",
                       i, extension_caps.major, extension_caps.minor, extension_caps.patch,
                       unsigned(extension_caps.query), extension_caps.features, extension_caps.plans);
    }

    if (cc->sequence_timing) {
        cc->rank_samples.resize(n_backends);
        for (size_t j = 0; j < n_backends; ++j) cc->rank_timing.emplace_back(std::make_unique<rpc_sequence_timing::collector>());
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
    GGML_LOG_DEBUG("[RPC comm] execution capabilities across all ranks: features=0x%" PRIx64 " plans=0x%" PRIx64 "\n",
                   cc->extension_features, cc->extension_plans);
    return cc;
}

static bool ggml_backend_rpc_comm_execution_extension_supports(void * comm_ctx, const char * plan_name) {
    const auto * cc = static_cast<const rpc_comm_ctx *>(comm_ctx);
    const uint32_t plan = rpc_execution_plan_id(plan_name);
    const uint64_t bit = rpc_execution_plan_bit(plan);
    if (!cc || !bit || !(cc->extension_plans & bit) ||
        !(cc->extension_features & RPC_EXECUTION_FEATURE_METADATA) ||
        cc->rank_capabilities.size() != cc->socks.size()) return false;
    return rpc_execution_all_ranks_support(cc->rank_capabilities.data(), cc->rank_capabilities.size(), plan);
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
static bool ggml_backend_rpc_comm_exec_sequence_impl(void * comm_ctx, size_t n_backends, size_t n_subgraphs,
        struct ggml_cgraph ** cgraphs, struct ggml_tensor ** ar_nodes, uint32_t plan, uint64_t * custom_stages) {
    const bool extension = plan != 0;
    if (custom_stages != nullptr) {
        *custom_stages = 0;
    }
    if (comm_ctx == nullptr) {
        return false;
    }
    auto * cc = static_cast<rpc_comm_ctx *>(comm_ctx);
    rpc_sequence_timing::sample timing_sample;
    timing_sample.plan = plan;
    timing_sample.sequence_graphs = n_subgraphs;
    const bool timing_enabled = cc->sequence_timing != nullptr;
    rpc_sequence_timing::scope timing_scope(timing_sample, cc->sequence_timing.get(), timing_enabled);
    auto * timing = timing_enabled ? &timing_scope : nullptr;
    if (cc->socks.size() != n_backends) {
        return false;
    }
    if (extension && (!n_backends || !n_subgraphs || n_subgraphs > UINT32_MAX || !cgraphs || !ar_nodes)) {
        return false;
    }
    if (timing) {
        for (auto & value : cc->rank_samples) {
            value = {};
            value.plan = plan;
            value.sequence_graphs = n_subgraphs;
        }
        timing->next(rpc_sequence_timing::build);
    }

    // The extension adds whole-graph use metadata before each has_ar field.
    // Build and validate every rank's message before any sequence is submitted.
    std::vector<std::vector<uint8_t>> msgs(n_backends);
    std::vector<uint64_t> stage_limits(extension ? n_backends : 0, 0);
    std::vector<int32_t> metadata_counts;
    std::unordered_map<uint64_t, uint64_t> submitted_metadata;
    for (size_t j = 0; j < n_backends; j++) {
        std::vector<uint8_t> & buf = msgs[j];
        if (extension) {
            const rpc_msg_graph_compute_seq_ext_header header = {
                GGML_RPC_EXECUTION_EXTENSION_PROTOCOL_VERSION, plan,
            };
            const auto * bytes = reinterpret_cast<const uint8_t *>(&header);
            buf.insert(buf.end(), bytes, bytes + sizeof(header));
        }
        uint32_t device  = cc->devices[j];
        uint32_t n_steps = (uint32_t) n_subgraphs;
        buf.insert(buf.end(), (uint8_t*)&device,  (uint8_t*)&device  + sizeof(device));
        buf.insert(buf.end(), (uint8_t*)&n_steps, (uint8_t*)&n_steps + sizeof(n_steps));
        for (size_t i = 0; i < n_subgraphs; i++) {
            ggml_cgraph * cg = cgraphs[j*n_subgraphs + i];
            ggml_tensor * ar = ar_nodes[j*n_subgraphs + i];
            if (!cg) return false;
            uint64_t uid = cg->uid;
            uint8_t has_graph = cc->warmed_uids.count(uid) ? 0 : 1;
            uint8_t has_ar = ar != nullptr ? 1 : 0;
            if (timing) {
                ++timing_sample.graphs_seen;
                ++cc->rank_samples[j].graphs_seen;
                timing_sample.graph_payloads += has_graph;
                cc->rank_samples[j].graph_payloads += has_graph;
                timing_sample.allreduces_seen += has_ar;
                cc->rank_samples[j].allreduces_seen += has_ar;
            }
            buf.insert(buf.end(), (uint8_t*)&uid, (uint8_t*)&uid + sizeof(uid));
            buf.push_back(has_graph);
            if (has_graph) {
                std::vector<uint8_t> gblob;
                serialize_graph(device, cg, gblob);
                uint32_t gsize = (uint32_t) gblob.size();
                buf.insert(buf.end(), (uint8_t*)&gsize, (uint8_t*)&gsize + sizeof(gsize));
                buf.insert(buf.end(), gblob.begin(), gblob.end());
            }
            if (extension) {
                uint64_t fingerprint;
                if (!rpc_graph_metadata_fingerprint(cg, ar, fingerprint)) return false;
                stage_limits[j] += cg->n_nodes;
                const uint64_t allreduce_id = reinterpret_cast<uintptr_t>(ar);
                const auto cached = cc->warmed_metadata.find(uid);
                const bool send_counts = has_graph || cached == cc->warmed_metadata.end() || cached->second != fingerprint;
                if (timing) {
                    timing_sample.metadata_payloads += send_counts;
                    cc->rank_samples[j].metadata_payloads += send_counts;
                }
                if (send_counts && !rpc_graph_metadata_collect(cg, ar, metadata_counts)) return false;
                const rpc_msg_graph_metadata_header header = {
                    fingerprint, allreduce_id, send_counts ? uint32_t(metadata_counts.size()) : 0,
                };
                const auto * bytes = reinterpret_cast<const uint8_t *>(&header);
                buf.insert(buf.end(), bytes, bytes + sizeof(header));
                if (send_counts) {
                    bytes = reinterpret_cast<const uint8_t *>(metadata_counts.data());
                    buf.insert(buf.end(), bytes, bytes + metadata_counts.size() * sizeof(int32_t));
                }
                if (uid == 0 || !submitted_metadata.emplace(uid, fingerprint).second) return false;
            }
            buf.push_back(has_ar);
            if (has_ar) {
                rpc_tensor art = serialize_tensor(ar);
                buf.insert(buf.end(), (uint8_t*)&art, (uint8_t*)&art + sizeof(art));
            }
        }
        if (timing) {
            timing_sample.bytes += buf.size();
            cc->rank_samples[j].bytes = buf.size();
        }
    }

    // Send concurrently and wait for all acks (one round-trip per token, total).
    if (timing) timing->next(rpc_sequence_timing::dispatch_join);
    std::vector<uint8_t> ok(n_backends, 0);
    std::vector<uint64_t> custom_counts(extension ? n_backends : 0, 0);
    std::vector<std::thread> threads;
    threads.reserve(n_backends);
    for (size_t j = 0; j < n_backends; j++) {
        threads.emplace_back([&, j]() {
            std::optional<rpc_sequence_timing::scope> rank_scope;
            auto * rank_timer = timing ? &rank_scope.emplace(cc->rank_samples[j], nullptr, true) : nullptr;
            if (extension) {
                rpc_msg_graph_compute_seq_ext_rsp rsp = {};
                if (send_rpc_cmd(cc->socks[j], RPC_CMD_GRAPH_COMPUTE_SEQ_EXT, msgs[j].data(), msgs[j].size(), &rsp, sizeof(rsp), rank_timer)) {
                    ok[j] = rsp.ok;
                    custom_counts[j] = rsp.custom_stages;
                }
            } else {
                rpc_msg_graph_compute_seq_rsp rsp = {};
                if (send_rpc_cmd(cc->socks[j], RPC_CMD_GRAPH_COMPUTE_SEQ, msgs[j].data(), msgs[j].size(), &rsp, sizeof(rsp), rank_timer)) {
                    ok[j] = rsp.ok;
                }
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    if (timing) timing->next(rpc_sequence_timing::commit);
    bool all_ok = true;
    uint64_t total_custom_stages = 0;
    for (size_t j = 0; j < n_backends; j++) {
        const bool valid_count = !extension ||
            (custom_counts[j] <= stage_limits[j] &&
             (ok[j] == 1 || custom_counts[j] == 0) &&
             (plan != RPC_EXECUTION_PLAN_QWEN3MOE_METADATA_ONLY || custom_counts[j] == 0) &&
             custom_counts[j] <= UINT64_MAX - total_custom_stages);
        if (timing) {
            cc->rank_samples[j].result = ok[j] == 1 && valid_count ? rpc_sequence_timing::success :
                !extension && ok[j] == 2 ? rpc_sequence_timing::cache_miss : rpc_sequence_timing::failed;
            cc->rank_timing[j]->record(cc->rank_samples[j]);
        }
        if (ok[j] != 1 || !valid_count) {
            if (extension) {
                GGML_LOG_ERROR("[RPC comm] execution extension failed on rank %zu (ok=%u, custom_stages=%" PRIu64
                               "); token cannot be retried\n", j, unsigned(ok[j]), custom_counts[j]);
            }
            all_ok = false;
        } else if (extension) {
            total_custom_stages += custom_counts[j];
        }
    }
    // Only the legacy caller may retry. Extension failures propagate to decode().
    if (!all_ok) {
        if (timing && !extension && std::find(ok.begin(), ok.end(), uint8_t(2)) != ok.end())
            timing_sample.result = rpc_sequence_timing::cache_miss;
        return false;
    }
    // all servers ran the full sequence; their graph caches now hold every (rank, subgraph) uid
    for (size_t k = 0; k < n_backends * n_subgraphs; k++) {
        cc->warmed_uids.insert(cgraphs[k]->uid);
    }
    for (const auto & item : submitted_metadata) cc->warmed_metadata[item.first] = item.second;
    if (custom_stages != nullptr) {
        *custom_stages = total_custom_stages;
    }
    if (extension && !cc->extension_counts_reported) {
        const char * stats_env = std::getenv("LLAMA_EXECUTION_EXTENSION_STATS");
        const bool direct_stats = stats_env != nullptr && strcmp(stats_env, "1") == 0;
        for (size_t j = 0; j < n_backends; ++j) {
            if (direct_stats) {
                fprintf(stderr, "[RPC comm] first extension sequence: rank=%zu custom_stages=%" PRIu64
                                " plan=%s metadata_graphs=%zu\n", j, custom_counts[j], rpc_execution_plan_name(plan), n_subgraphs);
            } else {
                GGML_LOG_INFO("[RPC comm] first extension sequence: rank=%zu custom_stages=%" PRIu64
                              " plan=%s metadata_graphs=%zu\n", j, custom_counts[j], rpc_execution_plan_name(plan), n_subgraphs);
            }
        }
        if (direct_stats) {
            fflush(stderr);
        }
        cc->extension_counts_reported = true;
    }
    if (timing) timing_sample.result = rpc_sequence_timing::success;
    return true;
}

static bool ggml_backend_rpc_comm_exec_sequence(void * comm_ctx, size_t n_backends, size_t n_subgraphs,
                                                struct ggml_cgraph ** cgraphs, struct ggml_tensor ** ar_nodes) {
    return ggml_backend_rpc_comm_exec_sequence_impl(comm_ctx, n_backends, n_subgraphs, cgraphs, ar_nodes, 0, nullptr);
}

static ggml_status ggml_backend_rpc_comm_exec_sequence_ext(void * comm_ctx, const char * plan_name,
        size_t n_backends, size_t n_subgraphs, struct ggml_cgraph ** cgraphs, struct ggml_tensor ** ar_nodes,
        uint64_t * custom_stages) {
    if (!ggml_backend_rpc_comm_execution_extension_supports(comm_ctx, plan_name)) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_rpc_comm_exec_sequence_impl(comm_ctx, n_backends, n_subgraphs, cgraphs, ar_nodes,
            rpc_execution_plan_id(plan_name), custom_stages)
        ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static void ggml_backend_rpc_comm_free(void * comm_ctx) {
    delete static_cast<rpc_comm_ctx *>(comm_ctx);
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
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
        if (std::strcmp(name, "ggml_backend_comm_execution_extension_supports") == 0) {
            return (void *)ggml_backend_rpc_comm_execution_extension_supports;
        }
        if (std::strcmp(name, "ggml_backend_comm_exec_sequence_ext") == 0) {
            return (void *)ggml_backend_rpc_comm_exec_sequence_ext;
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
