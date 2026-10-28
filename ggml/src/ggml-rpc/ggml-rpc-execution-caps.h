#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

// Legacy HELLO bits are deliberately not plan-ID bit positions.
static constexpr uint8_t RPC_CAP_QWEN3MOE_OPTIMIZED = 1 << 1;
static constexpr uint8_t RPC_CAP_QWEN3MOE_GATE_UP_REFERENCE = 1 << 2;
static constexpr uint8_t RPC_CAP_EXTENSION_GRAPH_METADATA = 1 << 3;
static constexpr uint8_t RPC_CAP_QWEN3MOE_METADATA_ONLY = 1 << 4;
static constexpr uint8_t RPC_CAP_QWEN3MOE_COMBINED_STAGES = 1 << 5;
static constexpr uint8_t RPC_CAP_QWEN3MOE_FULL_MOE = 1 << 6;
static constexpr uint8_t RPC_CAP_QWEN3MOE_FULL_MOE_Q8_256 = 1 << 7;
static constexpr uint8_t RPC_CAP_QWEN3MOE_FULL_MOE_Q8_128 = 1 << 0;

static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_DECODE = 1;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_GATE_UP_REFERENCE = 2;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_METADATA_ONLY = 3;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_COMBINED_STAGES = 4;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE = 5;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_256 = 6;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_128 = 7;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128 = 8;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_Q8_128 = 9;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128_DIRECT = 10;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_Q8_128 = 11;
static constexpr uint32_t RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_O_Q8_128 = 12;
static constexpr uint32_t RPC_EXECUTION_PLAN_MAX = RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_O_Q8_128;
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_DECODE_NAME = "qwen3moe.decode.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_GATE_UP_REFERENCE_NAME = "qwen3moe.decode.gate-up-reference.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_METADATA_ONLY_NAME = "qwen3moe.decode.metadata-only.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_COMBINED_STAGES_NAME = "qwen3moe.decode.combined-stages.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_NAME = "qwen3moe.decode.full-moe.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_256_NAME = "qwen3moe.decode.full-moe-q8-256.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_128_NAME = "qwen3moe.decode.full-moe-q8-128.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128_NAME = "qwen3moe.decode.full-ffn-q8-128.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_Q8_128_NAME = "qwen3moe.decode.split-moe-q8-128.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128_DIRECT_NAME = "qwen3moe.decode.full-ffn-q8-128-direct.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_Q8_128_NAME = "qwen3moe.decode.split-moe-qkv-prefix-q8-128.v1";
static constexpr const char * RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_O_Q8_128_NAME = "qwen3moe.decode.split-moe-qkv-prefix-o-q8-128.v1";

static inline const char * rpc_execution_plan_name(uint32_t plan) {
    switch (plan) {
        case 1: return RPC_EXECUTION_PLAN_QWEN3MOE_DECODE_NAME;
        case 2: return RPC_EXECUTION_PLAN_QWEN3MOE_GATE_UP_REFERENCE_NAME;
        case 3: return RPC_EXECUTION_PLAN_QWEN3MOE_METADATA_ONLY_NAME;
        case 4: return RPC_EXECUTION_PLAN_QWEN3MOE_COMBINED_STAGES_NAME;
        case 5: return RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_NAME;
        case 6: return RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_256_NAME;
        case 7: return RPC_EXECUTION_PLAN_QWEN3MOE_FULL_MOE_Q8_128_NAME;
        case 8: return RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128_NAME;
        case 9: return RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_Q8_128_NAME;
        case 10: return RPC_EXECUTION_PLAN_QWEN3MOE_FULL_FFN_Q8_128_DIRECT_NAME;
        case 11: return RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_Q8_128_NAME;
        case 12: return RPC_EXECUTION_PLAN_QWEN3MOE_SPLIT_MOE_QKV_PREFIX_O_Q8_128_NAME;
        default: return nullptr;
    }
}

static inline uint32_t rpc_execution_plan_id(const char * name) {
    if (name) for (uint32_t plan = 1; plan <= RPC_EXECUTION_PLAN_MAX; ++plan) {
        if (std::strcmp(name, rpc_execution_plan_name(plan)) == 0) return plan;
    }
    return 0;
}

static inline uint8_t rpc_execution_plan_capability(uint32_t plan) {
    switch (plan) {
        case 1: return RPC_CAP_QWEN3MOE_OPTIMIZED;
        case 2: return RPC_CAP_QWEN3MOE_GATE_UP_REFERENCE;
        case 3: return RPC_CAP_QWEN3MOE_METADATA_ONLY;
        case 4: return RPC_CAP_QWEN3MOE_COMBINED_STAGES;
        case 5: return RPC_CAP_QWEN3MOE_FULL_MOE;
        case 6: return RPC_CAP_QWEN3MOE_FULL_MOE_Q8_256;
        case 7: return RPC_CAP_QWEN3MOE_FULL_MOE_Q8_128;
        default: return 0; // Plans 8 and later have no legacy HELLO bit.
    }
}

static constexpr uint64_t RPC_EXECUTION_FEATURE_METADATA = UINT64_C(1) << 0;
static constexpr uint64_t RPC_EXECUTION_FEATURE_PREPARE_SEAL = UINT64_C(1) << 1;
static constexpr uint64_t RPC_EXECUTION_KNOWN_FEATURES = RPC_EXECUTION_FEATURE_METADATA | RPC_EXECUTION_FEATURE_PREPARE_SEAL;
static constexpr uint64_t RPC_EXECUTION_KNOWN_PLANS = UINT64_C(0x1ffe); // IDs 1..12; bit 0 reserved.
static constexpr uint64_t RPC_EXECUTION_LIFECYCLE_PLANS = UINT64_C(0x1fe0); // IDs 5..12.
static constexpr uint64_t RPC_EXECUTION_WIDE_PLANS = UINT64_C(0x1f00); // IDs 8..12 require queried support.
static constexpr uint32_t RPC_EXECUTION_CAPS_MAGIC = UINT32_C(0x31435845); // EXC1
static constexpr uint32_t RPC_EXECUTION_CAPS_SCHEMA = 1;
static constexpr uint32_t RPC_EXECUTION_CAPS_PROTOCOL = 2;
static constexpr uint8_t RPC_EXECUTION_CAPS_QUERY_MINOR = 2;
static constexpr uint8_t RPC_EXECUTION_CAPS_QUERY_MAJOR = 4;

enum rpc_execution_caps_status : uint32_t {
    RPC_EXECUTION_CAPS_OK = 0,
    RPC_EXECUTION_CAPS_UNSUPPORTED_SCHEMA = 1,
    RPC_EXECUTION_CAPS_BAD_REQUEST = 2,
};

#pragma pack(push, 1)
struct rpc_msg_execution_caps_req { uint32_t schema_version; uint32_t reserved; };
struct rpc_msg_execution_caps_rsp {
    uint32_t magic;
    uint32_t schema_version;
    uint32_t status;
    uint32_t execution_protocol;
    uint64_t feature_flags;
    uint64_t supported_plans;
};
#pragma pack(pop)
static_assert(sizeof(rpc_msg_execution_caps_req) == 8, "execution query request layout");
static_assert(sizeof(rpc_msg_execution_caps_rsp) == 32, "execution query response layout");

enum class rpc_execution_query_state { legacy_only, queried_ok, queried_negative };
struct rpc_execution_capabilities {
    uint8_t major = 0, minor = 0, patch = 0, legacy = 0;
    rpc_execution_query_state query = rpc_execution_query_state::legacy_only;
    uint32_t query_status = RPC_EXECUTION_CAPS_OK;
    uint32_t execution_protocol = 0;
    uint64_t features = 0, plans = 0;
};

static inline uint64_t rpc_execution_plan_bit(uint32_t plan) {
    return plan > 0 && plan < 64 ? UINT64_C(1) << plan : 0;
}
static inline bool rpc_execution_plan_requires_lifecycle(uint32_t plan) {
    return (rpc_execution_plan_bit(plan) & RPC_EXECUTION_LIFECYCLE_PLANS) != 0;
}
static inline bool rpc_execution_peer_version_supported(uint8_t major, uint8_t minor,
                                                        uint8_t client_major, uint8_t client_minor) {
    return major == client_major && minor <= client_minor;
}

static inline rpc_execution_capabilities rpc_execution_caps_from_legacy(uint8_t major, uint8_t minor,
                                                                        uint8_t patch, uint8_t legacy) {
    rpc_execution_capabilities caps;
    caps.major = major; caps.minor = minor; caps.patch = patch; caps.legacy = legacy;
    if (legacy & RPC_CAP_EXTENSION_GRAPH_METADATA) {
        caps.execution_protocol = RPC_EXECUTION_CAPS_PROTOCOL;
        caps.features |= RPC_EXECUTION_FEATURE_METADATA;
    }
    for (uint32_t plan = 1; plan <= 7; ++plan) {
        if (legacy & rpc_execution_plan_capability(plan)) caps.plans |= rpc_execution_plan_bit(plan);
    }
    // Legacy full-MoE advertisement already required both prepare/seal hooks.
    if (caps.plans & RPC_EXECUTION_LIFECYCLE_PLANS) caps.features |= RPC_EXECUTION_FEATURE_PREPARE_SEAL;
    return caps;
}

static inline bool rpc_execution_caps_apply_reply(rpc_execution_capabilities & caps,
                                                   const rpc_msg_execution_caps_rsp & response) {
    if (response.magic != RPC_EXECUTION_CAPS_MAGIC || response.schema_version != RPC_EXECUTION_CAPS_SCHEMA ||
        response.status > RPC_EXECUTION_CAPS_BAD_REQUEST) return false;
    rpc_execution_capabilities next = caps;
    next.query_status = response.status;
    next.features = 0; next.plans = 0; next.execution_protocol = 0;
    if (response.status != RPC_EXECUTION_CAPS_OK) {
        if (response.feature_flags || response.supported_plans || response.execution_protocol) return false;
        next.query = rpc_execution_query_state::queried_negative;
    } else {
        if (response.execution_protocol != RPC_EXECUTION_CAPS_PROTOCOL || (response.supported_plans & 1)) return false;
        next.query = rpc_execution_query_state::queried_ok;
        next.execution_protocol = response.execution_protocol;
        next.features = response.feature_flags & RPC_EXECUTION_KNOWN_FEATURES;
        next.plans = response.supported_plans & RPC_EXECUTION_KNOWN_PLANS;
        if (next.plans && !(next.features & RPC_EXECUTION_FEATURE_METADATA)) return false;
        if ((next.plans & RPC_EXECUTION_LIFECYCLE_PLANS) && !(next.features & RPC_EXECUTION_FEATURE_PREPARE_SEAL)) return false;
    }
    caps = next; // Wide reply is authoritative, including a valid zero mask.
    return true;
}

template<class Query>
static bool rpc_execution_caps_discover(rpc_execution_capabilities & caps, Query query) {
    if (caps.major != RPC_EXECUTION_CAPS_QUERY_MAJOR) return false;
    if (caps.query != rpc_execution_query_state::legacy_only) return true;
    if (caps.minor < RPC_EXECUTION_CAPS_QUERY_MINOR) return true;
    const rpc_msg_execution_caps_req request{RPC_EXECUTION_CAPS_SCHEMA, 0};
    rpc_msg_execution_caps_rsp response{};
    return query(request, response) && rpc_execution_caps_apply_reply(caps, response);
}

static inline bool rpc_execution_caps_supports(const rpc_execution_capabilities & caps, uint32_t plan) {
    const uint64_t bit = rpc_execution_plan_bit(plan);
    if (caps.major != RPC_EXECUTION_CAPS_QUERY_MAJOR || caps.query == rpc_execution_query_state::queried_negative ||
        !(bit & RPC_EXECUTION_KNOWN_PLANS) || !(caps.plans & bit) ||
        caps.execution_protocol != RPC_EXECUTION_CAPS_PROTOCOL || !(caps.features & RPC_EXECUTION_FEATURE_METADATA)) return false;
    if (rpc_execution_plan_requires_lifecycle(plan) && !(caps.features & RPC_EXECUTION_FEATURE_PREPARE_SEAL)) return false;
    return !(bit & RPC_EXECUTION_WIDE_PLANS) ||
           (caps.minor >= RPC_EXECUTION_CAPS_QUERY_MINOR && caps.query == rpc_execution_query_state::queried_ok);
}

static inline uint8_t rpc_execution_caps_legacy_projection(uint64_t features, uint64_t plans) {
    uint8_t legacy = (features & RPC_EXECUTION_FEATURE_METADATA) ? RPC_CAP_EXTENSION_GRAPH_METADATA : 0;
    for (uint32_t plan = 1; plan <= 7; ++plan) {
        if (plans & rpc_execution_plan_bit(plan)) legacy |= rpc_execution_plan_capability(plan);
    }
    return legacy;
}

static inline bool rpc_execution_all_ranks_support(const rpc_execution_capabilities * ranks,
                                                    size_t count, uint32_t plan) {
    if (!ranks || count == 0) return false;
    for (size_t i = 0; i < count; ++i) if (!rpc_execution_caps_supports(ranks[i], plan)) return false;
    return true;
}
