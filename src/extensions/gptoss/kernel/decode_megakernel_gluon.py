from collections import namedtuple
from pathlib import Path

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.language import core


GRID_SYNC_LL = str(Path(__file__).with_name("grid_sync.ll"))

HIDDEN = 2880
INTERMEDIATE = 2880
PADDED = 3072
Q_HEADS = 64
KV_HEADS = 8
HEAD = 64
QUERY = Q_HEADS * HEAD
KV = KV_HEADS * HEAD
QKV = QUERY + 2 * KV
EXPERTS = 32
EXPERTS_USED = 4
BLOCK_K = 32
K_BLOCKS = HIDDEN // BLOCK_K
PADDED_BLOCKS = PADDED // BLOCK_K
WARPS = 8
GRID_SIZE = 120
RMS_PARTIALS = GRID_SIZE
ROWS_PER_WAVE = 3
ROWS_PER_CTA = ROWS_PER_WAVE * WARPS
GATE_OUT_BLOCKS = INTERMEDIATE // ROWS_PER_CTA
GATE_TASKS = EXPERTS_USED * GATE_OUT_BLOCKS
QKV_TASKS = Q_HEADS * 16 + KV_HEADS * 16 + KV // 4
QKV_VALUES_BYTES = QKV * HIDDEN
ATTN_OUT_VALUES_BYTES = HIDDEN * QUERY
MOE_DOWN_VALUES_BYTES = EXPERTS * PADDED * PADDED // 2
MOE_GATE_VALUES_BYTES = 2 * MOE_DOWN_VALUES_BYTES
ATTENTION_MAX_OFFSET = 3.0 * 0.6931

BARRIER_COUNTER_COUNT = 4
EXPERT_PADDING_COUNT = EXPERTS_USED * (PADDED - INTERMEDIATE)
BARRIER_COUNTER_FP16_OFFSET = HIDDEN + EXPERTS_USED * PADDED

GL_HIDDEN = gl.constexpr(HIDDEN)
GL_INTERMEDIATE = gl.constexpr(INTERMEDIATE)
GL_PADDED = gl.constexpr(PADDED)
GL_Q_HEADS = gl.constexpr(Q_HEADS)
GL_KV_HEADS = gl.constexpr(KV_HEADS)
GL_HEAD = gl.constexpr(HEAD)
GL_QUERY = gl.constexpr(QUERY)
GL_KV = gl.constexpr(KV)
GL_EXPERTS = gl.constexpr(EXPERTS)
GL_EXPERTS_USED = gl.constexpr(EXPERTS_USED)
GL_K_BLOCKS = gl.constexpr(K_BLOCKS)
GL_PADDED_BLOCKS = gl.constexpr(PADDED_BLOCKS)
GL_WARPS = gl.constexpr(WARPS)
GL_GRID_SIZE = gl.constexpr(GRID_SIZE)
GL_RMS_PARTIALS = gl.constexpr(RMS_PARTIALS)
GL_ROWS_PER_WAVE = gl.constexpr(ROWS_PER_WAVE)
GL_ROWS_PER_CTA = gl.constexpr(ROWS_PER_CTA)
GL_GATE_OUT_BLOCKS = gl.constexpr(GATE_OUT_BLOCKS)
GL_GATE_TASKS = gl.constexpr(GATE_TASKS)
GL_QKV_TASKS = gl.constexpr(QKV_TASKS)
GL_QKV_VALUES_BYTES = gl.constexpr(QKV_VALUES_BYTES)
GL_ATTN_OUT_VALUES_BYTES = gl.constexpr(ATTN_OUT_VALUES_BYTES)
GL_MOE_DOWN_VALUES_BYTES = gl.constexpr(MOE_DOWN_VALUES_BYTES)
GL_MOE_GATE_VALUES_BYTES = gl.constexpr(MOE_GATE_VALUES_BYTES)
GL_ATTENTION_MAX_OFFSET = gl.constexpr(ATTENTION_MAX_OFFSET)
GL_BARRIER_COUNTER_COUNT = gl.constexpr(BARRIER_COUNTER_COUNT)
GL_EXPERT_PADDING_COUNT = gl.constexpr(EXPERT_PADDING_COUNT)
GL_BARRIER_COUNTER_FP16_OFFSET = gl.constexpr(BARRIER_COUNTER_FP16_OFFSET)


PARAM_FIELDS = (
    "next",
    "cur",
    "rms_partials",
    "activation_scratch",
    "query",
    "attn_parts",
    "attn_meta",
    "router",
    "expert_ids",
    "expert_weights",
    "cache_k",
    "cache_v",
    "kv_rows",
    "attn_norm",
    "qkv_values",
    "attn_q_bias",
    "attn_k_bias",
    "attn_v_bias",
    "attn_output_values",
    "attn_output_bias",
    "attn_sinks",
    "post_attention_norm",
    "router_weight",
    "router_bias",
    "moe_down_values",
    "moe_gate_up_values",
    "moe_down_bias",
    "moe_gate_up_bias",
    "n_kv",
    "kv_write_row",
    "attn_parallel_blocks",
    "position",
    "rms_epsilon",
    "rope_freq_scale",
    "rope_ext_factor",
    "rope_attn_factor",
    "rope_corr_low",
    "rope_corr_high",
    "rope_theta_scale",
    "reuse_attention_rms",
)

ParamTypes = namedtuple("ParamTypes", PARAM_FIELDS)
PARAM_TYPES = ParamTypes(
    "*fp32", "*fp32", "*fp32", "*fp16", "*fp16", "*fp32", "*fp32", "*fp32",
    "*i32", "*fp32", "*fp16", "*fp16", "*i32", "*fp32", "*i8", "*fp32",
    "*fp32", "*fp32", "*i8", "*fp32", "*fp32", "*fp32", "*fp32", "*fp32",
    "*u8", "*u8", "*fp32", "*fp32", "u32", "u32", "u32", "i32", "fp32",
    "fp32", "fp32", "fp32", "fp32", "fp32", "fp32", "u32",
)


@core.extern
def grid_sync(token, _semantic=None):
    return core.extern_elementwise(
        "grid_sync",
        GRID_SYNC_LL,
        [token],
        {(core.dtype("int32"),): ("__ockl_grid_sync_i32", core.dtype("int32"))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def get_local_linear_id(_semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {(): ("__ockl_get_local_linear_id", core.dtype("uint64"))},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def amd_byte_perm(a, b, selector, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [a, b, selector],
        {
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")): (
                "llvm.amdgcn.perm",
                core.dtype("uint32"),
            ),
            (core.dtype("uint32"), core.dtype("int32"), core.dtype("int32")): (
                "llvm.amdgcn.perm",
                core.dtype("uint32"),
            ),
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): (
                "llvm.amdgcn.perm",
                core.dtype("uint32"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def packed_dot2(a, b, acc, clamp, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [a, b, acc, clamp],
        {
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
                core.dtype("fp32"),
                core.dtype("int1"),
            ): ("__ockl_fdot2", core.dtype("fp32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def ocml_pow(a, b, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [a, b],
        {(core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32"))},
        is_pure=True,
        _semantic=_semantic,
    )


@gluon.jit
def decode_mxfp4_word(codes):
    low_magnitude = codes & 0x07070707
    low_sign = (codes & 0x08080808) << 4
    low_bytes = amd_byte_perm(0x46444240, 0x3E3C3800, low_magnitude) | low_sign
    low0 = amd_byte_perm(low_bytes, 0, 0x05010400)
    low1 = amd_byte_perm(low_bytes, 0, 0x07030602)

    high_codes = codes >> 4
    high_magnitude = high_codes & 0x07070707
    high_sign = (high_codes & 0x08080808) << 4
    high_bytes = amd_byte_perm(0x46444240, 0x3E3C3800, high_magnitude) | high_sign
    high0 = amd_byte_perm(high_bytes, 0, 0x05010400)
    high1 = amd_byte_perm(high_bytes, 0, 0x07030602)
    return low0, low1, high0, high1


@gluon.jit
def split4(x):
    shaped = gl.reshape(x, x.shape[0], x.shape[1], 2, 2)
    even, odd = gl.split(shaped)
    x0, x2 = gl.split(even)
    x1, x3 = gl.split(odd)
    return x0, x1, x2, x3


@gluon.jit
def mxfp4_dot_word(codes, activation4, acc, THREAD_LAYOUT: gl.constexpr):
    weight0, weight1, weight2, weight3 = decode_mxfp4_word(codes)
    act0, act1, act2, act3 = split4(activation4)
    act0 = gl.convert_layout(act0, THREAD_LAYOUT)
    act1 = gl.convert_layout(act1, THREAD_LAYOUT)
    act2 = gl.convert_layout(act2, THREAD_LAYOUT)
    act3 = gl.convert_layout(act3, THREAD_LAYOUT)
    acc = packed_dot2(weight0, act0, acc, False)
    acc = packed_dot2(weight1, act1, acc, False)
    acc = packed_dot2(weight2, act2, acc, False)
    return packed_dot2(weight3, act3, acc, False)


@gluon.jit
def mxfp4_dot_block(packed2, activation0, activation1, activation2, activation3, THREAD_LAYOUT: gl.constexpr):
    packed0, packed1 = gl.split(packed2)
    packed0 = gl.convert_layout(packed0, THREAD_LAYOUT)
    packed1 = gl.convert_layout(packed1, THREAD_LAYOUT)
    code0 = packed0.to(gl.uint32)
    code1 = (packed0 >> 32).to(gl.uint32)
    code2 = packed1.to(gl.uint32)
    code3 = (packed1 >> 32).to(gl.uint32)
    acc = mxfp4_dot_word(code0, activation0, 0.0, THREAD_LAYOUT)
    acc = mxfp4_dot_word(code1, activation1, acc, THREAD_LAYOUT)
    acc = mxfp4_dot_word(code2, activation2, acc, THREAD_LAYOUT)
    return mxfp4_dot_word(code3, activation3, acc, THREAD_LAYOUT)


@gluon.jit
def e8m0_scale(scale):
    return gl.maximum(scale.to(gl.uint32) << 23, 0x00400000).to(gl.float32, bitcast=True)


@gluon.jit
def add_packed_f16_offset(bits):
    low = (bits & 0xffff).to(gl.uint16).to(gl.float16, bitcast=True)
    high = (bits >> 16).to(gl.uint16).to(gl.float16, bitcast=True)
    low = (low - 1152.0).to(gl.uint16, bitcast=True).to(gl.uint32)
    high = (high - 1152.0).to(gl.uint16, bitcast=True).to(gl.uint32)
    return low | (high << 16)


@gluon.jit
def q8_word_dot(packed, x0, x1, acc):
    shifted = packed ^ 0x80808080
    low = add_packed_f16_offset(amd_byte_perm(shifted, 0x64646464, 0x00050004))
    high = add_packed_f16_offset(amd_byte_perm(shifted, 0x64646464, 0x00070006))
    acc = packed_dot2(low, x0, acc, False)
    return packed_dot2(high, x1, acc, False)


@gluon.jit
def q8_dot4(values, scales, activation, row0, row1, row2, row3, BLOCKS: gl.constexpr, ROW_WORDS: gl.constexpr):
    weight_layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    activation_layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    thread_layout: gl.constexpr = gl.SliceLayout(2, activation_layout)
    weight_thread_layout: gl.constexpr = gl.SliceLayout(2, weight_layout)
    weight_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, weight_layout))
    activation_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, activation_layout))
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    warp = gl.arange(0, 8, layout=warp_layout)[:, None]
    lane = gl.arange(0, 32, layout=lane_layout)[None, :]
    tid = warp * 32 + lane
    weight_word = gl.expand_dims(gl.expand_dims(gl.arange(0, 2, layout=weight_word_layout), 0), 0)
    activation_word = gl.expand_dims(gl.expand_dims(gl.arange(0, 4, layout=activation_word_layout), 0), 0)
    values_u32 = values.to(gl.pointer_type(gl.uint32), bitcast=True)
    activation_u32 = activation.to(gl.pointer_type(gl.uint32), bitcast=True)
    acc0 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
    acc1 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
    acc2 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
    acc3 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
    for block_round in gl.static_range(0, 2):
        block = tid // 4 + block_round * 64
        segment = tid % 4
        valid = block < BLOCKS
        weight_block = gl.convert_layout(block, weight_thread_layout)
        weight_segment = gl.convert_layout(segment, weight_thread_layout)
        weight_valid = gl.convert_layout(valid, weight_thread_layout)
        word_offset = gl.expand_dims(weight_block * 8 + weight_segment * 2, 2) + weight_word
        x_offset = (block * 16 + segment * 4)[:, :, None] + activation_word
        x = gl.load(activation_u32 + x_offset, mask=valid[:, :, None], other=0)
        x0, x1, x2, x3 = split4(x)
        x0 = gl.convert_layout(x0, thread_layout)
        x1 = gl.convert_layout(x1, thread_layout)
        x2 = gl.convert_layout(x2, thread_layout)
        x3 = gl.convert_layout(x3, thread_layout)
        load_mask = gl.expand_dims(weight_valid, 2)
        packed0 = gl.load(values_u32 + row0 * ROW_WORDS + word_offset, mask=load_mask, other=0, cache_modifier=".cg")
        packed1 = gl.load(values_u32 + row1 * ROW_WORDS + word_offset, mask=load_mask, other=0, cache_modifier=".cg")
        packed2 = gl.load(values_u32 + row2 * ROW_WORDS + word_offset, mask=load_mask, other=0, cache_modifier=".cg")
        packed3 = gl.load(values_u32 + row3 * ROW_WORDS + word_offset, mask=load_mask, other=0, cache_modifier=".cg")
        p00, p01 = gl.split(packed0)
        p10, p11 = gl.split(packed1)
        p20, p21 = gl.split(packed2)
        p30, p31 = gl.split(packed3)
        p00 = gl.convert_layout(p00, thread_layout)
        p01 = gl.convert_layout(p01, thread_layout)
        p10 = gl.convert_layout(p10, thread_layout)
        p11 = gl.convert_layout(p11, thread_layout)
        p20 = gl.convert_layout(p20, thread_layout)
        p21 = gl.convert_layout(p21, thread_layout)
        p30 = gl.convert_layout(p30, thread_layout)
        p31 = gl.convert_layout(p31, thread_layout)
        local0 = q8_word_dot(p00, x0, x1, 0.0)
        local0 = q8_word_dot(p01, x2, x3, local0)
        local1 = q8_word_dot(p10, x0, x1, 0.0)
        local1 = q8_word_dot(p11, x2, x3, local1)
        local2 = q8_word_dot(p20, x0, x1, 0.0)
        local2 = q8_word_dot(p21, x2, x3, local2)
        local3 = q8_word_dot(p30, x0, x1, 0.0)
        local3 = q8_word_dot(p31, x2, x3, local3)
        scale0 = gl.load(scales + row0 * BLOCKS + block, mask=valid, other=0.0).to(gl.float32)
        scale1 = gl.load(scales + row1 * BLOCKS + block, mask=valid, other=0.0).to(gl.float32)
        scale2 = gl.load(scales + row2 * BLOCKS + block, mask=valid, other=0.0).to(gl.float32)
        scale3 = gl.load(scales + row3 * BLOCKS + block, mask=valid, other=0.0).to(gl.float32)
        acc0 += scale0 * local0
        acc1 += scale1 * local1
        acc2 += scale2 * local2
        acc3 += scale3 * local3
    pair01 = gl.join(acc0, acc1)
    pair23 = gl.join(acc2, acc3)
    joined = gl.join(pair01, pair23)
    lanes4 = gl.reshape(gl.permute(joined, 0, 1, 3, 2), 8, 32, 4)
    waves4 = gl.sum(lanes4, axis=1)
    outputs = gl.sum(waves4, axis=0)
    output_layout: gl.constexpr = gl.BlockedLayout([4], [32], [8], [0])
    return gl.convert_layout(outputs, output_layout)


@gluon.jit
def rope_table_pointer(p):
    return (p.activation_scratch + GL_HIDDEN).to(gl.pointer_type(gl.float32), bitcast=True)


@gluon.jit
def attention_rms_norm_and_parallel_rope_table(p):
    pid = gl.program_id(0)
    if pid == 0:
        value_layout: gl.constexpr = gl.BlockedLayout([16], [32], [8], [0])
        offsets = gl.arange(0, 4096, layout=value_layout)
        valid = offsets < GL_HIDDEN
        values = gl.load(p.cur + offsets, mask=valid, other=0.0)
        if p.reuse_attention_rms == 0:
            total = gl.sum(values * values, axis=0)
        else:
            partial_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
            partial = gl.arange(0, 256, layout=partial_layout)
            total = gl.sum(
                gl.load(p.rms_partials + partial, mask=partial < GL_RMS_PARTIALS, other=0.0),
                axis=0,
            )
        scale = gl.rsqrt(total / GL_HIDDEN + p.rms_epsilon)
        norm = gl.load(p.attn_norm + offsets, mask=valid, other=0.0)
        gl.store(p.activation_scratch + offsets, (values * scale * norm).to(gl.float16), mask=valid)

    if pid == 1:
        pair_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
        pair = gl.arange(0, 256, layout=pair_layout)
        pair_valid = pair < GL_HEAD // 2
        theta_extrap = p.position.to(gl.float32) * ocml_pow(p.rope_theta_scale, pair.to(gl.float32))
        theta_interp = p.rope_freq_scale * theta_extrap
        ramp_y = (pair.to(gl.float32) - p.rope_corr_low) / gl.maximum(
            0.001,
            p.rope_corr_high - p.rope_corr_low,
        )
        ramp = 1.0 - gl.minimum(1.0, gl.maximum(0.0, ramp_y))
        mix = ramp * p.rope_ext_factor
        theta = theta_interp * (1.0 - mix) + theta_extrap * mix
        magnitude = p.rope_attn_factor * gl.where(
            p.rope_ext_factor != 0.0,
            1.0 + 0.1 * gl.log(1.0 / p.rope_freq_scale),
            1.0,
        )
        table = rope_table_pointer(p)
        gl.store(table + pair, gl.cos(theta) * magnitude, mask=pair_valid)
        gl.store(table + GL_HEAD // 2 + pair, gl.sin(theta) * magnitude, mask=pair_valid)


@gluon.jit
def rope_pairs_table(p, dots, head, pair0, bias, output, output_base, query_scale):
    pair_layout: gl.constexpr = gl.BlockedLayout([2], [32], [8], [0])
    shaped = gl.reshape(dots, 2, 2)
    x0, x1 = gl.split(shaped)
    x0 = gl.reshape(x0, 2)
    x1 = gl.reshape(x1, 2)
    pair_offset = gl.arange(0, 2, layout=pair_layout)
    x0 = gl.convert_layout(x0, pair_layout)
    x1 = gl.convert_layout(x1, pair_layout)
    pair = pair0 + pair_offset
    low = head * GL_HEAD + pair
    high = low + GL_HEAD // 2
    x0 = (x0 + gl.load(bias + low)).to(gl.float16).to(gl.float32)
    x1 = (x1 + gl.load(bias + high)).to(gl.float16).to(gl.float32)
    table = rope_table_pointer(p)
    cosine = gl.load(table + pair)
    sine = gl.load(table + GL_HEAD // 2 + pair)
    y0 = gl.fma(x0, cosine, -(x1 * sine)) * query_scale
    y1 = gl.fma(x1, cosine, x0 * sine) * query_scale
    gl.store(output + output_base + low, y0.to(gl.float16))
    gl.store(output + output_base + high, y1.to(gl.float16))


@gluon.jit
def qkv_rope_cache(p):
    values = p.qkv_values
    scales = (p.qkv_values + GL_QKV_VALUES_BYTES).to(gl.pointer_type(gl.float16), bitcast=True)
    output_layout: gl.constexpr = gl.BlockedLayout([4], [32], [8], [0])
    output_offset = gl.arange(0, 4, layout=output_layout)
    task = gl.program_id(0)
    while task < GL_QKV_TASKS:
        is_q = task < GL_Q_HEADS * 16
        is_qk = task < (GL_Q_HEADS + GL_KV_HEADS) * 16
        qk_task = gl.where(is_q, task, task - GL_Q_HEADS * 16)
        head = qk_task // 16
        pair0 = (qk_task % 16) * 2
        qk_row0 = gl.where(is_q, 0, GL_QUERY) + head * GL_HEAD + pair0
        value_row0 = GL_QUERY + GL_KV + (task - (GL_Q_HEADS + GL_KV_HEADS) * 16) * 4
        row0 = gl.where(is_qk, qk_row0, value_row0)
        row1 = gl.where(is_qk, qk_row0 + GL_HEAD // 2, value_row0 + 1)
        row2 = gl.where(is_qk, qk_row0 + 1, value_row0 + 2)
        row3 = gl.where(is_qk, qk_row0 + GL_HEAD // 2 + 1, value_row0 + 3)
        dots = q8_dot4(
            values,
            scales,
            p.activation_scratch,
            row0,
            row1,
            row2,
            row3,
            GL_K_BLOCKS,
            GL_HIDDEN // 4,
        )
        if is_qk:
            if is_q:
                rope_pairs_table(p, dots, head, pair0, p.attn_q_bias, p.query, 0, 0.125)
            else:
                rope_pairs_table(
                    p,
                    dots,
                    head,
                    pair0,
                    p.attn_k_bias,
                    p.cache_k,
                    p.kv_write_row * GL_KV,
                    1.0,
                )
        else:
            output_row = (task - (GL_Q_HEADS + GL_KV_HEADS) * 16) * 4
            bias = gl.load(p.attn_v_bias + output_row + output_offset)
            gl.store(
                p.cache_v + p.kv_write_row * GL_KV + output_row + output_offset,
                (dots + bias).to(gl.float16),
            )
        task += GL_GRID_SIZE


@gluon.jit
def sliding_window_attention(p):
    pid = gl.program_id(0)
    thread_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [8, 1], [1, 0])
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    score_pair_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 2], [8, 1, 1], [2, 1, 0])
    score_layout: gl.constexpr = gl.SliceLayout(2, score_pair_layout)
    score_warp_layout: gl.constexpr = gl.SliceLayout(1, score_layout)
    score_key_layout: gl.constexpr = gl.SliceLayout(0, score_layout)
    meta_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    value_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    warp_1d = gl.arange(0, 8, layout=warp_layout)
    lane_1d = gl.arange(0, 32, layout=lane_layout)
    warp = warp_1d[:, None]
    lane = lane_1d[None, :]
    maxima = gl.allocate_shared_memory(gl.float32, [8], meta_layout)
    sums = gl.allocate_shared_memory(gl.float32, [8], meta_layout)
    partials = gl.allocate_shared_memory(gl.float32, [8, 64], value_layout)

    if pid < GL_Q_HEADS:
        query_head = pid + warp_1d * 0
        kv_head = query_head // 8
        key = warp * 16 + lane // 2
        active_pair = key < p.n_kv
        row = gl.load(p.kv_rows + key, mask=active_pair, other=0)
        key_base = row * GL_KV + kv_head[:, None] * GL_HEAD
        cache_k_u32 = p.cache_k.to(gl.pointer_type(gl.uint32), bitcast=True)
        query_u32 = p.query.to(gl.pointer_type(gl.uint32), bitcast=True)
        score_pair = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        for pair in gl.static_range(0, 16):
            pair_index = (lane & 1) * 16 + pair
            query_pair = gl.load(query_u32 + query_head[:, None] * 32 + pair_index)
            key_pair = gl.load(cache_k_u32 + key_base // 2 + pair_index, mask=active_pair, other=0)
            score_pair = packed_dot2(key_pair, query_pair, score_pair, False)

        score = gl.sum(gl.reshape(score_pair, 8, 16, 2), axis=2)
        active = gl.max(gl.reshape(active_pair.to(gl.int32), 8, 16, 2), axis=2) != 0
        key_slot = gl.arange(0, 16, layout=score_key_layout)[None, :]

        local_max = gl.max(
            gl.where(active, score + GL_ATTENTION_MAX_OFFSET, -1.7014117e38),
            axis=1,
        )
        maxima.store(local_max)
        gl.barrier()

        max0 = maxima.gather(warp_1d * 0, axis=0)
        max1 = maxima.gather(warp_1d * 0 + 1, axis=0)
        max2 = maxima.gather(warp_1d * 0 + 2, axis=0)
        max3 = maxima.gather(warp_1d * 0 + 3, axis=0)
        max4 = maxima.gather(warp_1d * 0 + 4, axis=0)
        max5 = maxima.gather(warp_1d * 0 + 5, axis=0)
        max6 = maxima.gather(warp_1d * 0 + 6, axis=0)
        max7 = maxima.gather(warp_1d * 0 + 7, axis=0)
        sink = gl.load(p.attn_sinks + query_head)
        maximum = gl.maximum(
            gl.maximum(gl.maximum(max0, max1), gl.maximum(max2, max3)),
            gl.maximum(gl.maximum(max4, max5), gl.maximum(max6, max7)),
        )
        maximum = gl.maximum(maximum, sink)
        maximum_score = gl.convert_layout(maximum, score_warp_layout)
        weight = gl.where(active, gl.exp(score - maximum_score[:, None]), 0.0)
        local_sum = gl.sum(weight, axis=1)
        sums.store(local_sum)
        gl.barrier()

        sum0 = sums.gather(warp_1d * 0, axis=0)
        sum1 = sums.gather(warp_1d * 0 + 1, axis=0)
        sum2 = sums.gather(warp_1d * 0 + 2, axis=0)
        sum3 = sums.gather(warp_1d * 0 + 3, axis=0)
        sum4 = sums.gather(warp_1d * 0 + 4, axis=0)
        sum5 = sums.gather(warp_1d * 0 + 5, axis=0)
        sum6 = sums.gather(warp_1d * 0 + 6, axis=0)
        sum7 = sums.gather(warp_1d * 0 + 7, axis=0)
        denominator = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + gl.exp(sink - maximum)
        value0 = gl.zeros([8, 32], gl.float16, layout=thread_layout)
        value1 = gl.zeros([8, 32], gl.float16, layout=thread_layout)
        for key_lane in core.range(0, 16, loop_unroll_factor=8):
            logical = warp_1d * 16 + key_lane
            valid = logical < p.n_kv
            value_row = gl.load(p.kv_rows + logical, mask=valid, other=0)
            base = value_row * GL_KV + kv_head * GL_HEAD
            broadcast = gl.sum(gl.where(key_slot == key_lane, weight, 0.0), axis=1)
            broadcast = gl.convert_layout(broadcast, warp_layout).to(gl.float16)
            v0 = gl.load(p.cache_v + base[:, None] + lane, mask=valid[:, None], other=0.0)
            v1 = gl.load(p.cache_v + base[:, None] + lane + 32, mask=valid[:, None], other=0.0)
            value0 = (value0 + v0 * broadcast[:, None]).to(gl.float16)
            value1 = (value1 + v1 * broadcast[:, None]).to(gl.float16)

        partials.slice(0, 32, dim=1).store(value0.to(gl.float32))
        partials.slice(32, 32, dim=1).store(value1.to(gl.float32))
        gl.barrier()

        row_index = warp * 0 + lane * 0
        out0 = partials.slice(0, 32, dim=1).gather(row_index.to(gl.int32), axis=0)
        out1 = partials.slice(32, 32, dim=1).gather(row_index.to(gl.int32), axis=0)
        for part in gl.static_range(1, 8):
            out0 += partials.slice(0, 32, dim=1).gather((row_index + part).to(gl.int32), axis=0)
            out1 += partials.slice(32, 32, dim=1).gather((row_index + part).to(gl.int32), axis=0)
        output_base = query_head[:, None] * GL_HEAD
        writer = warp == 0
        gl.store(p.query + output_base + lane, (out0 / denominator[:, None]).to(gl.float16), mask=writer)
        gl.store(p.query + output_base + lane + 32, (out1 / denominator[:, None]).to(gl.float16), mask=writer)


@gluon.jit
def full_attention_parts_u64(p):
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [8, 1], [1, 0])
    query_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [8, 1], [1, 0])
    compute_layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    thread_layout: gl.constexpr = gl.SliceLayout(2, compute_layout)
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    kv_shared64_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[16, 1]], [32, 16], [1, 0])
    kv_shared32_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[32, 2]], [32, 32], [1, 0])
    score_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    kv_shared = gl.allocate_shared_memory(gl.uint64, [2, 32, 16], kv_shared64_layout)
    query_shared = gl.allocate_shared_memory(gl.uint32, [8, 32], score_shared_layout)
    weights = gl.allocate_shared_memory(gl.float16, [8, 128], score_shared_layout)
    warp_1d = gl.arange(0, 8, layout=warp_layout)
    lane_1d = gl.arange(0, 32, layout=lane_layout)
    warp = warp_1d[:, None]
    lane = lane_1d[None, :]
    copy_key = gl.arange(0, 32, layout=gl.SliceLayout(1, copy_layout))
    copy_quad = gl.arange(0, 16, layout=gl.SliceLayout(0, copy_layout))
    cache_k_u64 = p.cache_k.to(gl.pointer_type(gl.uint64), bitcast=True)
    cache_v_u64 = p.cache_v.to(gl.pointer_type(gl.uint64), bitcast=True)
    query_u32 = p.query.to(gl.pointer_type(gl.uint32), bitcast=True)
    parallel_blocks = p.attn_parallel_blocks.to(gl.int32)
    task_count = GL_KV_HEADS * parallel_blocks
    task = gl.program_id(0)

    while task < task_count:
        kv_head = task // parallel_blocks
        part = task - kv_head * parallel_blocks
        query_head = kv_head * 8 + warp_1d
        query_head_load = kv_head * 8 + gl.arange(0, 8, layout=gl.SliceLayout(1, query_layout))
        query_pair_load = gl.arange(0, 32, layout=gl.SliceLayout(0, query_layout))
        query_bits = gl.load(query_u32 + query_head_load[:, None] * 32 + query_pair_load[None, :])
        query_shared.store(query_bits)
        gl.barrier()

        kq_max = gl.full([8], -1.7014117e38, gl.float32, warp_layout)
        kq_sum = gl.zeros([8], gl.float32, layout=warp_layout)
        value0 = gl.zeros([8, 32], gl.float16, layout=thread_layout)
        value1 = gl.zeros([8, 32], gl.float16, layout=thread_layout)
        tile = part * 128
        tile_stride = parallel_blocks * 128
        while tile < p.n_kv:
            score0 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            score1 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            score2 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            score3 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            for stage in gl.static_range(0, 4):
                key = tile + stage * 32 + copy_key
                valid = key < p.n_kv
                row = gl.load(p.kv_rows + key, mask=valid, other=0)
                cache_offsets = row[:, None] * (GL_KV // 4) + kv_head * (GL_HEAD // 4) + copy_quad[None, :]
                cache_bits64 = gl.amd.cdna3.buffer_load(cache_k_u64, cache_offsets)
                cache_bits64 = gl.where(valid[:, None], cache_bits64, 0)
                kv_stage = kv_shared.index(stage & 1).permute((1, 0))
                kv_stage.store(gl.permute(cache_bits64, 1, 0))
                gl.barrier()

                dot = gl.zeros([8, 32], gl.float32, layout=thread_layout)
                for d_pair in core.range(0, 32, loop_unroll_factor=1):
                    quad_index = d_pair // 2 + warp * 0 + lane * 0
                    k64 = kv_stage.gather(quad_index.to(gl.int32), axis=0)
                    k = gl.where((d_pair & 1) == 0, k64.to(gl.uint32), (k64 >> 32).to(gl.uint32))
                    q = query_shared.gather((d_pair + warp * 0 + lane * 0).to(gl.int32), axis=1)
                    dot = packed_dot2(k, q, dot, False)
                if stage == 0:
                    score0 = dot
                elif stage == 1:
                    score1 = dot
                elif stage == 2:
                    score2 = dot
                else:
                    score3 = dot
                gl.barrier()

            active0 = tile + lane < p.n_kv
            active1 = tile + 32 + lane < p.n_kv
            active2 = tile + 64 + lane < p.n_kv
            active3 = tile + 96 + lane < p.n_kv
            tile_max = gl.maximum(
                gl.maximum(
                    gl.max(gl.where(active0, score0 + GL_ATTENTION_MAX_OFFSET, -1.7014117e38), axis=1),
                    gl.max(gl.where(active1, score1 + GL_ATTENTION_MAX_OFFSET, -1.7014117e38), axis=1),
                ),
                gl.maximum(
                    gl.max(gl.where(active2, score2 + GL_ATTENTION_MAX_OFFSET, -1.7014117e38), axis=1),
                    gl.max(gl.where(active3, score3 + GL_ATTENTION_MAX_OFFSET, -1.7014117e38), axis=1),
                ),
            )
            maximum_new = gl.maximum(kq_max, tile_max)
            old_scale = gl.exp(kq_max - maximum_new)
            kq_max = maximum_new
            weight0 = gl.where(active0, gl.exp(score0 - kq_max[:, None]), 0.0)
            weight1 = gl.where(active1, gl.exp(score1 - kq_max[:, None]), 0.0)
            weight2 = gl.where(active2, gl.exp(score2 - kq_max[:, None]), 0.0)
            weight3 = gl.where(active3, gl.exp(score3 - kq_max[:, None]), 0.0)
            kq_sum = kq_sum * old_scale + gl.sum(weight0, axis=1) + gl.sum(weight1, axis=1)
            kq_sum += gl.sum(weight2, axis=1) + gl.sum(weight3, axis=1)
            weights.slice(0, 32, dim=1).store(weight0.to(gl.float16))
            weights.slice(32, 32, dim=1).store(weight1.to(gl.float16))
            weights.slice(64, 32, dim=1).store(weight2.to(gl.float16))
            weights.slice(96, 32, dim=1).store(weight3.to(gl.float16))
            gl.barrier()

            for stage in gl.static_range(0, 4):
                key = tile + stage * 32 + copy_key
                valid = key < p.n_kv
                row = gl.load(p.kv_rows + key, mask=valid, other=0)
                cache_offsets = row[:, None] * (GL_KV // 4) + kv_head * (GL_HEAD // 4) + copy_quad[None, :]
                cache_bits64 = gl.amd.cdna3.buffer_load(cache_v_u64, cache_offsets)
                cache_bits64 = gl.where(valid[:, None], cache_bits64, 0)
                kv_stage64 = kv_shared.index(stage & 1)
                kv_stage64.store(cache_bits64)
                gl.barrier()
                kv_stage32 = kv_stage64._reinterpret(gl.uint32, [32, 32], kv_shared32_layout)
                for key_in_chunk in core.range(0, 32, loop_unroll_factor=4):
                    weight_index = stage * 32 + key_in_chunk + warp * 0 + lane * 0
                    weight = weights.gather(weight_index.to(gl.int32), axis=1)
                    pair_index = key_in_chunk + warp * 0 + lane * 0
                    pair_bits = kv_stage32.gather(pair_index.to(gl.int32), axis=0)
                    low_bits = (pair_bits & 0xffff).to(gl.uint16)
                    high_bits = (pair_bits >> 16).to(gl.uint16)
                    v0 = low_bits.to(gl.float16, bitcast=True)
                    v1 = high_bits.to(gl.float16, bitcast=True)
                    if stage == 0 and key_in_chunk == 0:
                        value0 = (value0 * old_scale[:, None].to(gl.float16) + v0 * weight).to(gl.float16)
                        value1 = (value1 * old_scale[:, None].to(gl.float16) + v1 * weight).to(gl.float16)
                    else:
                        value0 = (value0 + v0 * weight).to(gl.float16)
                        value1 = (value1 + v1 * weight).to(gl.float16)
                gl.barrier()
            tile += tile_stride

        sink = gl.load(p.attn_sinks + query_head)
        sink_here = part == 0
        maximum_new = gl.where(sink_here, gl.maximum(kq_max, sink), kq_max)
        old_scale = gl.where(sink_here, gl.exp(kq_max - maximum_new), 1.0)
        kq_sum = gl.where(sink_here, kq_sum * old_scale + gl.exp(sink - maximum_new), kq_sum)
        value0 = (value0 * old_scale[:, None].to(gl.float16)).to(gl.float16)
        value1 = (value1 * old_scale[:, None].to(gl.float16)).to(gl.float16)
        part_index = query_head * parallel_blocks + part
        gl.store(p.attn_parts + part_index[:, None] * GL_HEAD + 2 * lane, value0.to(gl.float32))
        gl.store(p.attn_parts + part_index[:, None] * GL_HEAD + 2 * lane + 1, value1.to(gl.float32))
        gl.store(p.attn_meta + part_index * 2, maximum_new)
        gl.store(p.attn_meta + part_index * 2 + 1, kq_sum)
        task += GL_GRID_SIZE


@gluon.jit
def clear_full_attention_counters(p):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    tid = gl.arange(0, 256, layout=layout)
    counters = p.rms_partials.to(gl.pointer_type(gl.int32), bitcast=True)
    if gl.program_id(0) == 0:
        gl.store(counters + tid, 0, mask=tid < GL_KV_HEADS)


@gluon.jit
def full_attention_last_arrival(p):
    thread_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [8, 1], [1, 0])
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    warp_1d = gl.arange(0, 8, layout=warp_layout)
    lane_1d = gl.arange(0, 32, layout=lane_layout)
    warp = warp_1d[:, None]
    lane = lane_1d[None, :]

    full_attention_parts_u64(p)
    gl.barrier()

    pid = gl.program_id(0)
    parallel_blocks = p.attn_parallel_blocks.to(gl.int32)
    task_count = GL_KV_HEADS * parallel_blocks
    producer = pid < task_count
    kv_head = pid // parallel_blocks
    owner = producer & (warp == 0) & (lane == 0)
    counters = p.rms_partials.to(gl.pointer_type(gl.int32), bitcast=True)
    old = gl.atomic_add(
        counters + kv_head + warp * 0 + lane * 0,
        gl.full([8, 32], 1, gl.int32, layout=thread_layout),
        mask=owner,
        sem="acq_rel",
        scope="gpu",
    )
    gl.store(counters + GL_KV_HEADS + pid + warp * 0 + lane * 0, old, mask=owner)
    gl.barrier()

    arrival = gl.load(counters + GL_KV_HEADS + pid, mask=producer, other=-1)
    if producer & (arrival == parallel_blocks - 1):
        query_head = kv_head * 8 + warp_1d
        meta_base = query_head[:, None] * parallel_blocks * 2
        maximum = gl.load(p.attn_meta + meta_base + lane * 0)
        part = 1
        while part < parallel_blocks:
            part_max = gl.load(p.attn_meta + meta_base + part * 2 + lane * 0)
            maximum = gl.maximum(maximum, part_max)
            part += 1

        numerator0 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        numerator1 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        denominator = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        part = 0
        while part < parallel_blocks:
            part_max = gl.load(p.attn_meta + meta_base + part * 2 + lane * 0)
            part_sum = gl.load(p.attn_meta + meta_base + part * 2 + 1 + lane * 0)
            value_base = (query_head[:, None] * parallel_blocks + part) * GL_HEAD
            value0 = gl.load(p.attn_parts + value_base + 2 * lane)
            value1 = gl.load(p.attn_parts + value_base + 2 * lane + 1)
            scale = gl.exp(part_max - maximum)
            numerator0 += scale * value0
            numerator1 += scale * value1
            denominator += scale * part_sum
            part += 1

        output_base = query_head[:, None] * GL_HEAD
        gl.store(p.query + output_base + 2 * lane, (numerator0 / denominator).to(gl.float16))
        gl.store(p.query + output_base + 2 * lane + 1, (numerator1 / denominator).to(gl.float16))


@gluon.jit
def attention_output_residual_rms(p):
    values = p.attn_output_values
    scales = (p.attn_output_values + GL_ATTN_OUT_VALUES_BYTES).to(gl.pointer_type(gl.float16), bitcast=True)
    output_layout: gl.constexpr = gl.BlockedLayout([4], [32], [8], [0])
    output_offset = gl.arange(0, 4, layout=output_layout)
    rms_values = gl.zeros([4], gl.float32, layout=output_layout)
    task = gl.program_id(0)
    while task < GL_HIDDEN // 4:
        row = task * 4
        dots = q8_dot4(
            values,
            scales,
            p.query,
            row,
            row + 1,
            row + 2,
            row + 3,
            128,
            GL_QUERY // 4,
        )
        current = gl.load(p.cur + row + output_offset)
        bias = gl.load(p.attn_output_bias + row + output_offset)
        result = dots + bias + current
        gl.store(p.next + row + output_offset, result)
        rms_values += result * result
        task += GL_GRID_SIZE
    gl.store(p.rms_partials + gl.program_id(0), gl.sum(rms_values, axis=0))


@gluon.jit
def grouped_index(row):
    block = row // 32
    within = row % 32
    group = within // 8
    item = within % 8
    reordered = gl.where((item & 1) == 0, item // 2, 4 + item // 2)
    return block * 32 + group * 8 + reordered


@gluon.jit
def post_attention_norm_and_router(p):
    pid = gl.program_id(0)
    partial_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    partial = gl.arange(0, 256, layout=partial_layout)
    total = gl.sum(gl.load(p.rms_partials + partial, mask=partial < GL_RMS_PARTIALS, other=0.0), axis=0)
    scale = gl.rsqrt(total / GL_HIDDEN + p.rms_epsilon)

    linear_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    local = gl.arange(0, 256, layout=linear_layout)
    offset = pid * 256 + local
    mask = offset < GL_HIDDEN
    values = gl.load(p.next + offset, mask=mask, other=0.0)
    norm = gl.load(p.post_attention_norm + offset, mask=mask, other=0.0)
    gl.store(p.activation_scratch + grouped_index(offset), (values * scale * norm).to(gl.float16), mask=mask)

    if pid < GL_EXPERTS:
        reduce_layout: gl.constexpr = gl.BlockedLayout([16], [32], [8], [0])
        column = gl.arange(0, 4096, layout=reduce_layout)
        valid = column < GL_HIDDEN
        x = gl.load(p.next + column, mask=valid, other=0.0)
        n = gl.load(p.post_attention_norm + column, mask=valid, other=0.0)
        w = gl.load(p.router_weight + pid * GL_HIDDEN + column, mask=valid, other=0.0)
        result = gl.sum(w * x * scale * n, axis=0) + gl.load(p.router_bias + pid)
        gl.store(p.router + pid, result)


@gluon.jit
def unpack_top4_score(key):
    ordered = (key >> 6).to(gl.uint32)
    bits = gl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ordered ^ 0xffffffff)
    bits = gl.where((bits & 0x7fffffff) == 0, bits | ((key & 1).to(gl.uint32) << 31), bits)
    return bits.to(gl.float32, bitcast=True)


@gluon.jit
def select_top4(p, WARP_LAYOUT: gl.constexpr, LANE_LAYOUT: gl.constexpr):
    warp = gl.arange(0, 8, layout=WARP_LAYOUT)[:, None]
    lane = gl.arange(0, 32, layout=LANE_LAYOUT)[None, :]
    score = gl.load(p.router + lane + warp * 0)
    score = gl.where(score == score, score, -3.402823466e38)
    raw_bits = score.to(gl.uint32, bitcast=True)
    zero_sign = gl.where(score == 0.0, raw_bits >> 31, 0).to(gl.uint64)
    bits = gl.where(score == 0.0, 0, raw_bits)
    ordered = gl.where((bits & 0x80000000) != 0, bits ^ 0xffffffff, bits ^ 0x80000000)
    expert = (lane + warp * 0).to(gl.int32)
    tie = (31 - expert).to(gl.uint64)
    key = (ordered.to(gl.uint64) << 6) | (tie << 1) | zero_sign
    neg_inf_key = gl.full(key.shape, 0x1fffffc0, gl.uint64) | (tie << 1)
    best_key0 = gl.max(key, axis=1)
    id0 = (31 - ((best_key0 >> 1) & 31)).to(gl.int32)
    key = gl.where(lane == id0[:, None], neg_inf_key, key)
    best_key1 = gl.max(key, axis=1)
    id1 = (31 - ((best_key1 >> 1) & 31)).to(gl.int32)
    key = gl.where(lane == id1[:, None], neg_inf_key, key)
    best_key2 = gl.max(key, axis=1)
    id2 = (31 - ((best_key2 >> 1) & 31)).to(gl.int32)
    key = gl.where(lane == id2[:, None], neg_inf_key, key)
    best_key3 = gl.max(key, axis=1)
    id3 = (31 - ((best_key3 >> 1) & 31)).to(gl.int32)
    best0 = unpack_top4_score(best_key0)
    best1 = unpack_top4_score(best_key1)
    best2 = unpack_top4_score(best_key2)
    best3 = unpack_top4_score(best_key3)
    maximum = gl.maximum(gl.maximum(best0, best1), gl.maximum(best2, best3))
    w0 = gl.exp(best0 - maximum)
    w1 = gl.exp(best1 - maximum)
    w2 = gl.exp(best2 - maximum)
    w3 = gl.exp(best3 - maximum)
    total = w0 + w1 + w2 + w3
    return id0, id1, id2, id3, w0 / total, w1 / total, w2 / total, w3 / total


@gluon.jit
def swiglu_value(gate, up):
    exponential = libdevice.fast_expf(-1.702 * gate)
    activated = libdevice.fast_dividef(gate, 1.0 + exponential)
    return activated * (1.0 + up)


@gluon.jit
def barrier_counters(p):
    return (p.activation_scratch + GL_BARRIER_COUNTER_FP16_OFFSET).to(
        gl.pointer_type(gl.int32), bitcast=True
    )


@gluon.jit
def initialize_fixed_barrier_counters(p):
    scalar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    scalar = gl.arange(0, 1, layout=scalar_layout)
    owner = (scalar == 0) & (get_local_linear_id() == 0)
    counters = barrier_counters(p)
    if gl.program_id(0) == 0:
        for index in core.static_range(0, GL_BARRIER_COUNTER_COUNT):
            gl.store(counters + index + scalar, 0, mask=owner)


@gluon.jit
def fixed_grid_sync(counter):
    scalar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    scalar = gl.arange(0, 1, layout=scalar_layout)
    owner = (scalar == 0) & (get_local_linear_id() == 0)
    zero = gl.zeros([1], gl.int32, layout=scalar_layout)
    gl.barrier()
    old = gl.atomic_add(
        counter + scalar,
        gl.full([1], 1, gl.int32, layout=scalar_layout),
        mask=owner,
        sem="acq_rel",
        scope="gpu",
    )
    seen = old + 1
    while (seen != GL_GRID_SIZE).item():
        seen = gl.atomic_add(
            counter + scalar,
            zero,
            mask=owner,
            sem="relaxed",
            scope="gpu",
        )
    gl.atomic_add(
        counter + scalar,
        zero,
        mask=owner,
        sem="acquire",
        scope="gpu",
    )
    gl.barrier()


@gluon.jit
def moe_gate_up(p):
    weight_layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    activation_layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    thread_layout: gl.constexpr = gl.SliceLayout(2, weight_layout)
    activation_thread_layout: gl.constexpr = gl.SliceLayout(2, activation_layout)
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    weight_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, weight_layout))
    activation_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, activation_layout))
    warp_1d = gl.arange(0, 8, layout=warp_layout)
    lane_1d = gl.arange(0, 32, layout=lane_layout)
    weight_word_1d = gl.arange(0, 2, layout=weight_word_layout)
    activation_word_1d = gl.arange(0, 4, layout=activation_word_layout)
    warp = warp_1d[:, None]
    lane = lane_1d[None, :]
    weight_word = gl.expand_dims(gl.expand_dims(weight_word_1d, 0), 0)
    activation_word = gl.expand_dims(gl.expand_dims(activation_word_1d, 0), 0)
    values_u64 = p.moe_gate_up_values.to(gl.pointer_type(gl.uint64), bitcast=True)
    scales = p.moe_gate_up_values + GL_MOE_GATE_VALUES_BYTES
    activation_u32 = p.activation_scratch.to(gl.pointer_type(gl.uint32), bitcast=True)
    id0, id1, id2, id3, w0, w1, w2, w3 = select_top4(p, warp_layout, lane_layout)

    padding_index = gl.program_id(0) * 256 + warp * 32 + lane
    expert_slot = padding_index // (GL_PADDED - GL_INTERMEDIATE)
    padding_offset = padding_index % (GL_PADDED - GL_INTERMEDIATE)
    padding_mask = padding_index < GL_EXPERT_PADDING_COUNT
    gl.store(
        p.activation_scratch + GL_HIDDEN + expert_slot * GL_PADDED + GL_INTERMEDIATE + padding_offset,
        0.0,
        mask=padding_mask,
    )

    if gl.program_id(0) == 0:
        first_warp = warp_1d == 0
        zero_offset = warp_1d * 0
        gl.store(p.expert_ids + zero_offset, id0, mask=first_warp)
        gl.store(p.expert_ids + 1 + zero_offset, id1, mask=first_warp)
        gl.store(p.expert_ids + 2 + zero_offset, id2, mask=first_warp)
        gl.store(p.expert_ids + 3 + zero_offset, id3, mask=first_warp)
        gl.store(p.expert_weights + zero_offset, w0, mask=first_warp)
        gl.store(p.expert_weights + 1 + zero_offset, w1, mask=first_warp)
        gl.store(p.expert_weights + 2 + zero_offset, w2, mask=first_warp)
        gl.store(p.expert_weights + 3 + zero_offset, w3, mask=first_warp)

    task = gl.program_id(0)
    while task < GL_GATE_TASKS:
        slot = task // GL_GATE_OUT_BLOCKS
        output_block = task % GL_GATE_OUT_BLOCKS
        expert = gl.where(slot == 0, id0, gl.where(slot == 1, id1, gl.where(slot == 2, id2, id3)))
        logical_row = output_block * GL_ROWS_PER_CTA + warp * GL_ROWS_PER_WAVE
        logical_row_1d = output_block * GL_ROWS_PER_CTA + warp_1d * GL_ROWS_PER_WAVE
        output_base = GL_HIDDEN + slot * GL_PADDED
        for row_in_wave in core.range(0, GL_ROWS_PER_WAVE, loop_unroll_factor=1):
            physical_row = expert[:, None] * (2 * GL_PADDED) + (logical_row + row_in_wave) * 2
            acc0 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            acc1 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
            block_group = task * 0
            while block_group < 3:
                block = lane + block_group * 32 + warp * 0
                valid = block < GL_K_BLOCKS
                block_activation = gl.convert_layout(block, activation_thread_layout)
                valid_activation = gl.convert_layout(valid, activation_thread_layout)
                act_base = gl.expand_dims(block_activation, 2) * 16
                act_mask = gl.expand_dims(valid_activation, 2)
                value_stride = GL_PADDED_BLOCKS * 2
                value_base = gl.expand_dims(physical_row * value_stride + block * 2, 2) + weight_word
                value_mask = gl.expand_dims(valid, 2)
                scale_base = physical_row * GL_PADDED_BLOCKS + block
                with gl.amd.warp_pipeline_stage("load"):
                    activation0 = gl.load(activation_u32 + act_base + activation_word, mask=act_mask, other=0)
                    activation1 = gl.load(activation_u32 + act_base + 4 + activation_word, mask=act_mask, other=0)
                    activation2 = gl.load(activation_u32 + act_base + 8 + activation_word, mask=act_mask, other=0)
                    activation3 = gl.load(activation_u32 + act_base + 12 + activation_word, mask=act_mask, other=0)
                    packed0 = gl.load(
                        values_u64 + value_base,
                        mask=value_mask,
                        other=0,
                        cache_modifier=".cg",
                    )
                    packed1 = gl.load(
                        values_u64 + value_base + value_stride,
                        mask=value_mask,
                        other=0,
                        cache_modifier=".cg",
                    )
                    scale0_u8 = gl.load(scales + scale_base, mask=valid, other=0, cache_modifier=".cg")
                    scale1_u8 = gl.load(
                        scales + scale_base + GL_PADDED_BLOCKS,
                        mask=valid,
                        other=0,
                        cache_modifier=".cg",
                    )
                with gl.amd.warp_pipeline_stage("compute"):
                    acc0 += e8m0_scale(scale0_u8) * mxfp4_dot_block(
                        packed0, activation0, activation1, activation2, activation3, thread_layout
                    )
                    acc1 += e8m0_scale(scale1_u8) * mxfp4_dot_block(
                        packed1, activation0, activation1, activation2, activation3, thread_layout
                    )
                block_group += 1

            sum0 = gl.sum(acc0, axis=1)
            sum1 = gl.sum(acc1, axis=1)
            output_row = logical_row_1d + row_in_wave
            bias_base = expert * (2 * GL_INTERMEDIATE) + 2 * output_row
            gate = gl.minimum(sum0 + gl.load(p.moe_gate_up_bias + bias_base), 7.0)
            up = gl.maximum(gl.minimum(sum1 + gl.load(p.moe_gate_up_bias + bias_base + 1), 7.0), -7.0)
            result = swiglu_value(gate, up)
            gl.store(
                p.activation_scratch + output_base + grouped_index(output_row),
                result.to(gl.float16),
            )
        task += GL_GRID_SIZE


@gluon.jit
def moe_down(p):
    weight_layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    activation_layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 32, 1], [8, 1, 1], [2, 1, 0])
    thread_layout: gl.constexpr = gl.SliceLayout(2, weight_layout)
    activation_thread_layout: gl.constexpr = gl.SliceLayout(2, activation_layout)
    warp_layout: gl.constexpr = gl.SliceLayout(1, thread_layout)
    lane_layout: gl.constexpr = gl.SliceLayout(0, thread_layout)
    weight_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, weight_layout))
    activation_word_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, activation_layout))
    warp_1d = gl.arange(0, GL_WARPS, layout=warp_layout)
    lane_1d = gl.arange(0, 32, layout=lane_layout)
    weight_word_1d = gl.arange(0, 2, layout=weight_word_layout)
    activation_word_1d = gl.arange(0, 4, layout=activation_word_layout)
    warp = warp_1d[:, None]
    lane = lane_1d[None, :]
    weight_word = gl.expand_dims(gl.expand_dims(weight_word_1d, 0), 0)
    activation_word = gl.expand_dims(gl.expand_dims(activation_word_1d, 0), 0)
    values_u64 = p.moe_down_values.to(gl.pointer_type(gl.uint64), bitcast=True)
    scales = p.moe_down_values + GL_MOE_DOWN_VALUES_BYTES
    activation_u32 = (p.activation_scratch + GL_HIDDEN).to(gl.pointer_type(gl.uint32), bitcast=True)
    pid = gl.program_id(0)
    row0 = (pid * GL_WARPS + warp_1d) * GL_ROWS_PER_WAVE
    total0 = gl.zeros([8], gl.float32, layout=warp_layout)
    total1 = gl.zeros([8], gl.float32, layout=warp_layout)
    total2 = gl.zeros([8], gl.float32, layout=warp_layout)
    for slot in core.range(0, GL_EXPERTS_USED, loop_unroll_factor=1):
        expert = gl.load(p.expert_ids + slot)
        weight = gl.load(p.expert_weights + slot)
        physical_row = expert * GL_PADDED + row0[:, None]
        acc0 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        acc1 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        acc2 = gl.zeros([8, 32], gl.float32, layout=thread_layout)
        for block_round in core.range(0, 3, loop_unroll_factor=1):
            block = lane + block_round * 32 + warp * 0
            block_activation = gl.convert_layout(block, activation_thread_layout)
            activation_base = slot * (GL_PADDED_BLOCKS * 16) + gl.expand_dims(block_activation, 2) * 16
            activation0 = gl.load(activation_u32 + activation_base + activation_word)
            activation1 = gl.load(activation_u32 + activation_base + 4 + activation_word)
            activation2 = gl.load(activation_u32 + activation_base + 8 + activation_word)
            activation3 = gl.load(activation_u32 + activation_base + 12 + activation_word)
            row_stride = GL_PADDED_BLOCKS * 2
            value_base0 = physical_row * row_stride + block * 2
            value_base1 = value_base0 + row_stride
            value_base2 = value_base1 + row_stride
            packed0 = gl.load(
                values_u64 + gl.expand_dims(value_base0, 2) + weight_word,
                cache_modifier=".cg",
            )
            packed1 = gl.load(
                values_u64 + gl.expand_dims(value_base1, 2) + weight_word,
                cache_modifier=".cg",
            )
            packed2 = gl.load(
                values_u64 + gl.expand_dims(value_base2, 2) + weight_word,
                cache_modifier=".cg",
            )
            scale_base0 = physical_row * GL_PADDED_BLOCKS + block
            scale0 = e8m0_scale(gl.load(scales + scale_base0, cache_modifier=".cg"))
            scale1 = e8m0_scale(gl.load(scales + scale_base0 + GL_PADDED_BLOCKS, cache_modifier=".cg"))
            scale2 = e8m0_scale(gl.load(scales + scale_base0 + 2 * GL_PADDED_BLOCKS, cache_modifier=".cg"))
            acc0 += scale0 * mxfp4_dot_block(
                packed0, activation0, activation1, activation2, activation3, thread_layout
            )
            acc1 += scale1 * mxfp4_dot_block(
                packed1, activation0, activation1, activation2, activation3, thread_layout
            )
            acc2 += scale2 * mxfp4_dot_block(
                packed2, activation0, activation1, activation2, activation3, thread_layout
            )
        dot0 = gl.sum(acc0, axis=1)
        dot1 = gl.sum(acc1, axis=1)
        dot2 = gl.sum(acc2, axis=1)
        bias_base = expert * GL_HIDDEN + row0
        total0 += weight * (dot0 + gl.load(p.moe_down_bias + bias_base))
        total1 += weight * (dot1 + gl.load(p.moe_down_bias + bias_base + 1))
        total2 += weight * (dot2 + gl.load(p.moe_down_bias + bias_base + 2))
    result0 = gl.load(p.next + row0) + total0
    result1 = gl.load(p.next + row0 + 1) + total1
    result2 = gl.load(p.next + row0 + 2) + total2
    gl.store(p.next + row0, result0)
    gl.store(p.next + row0 + 1, result1)
    gl.store(p.next + row0 + 2, result2)
    rms = result0 * result0 + result1 * result1 + result2 * result2
    gl.store(p.rms_partials + pid, gl.sum(rms, axis=0))


@gluon.jit
def decode_layer_swa(p):
    initialize_fixed_barrier_counters(p)
    counters = barrier_counters(p)
    attention_rms_norm_and_parallel_rope_table(p)
    grid_sync(gl.program_id(0))

    qkv_rope_cache(p)
    fixed_grid_sync(counters + 0)

    sliding_window_attention(p)
    fixed_grid_sync(counters + 1)

    attention_output_residual_rms(p)
    fixed_grid_sync(counters + 2)

    post_attention_norm_and_router(p)
    fixed_grid_sync(counters + 3)

    moe_gate_up(p)
    grid_sync(gl.program_id(0))
    moe_down(p)


@gluon.jit
def decode_layer_full(p):
    initialize_fixed_barrier_counters(p)
    counters = barrier_counters(p)
    attention_rms_norm_and_parallel_rope_table(p)
    grid_sync(gl.program_id(0))

    clear_full_attention_counters(p)
    qkv_rope_cache(p)
    fixed_grid_sync(counters + 0)

    full_attention_last_arrival(p)
    fixed_grid_sync(counters + 1)

    attention_output_residual_rms(p)
    fixed_grid_sync(counters + 2)

    post_attention_norm_and_router(p)
    fixed_grid_sync(counters + 3)

    moe_gate_up(p)
    grid_sync(gl.program_id(0))
    moe_down(p)


@gluon.jit
def gptoss_decode_layer_swa_gluon_kernel(p):
    decode_layer_swa(p)


@gluon.jit
def gptoss_decode_layer_full_gluon_kernel(p):
    decode_layer_full(p)
