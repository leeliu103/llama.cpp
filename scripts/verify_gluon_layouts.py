#!/usr/bin/env python3
import math
import sys

import triton.experimental.gluon.language as gl
from triton._C.libtriton import ir, gluon_ir, linear_layout


def to_linear_layout(layout, shape):
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = gluon_ir.GluonOpBuilder(ctx)
    return builder.to_linear_layout(layout._to_ir(builder), shape)


def _linear_from_distributed(layout):
    bases = [
        ("register", layout.reg_bases),
        ("lane", layout.lane_bases),
        ("warp", layout.warp_bases),
        ("block", layout.block_bases),
    ]
    out_dim_names = [f"dim{i}" for i in range(len(layout.shape))]
    return linear_layout.LinearLayout.from_bases(bases, out_dim_names, layout.shape, True)


def _linear_from_shared(layout):
    bases = [("offset", layout.offset_bases)]
    if layout.block_bases:
        bases.append(("block", layout.block_bases))
    out_dim_names = [f"dim{i}" for i in range(len(layout.shape))]
    return linear_layout.LinearLayout.from_bases(bases, out_dim_names, layout.shape, True)


def make_linear_layout(layout):
    if isinstance(layout, gl.DistributedLinearLayout):
        return _linear_from_distributed(layout)
    if isinstance(layout, gl.SharedLinearLayout):
        return _linear_from_shared(layout)
    raise TypeError(f"Unsupported layout type: {type(layout)}")


def ggml_blocked_base_2d(lane, warp, s0, s1, t0, t1, w0, w1, o0, o1):
    if o0 == 0:
        t0_idx = lane % t0
        lane //= t0
        t1_idx = lane % t1
    else:
        t1_idx = lane % t1
        lane //= t1
        t0_idx = lane % t0

    if o0 == 0:
        w0_idx = warp % w0
        warp //= w0
        w1_idx = warp % w1
    else:
        w1_idx = warp % w1
        warp //= w1
        w0_idx = warp % w0

    base0 = (w0_idx * t0 + t0_idx) * s0
    base1 = (w1_idx * t1 + t1_idx) * s1
    return base0, base1


def ggml_swizzle_col(row, col, vec, per_phase, max_phase):
    if vec == 0 or max_phase == 1:
        return col
    phase = (row // per_phase) & (max_phase - 1)
    return col ^ (phase * vec)


def _coords_from_reg_2d(reg, s0, s1, order):
    if order[0] == 0:
        dim0 = reg % s0
        dim1 = (reg // s0) % s1
    else:
        dim1 = reg % s1
        dim0 = (reg // s1) % s0
    return dim0, dim1


def _blocked_thread_coords(shape, s0, s1, t0, t1, w0, w1, o0, o1, lane, warp):
    base0, base1 = ggml_blocked_base_2d(lane, warp, s0, s1, t0, t1, w0, w1, o0, o1)
    block0 = s0 * t0 * w0
    block1 = s1 * t1 * w1
    tiles0 = (shape[0] + block0 - 1) // block0
    tiles1 = (shape[1] + block1 - 1) // block1
    coords = set()
    for t0_idx in range(tiles0):
        for t1_idx in range(tiles1):
            b0 = base0 + t0_idx * block0
            b1 = base1 + t1_idx * block1
            for r0 in range(s0):
                for r1 in range(s1):
                    d0 = b0 + r0
                    d1 = b1 + r1
                    if d0 < shape[0] and d1 < shape[1]:
                        coords.add((d0, d1))
    return coords


def check_blocked_layout(name, shape, layout, params):
    lin = to_linear_layout(layout, shape)
    ll = make_linear_layout(lin)
    reg_count = 1 << len(lin.reg_bases)
    warp_count = 1 << len(lin.warp_bases)
    lane_count = 1 << len(lin.lane_bases)
    ok = True

    for warp in range(warp_count):
        for lane in range(lane_count):
            coords_ref = _blocked_thread_coords(shape, *params, lane, warp)
            coords_layout = set()
            for reg in range(reg_count):
                out = ll.apply({"register": reg, "lane": lane, "warp": warp, "block": 0})
                d0 = out["dim0"]
                d1 = out["dim1"] if len(shape) > 1 else 0
                if d0 < shape[0] and (len(shape) == 1 or d1 < shape[1]):
                    coords_layout.add((d0, d1))
            if coords_ref != coords_layout:
                print(f"{name}: mismatch at lane={lane} warp={warp}")
                ok = False
                break
        if not ok:
            break
    return ok


def check_shared_swizzle(name, shape, layout, vec, per_phase, max_phase, order):
    lin = to_linear_layout(layout, shape)
    ll = make_linear_layout(lin)
    ok = True
    for d0 in range(shape[0]):
        for d1 in range(shape[1]):
            dims = [d0, d1]
            row = dims[order[1]]
            col = dims[order[0]]
            col_sw = ggml_swizzle_col(row, col, vec, per_phase, max_phase)
            num_cols = shape[order[0]]
            offset = row * num_cols + col_sw
            inputs = {"offset": offset}
            if lin.block_bases:
                inputs["block"] = 0
            out = ll.apply(inputs)
            if out["dim0"] != d0 or out["dim1"] != d1:
                print(f"{name}: mismatch at dim0={d0} dim1={d1}")
                ok = False
                break
        if not ok:
            break
    return ok


def ggml_tile_coords_i_major(lane, reg, i, j):
    ne = i * j // 32
    row = lane % i
    col_pair = ne * (lane // i) + (reg // 2)
    col = 2 * col_pair + (reg % 2)
    return row, col


def ggml_tile_coords_j_major(lane, reg, i, j):
    row_i, col_i = ggml_tile_coords_i_major(lane, reg, i, j)
    return col_i, row_i


def ggml_tile_coords_c(lane, reg, i, j):
    ne = i * j // 32
    row = ne * (lane // i) + reg
    col = lane % i
    return row, col


def check_dot_layout(name, layout, shape, coords_fn):
    lin = to_linear_layout(layout, shape)
    ll = make_linear_layout(lin)
    reg_count = 1 << len(lin.reg_bases)
    tile_i = shape[0]
    tile_j = shape[1] // 2
    ok_direct = True
    ok_swapped = True

    for lane in range(32):
        for reg in range(reg_count):
            out = ll.apply({"register": reg, "lane": lane, "warp": 0, "block": 0})
            exp0, exp1 = coords_fn(lane, reg, tile_i, tile_j)
            if (out["dim0"], out["dim1"]) != (exp0, exp1):
                ok_direct = False
            if (out["dim0"], out["dim1"]) != (exp1, exp0):
                ok_swapped = False
        if not ok_direct and not ok_swapped:
            break

    if ok_direct:
        return True, "dim0=row dim1=col"
    if ok_swapped:
        return True, "dim0=col dim1=row"
    return False, "mismatch"


def check_wmma_layout(name, layout, shape):
    lin = to_linear_layout(layout, shape)
    ll = make_linear_layout(lin)
    reg_count = 1 << len(lin.reg_bases)
    ok_direct = True
    ok_swapped = True

    for lane in range(32):
        for reg in range(reg_count):
            out = ll.apply({"register": reg, "lane": lane, "warp": 0, "block": 0})
            exp0, exp1 = ggml_tile_coords_c(lane, reg, shape[0], shape[1])
            if (out["dim0"], out["dim1"]) != (exp0, exp1):
                ok_direct = False
            if (out["dim0"], out["dim1"]) != (exp1, exp0):
                ok_swapped = False
        if not ok_direct and not ok_swapped:
            break

    if ok_direct:
        return True, "dim0=row dim1=col"
    if ok_swapped:
        return True, "dim0=col dim1=row"
    return False, "mismatch"


def main():
    block_m = 64
    block_n = 32
    block_ds = [128]

    failures = 0

    # Blocked layouts in half2 units.
    blocked_q = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    blocked_k = gl.BlockedLayout([4, 1], [16, 2], [1, 4], [0, 1])
    blocked_v = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    blocked_m = gl.BlockedLayout([1], [32], [4], [0])

    # Swizzled shared layouts in half2 units.
    shared_q = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
    shared_k = gl.SwizzledSharedLayout(4, 1, 16, order=[0, 1])
    shared_v = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    for block_d in block_ds:
        print(f"block_d={block_d}")

        if check_blocked_layout(
            "blocked_q", [block_m, block_d // 2], blocked_q, (1, 4, 2, 16, 4, 1, 1, 0)
        ):
            print("blocked_q: OK")
        else:
            failures += 1
        if check_blocked_layout(
            "blocked_k", [block_d // 2, block_n], blocked_k, (4, 1, 16, 2, 1, 4, 0, 1)
        ):
            print("blocked_k: OK")
        else:
            failures += 1
        if check_blocked_layout(
            "blocked_v", [block_n, block_d // 2], blocked_v, (1, 4, 2, 16, 4, 1, 1, 0)
        ):
            print("blocked_v: OK")
        else:
            failures += 1

        lin_m = to_linear_layout(blocked_m, [block_m])
        ll_m = make_linear_layout(lin_m)
        reg_count = 1 << len(lin_m.reg_bases)
        lane_count = 1 << len(lin_m.lane_bases)
        warp_count = 1 << len(lin_m.warp_bases)
        ok_m = True
        for warp in range(warp_count):
            for lane in range(lane_count):
                coords_layout = set()
                for reg in range(reg_count):
                    out = ll_m.apply({"register": reg, "lane": lane, "warp": warp, "block": 0})
                    if out["dim0"] < block_m:
                        coords_layout.add(out["dim0"])
                if not coords_layout:
                    ok_m = False
                    break
            if not ok_m:
                break
        if ok_m:
            print("blocked_m: OK")
        else:
            print("blocked_m: mismatch")
            failures += 1

        max_phase = block_d // 8
        if check_shared_swizzle("shared_q", [block_m, block_d // 2], shared_q, 4, 1, max_phase, [1, 0]):
            print("shared_q: OK")
        else:
            failures += 1
        if check_shared_swizzle("shared_k", [block_d // 2, block_n], shared_k, 4, 1, max_phase, [0, 1]):
            print("shared_k: OK")
        else:
            failures += 1
        if check_shared_swizzle("shared_v", [block_n, block_d // 2], shared_v, 1, 1, 1, [1, 0]):
            print("shared_v: OK")
        else:
            failures += 1

    # WMMA and dot layouts (scalar shapes).
    wmma_layout = gl.amd.AMDWMMALayout(version=2, transposed=True, warp_bases=[[1, 0], [2, 0]])
    ok, msg = check_wmma_layout("wmma_layout", wmma_layout, [16, 16])
    if not ok:
        print(f"wmma_layout: {msg}")
        failures += 1
    else:
        print(f"wmma_layout: OK ({msg})")

    dot_p = gl.DotOperandLayout(0, wmma_layout, 8)
    ok, msg = check_dot_layout("dot_p", dot_p, [16, 16], ggml_tile_coords_i_major)
    if not ok:
        print(f"dot_p: {msg}")
        failures += 1
    else:
        print(f"dot_p: OK ({msg})")

    dot_k = gl.DotOperandLayout(1, wmma_layout, 8)
    ok, msg = check_dot_layout("dot_k", dot_k, [16, 16], ggml_tile_coords_j_major)
    if not ok:
        print(f"dot_k: {msg}")
        failures += 1
    else:
        print(f"dot_k: OK ({msg})")

    dot_v = gl.DotOperandLayout(1, wmma_layout, 8)
    ok, msg = check_dot_layout("dot_v", dot_v, [16, 16], ggml_tile_coords_j_major)
    if not ok:
        print(f"dot_v: {msg}")
        failures += 1
    else:
        print(f"dot_v: OK ({msg})")

    if failures:
        print(f"FAILED: {failures} layout checks failed")
        return 1
    print("OK: all layout checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
