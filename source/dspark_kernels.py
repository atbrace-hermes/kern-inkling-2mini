# SPDX-License-Identifier: Apache-2.0
"""DSpark 3-layer bidirectional block drafter Triton kernels for SM120 AOT."""

import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as tldevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as tldevice

_EX2_ASM = tl.constexpr("ex2.approx.ftz.f32 $0, $1;")


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise(
        _EX2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    row_stride,
    tap_stride,
    hidden_stride,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """RMSNorm of alpha-weighted sum of `TAPS` target auxiliary states (FP32 math, 1 BF16 round)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0.0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * row_stride + tap * tap_stride + h * hidden_stride,
            mask=h < HIDDEN,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, mask=h < HIDDEN, other=0.0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + h, (mixed * inv_rms * weight).to(OUT.dtype.element_ty), mask=h < HIDDEN)


@triton.jit
def _qk_norm_rope_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    INV_FREQ,
    POS,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    """In-place per-head Q/K RMSNorm (rounded to BF16) + Neox half-split RoPE using FP32 trig."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.maximum(tl.load(POS + t), 0).to(tl.float32)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    inv_freq = tl.load(INV_FREQ + half_ar).to(tl.float32)
    angle = pos * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = X + t * row_stride + col0
    if NORM:
        w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w1 = tl.load(w_ptr + half_ar).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
    else:
        x1 = tl.load(row + half_ar).to(tl.float32)
        x2 = tl.load(row + HALF + half_ar).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    """`silu(gate) * up` in FP32 with IEEE div_rn, single BF16 round at store."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    tl.store(Y + row * N + cols, (silu * up).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_residual_kernel(
    X,
    RESIDUAL,
    WEIGHT,
    OUT,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Optional in-place `residual += x`, then RMSNorm(residual) -> `out`."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    x = tl.load(X + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    if ADD_RESIDUAL:
        res = tl.load(RESIDUAL + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        x = (x + res).to(tl.bfloat16)
        tl.store(RESIDUAL + row * HIDDEN + cols, x, mask=mask)
        x = x.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / HIDDEN + EPS)
    w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + cols, (x * inv_rms * w).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _dspark_ctx_store_ring_kernel(
    KV,
    Positions,
    Lines,
    Lengths,
    Count,
    Cache,
    k_scale,
    v_scale,
    cache_seq_stride,
    kv_row_stride,
    tokens,
    PREFILL: tl.constexpr,
    COMMIT: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_WIDTH: tl.constexpr,
):
    """Store context K/V into per-seq FP8 E4M3 ring buffer in `dspark_state`.

    - `PREFILL=1, COMMIT=0`: single seq (`Lines[0]`), writes only `pos >= Lengths[0] - RING` to avoid ring wrap race.
    - `PREFILL=0, COMMIT=0`: `decode` (`rows=1`), writes `tokens` rows (`seq = r`).
    - `PREFILL=0, COMMIT=1`: `round` (`rows=BLOCK_SIZE`), writes ONLY `row_in_seq < Count[seq]`.
    """
    r = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    r_ok = r < tokens
    r_safe = tl.where(r_ok, r, 0)

    if PREFILL:
        seq = 0
        row_in_seq = 0
    elif COMMIT:
        seq = r_safe // BLOCK_SIZE
        row_in_seq = r_safe % BLOCK_SIZE
    else:
        seq = r_safe
        row_in_seq = 0

    line = tl.load(Lines + seq, mask=r_ok, other=0).to(tl.int64)
    pos_raw = tl.load(Positions + r_safe, mask=r_ok, other=-1).to(tl.int64)

    if COMMIT:
        n_acc = tl.load(Count + seq, mask=r_ok, other=0)
        write = row_in_seq < n_acc
    else:
        end = tl.load(Lengths + seq, mask=r_ok, other=0).to(tl.int64)
        write = (pos_raw >= end - RING) & (pos_raw < end)

    active = r_ok & (line > 0) & (pos_raw >= 0) & write
    d_mask = (d < KV_WIDTH) & active

    slot = tl.maximum(pos_raw, 0) % RING
    base = tl.maximum(line, 0) * cache_seq_stride + slot * (2 * KV_WIDTH)
    src_base = r_safe.to(tl.int64) * kv_row_stride

    k_val = tl.load(KV + src_base + d, mask=d_mask, other=0.0).to(tl.float32) / k_scale
    v_val = tl.load(KV + src_base + KV_WIDTH + d, mask=d_mask, other=0.0).to(tl.float32) / v_scale
    tl.store(Cache + base + d, k_val.to(Cache.dtype.element_ty), mask=d_mask)
    tl.store(Cache + base + KV_WIDTH + d, v_val.to(Cache.dtype.element_ty), mask=d_mask)


@triton.jit
def _dspark_ctx_store_paged_kernel(
    KV,
    Slots,
    Valid,
    Count,
    Cache,
    k_scale,
    v_scale,
    kv_layer_offset,
    kv_page_stride,
    kv_slot_stride,
    kv_head_stride,
    kv_v_offset,
    kv_row_stride,
    tokens,
    PAGE: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_WIDTH: tl.constexpr,
):
    """Store context K/V for DSpark full layer 1 into paged `states.kv` FP8 E4M3."""
    r = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    r_ok = r < tokens
    r_safe = tl.where(r_ok, r, 0)

    if COMMIT:
        seq = r_safe // BLOCK_SIZE
        row_in_seq = r_safe % BLOCK_SIZE
        n_acc = tl.load(Count + seq, mask=r_ok, other=0)
        commit_ok = row_in_seq < n_acc
    else:
        commit_ok = True

    slot = tl.load(Slots + r_safe, mask=r_ok, other=-1).to(tl.int64)
    valid = tl.load(Valid + r_safe, mask=r_ok, other=0)
    active = r_ok & commit_ok & (slot >= 0) & (valid != 0)
    d_mask = (d < KV_WIDTH) & active

    slot_safe = tl.maximum(slot, 0)
    page = slot_safe // PAGE
    in_page = slot_safe % PAGE
    head = d // 128
    col = d % 128
    base_k = kv_layer_offset + page * kv_page_stride + in_page * kv_slot_stride + head * kv_head_stride + col
    base_v = base_k + kv_v_offset
    src_base = r_safe.to(tl.int64) * kv_row_stride

    k_val = tl.load(KV + src_base + d, mask=d_mask, other=0.0).to(tl.float32) / k_scale
    v_val = tl.load(KV + src_base + KV_WIDTH + d, mask=d_mask, other=0.0).to(tl.float32) / v_scale
    tl.store(Cache + base_k, k_val.to(Cache.dtype.element_ty), mask=d_mask)
    tl.store(Cache + base_v, v_val.to(Cache.dtype.element_ty), mask=d_mask)


@triton.jit
def _dspark_attn_ring_kernel(
    Q,
    K,
    V,
    Cache,
    Lines,
    Lengths,
    Out,
    k_scale,
    v_scale,
    softmax_scale,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    stride_out_t,
    cache_seq_stride,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Bidirectional block attention for local DSpark layers (0, 2) over FP8 ring + live BF16 block."""
    G: tl.constexpr = H // HKV
    LOG2E: tl.constexpr = 1.4426950408889634
    kv_head = tl.program_id(0)
    seq = tl.program_id(1)

    seqlen_k = tl.load(Lengths + seq)
    line = tl.load(Lines + seq).to(tl.int64)
    prefix = tl.maximum(seqlen_k - BLOCK_SIZE, 0)

    rows = tl.arange(0, BLOCK_M)
    tok = rows // G
    row_valid = (tok < BLOCK_SIZE) & (seqlen_k > 0)
    head = kv_head * G + (rows % G)
    token = (seq * BLOCK_SIZE + tok).to(tl.int64)
    offs_d = tl.arange(0, D)

    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * D + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    first_key = tl.maximum(0, seqlen_k - WINDOW) if WINDOW > 0 else 0
    n_lo = first_key // BLOCK_N
    n_hi = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo, 0)

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    kv_width = HKV * D
    cache_base = tl.maximum(line, 0) * cache_seq_stride
    offs_n = tl.arange(0, BLOCK_N)

    for i in range(num_iters):
        keys = (n_lo + i) * BLOCK_N + offs_n
        in_window = (keys >= first_key) & (keys < seqlen_k)
        is_old = in_window & (keys < prefix) & (line > 0)
        is_live = in_window & (keys >= prefix)

        ring_slots = (keys % RING).to(tl.int64)
        old_at = cache_base + ring_slots * (2 * kv_width) + kv_head * D
        old_k_fp8 = tl.load(Cache + old_at[None, :] + offs_d[:, None], mask=is_old[None, :], other=0.0)
        old_v_fp8 = tl.load(Cache + old_at[:, None] + kv_width + offs_d[None, :], mask=is_old[:, None], other=0.0)
        old_k = (old_k_fp8.to(tl.float32) * k_scale).to(tl.bfloat16)
        old_v = (old_v_fp8.to(tl.float32) * v_scale).to(tl.bfloat16)

        live_tok = (seq * BLOCK_SIZE + tl.maximum(keys - prefix, 0)).to(tl.int64)
        live_k = tl.load(
            K + live_tok[None, :] * stride_k_t + kv_head * D + offs_d[:, None],
            mask=is_live[None, :],
            other=0.0,
        )
        live_v = tl.load(
            V + live_tok[:, None] * stride_v_t + kv_head * D + offs_d[None, :],
            mask=is_live[:, None],
            other=0.0,
        )

        k = tl.where((keys < prefix)[None, :], old_k, live_k)
        v = tl.where((keys < prefix)[:, None], old_v, live_v)

        s = tl.dot(q, k) * qk_scale
        visible = in_window & ((line > 0) | (keys >= prefix))
        s = tl.where(visible[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    empty = l_i == 0.0
    o = tl.where(seqlen_k > 0, acc / tl.where(empty, 1.0, l_i)[:, None], 0.0)
    tl.store(
        Out + token[:, None] * stride_out_t + head[:, None] * D + offs_d[None, :],
        o.to(Out.dtype.element_ty),
        mask=(tok < BLOCK_SIZE)[:, None],
    )


@triton.jit
def _dspark_attn_paged_kernel(
    Q,
    K,
    V,
    Cache,
    BlockTable,
    Lengths,
    Out,
    Lse,
    k_scale,
    v_scale,
    softmax_scale,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    stride_bt,
    stride_out_t,
    kv_layer_offset,
    kv_page_stride,
    kv_slot_stride,
    kv_head_stride,
    kv_v_offset,
    num_tokens,
    num_splits,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE: tl.constexpr,
    MODE_PARTIAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Bidirectional block attention for full DSpark layer 1 over FP8 paged `states.kv` + live BF16 block."""
    G: tl.constexpr = H // HKV
    LOG2E: tl.constexpr = 1.4426950408889634
    kv_head = tl.program_id(0) // num_splits
    split = tl.program_id(0) % num_splits
    seq = tl.program_id(1)

    seqlen_k = tl.load(Lengths + seq)
    rows = tl.arange(0, BLOCK_M)
    tok = rows // G
    head = kv_head * G + (rows % G)
    token = (seq * BLOCK_SIZE + tok).to(tl.int64)
    row_store = tok < BLOCK_SIZE
    offs_d = tl.arange(0, D)

    if seqlen_k <= 0:
        z = tl.zeros([BLOCK_M, D], tl.float32)
        if MODE_PARTIAL:
            part = (split.to(tl.int64) * num_tokens + token) * H + head
            part_mask = row_store & (token < num_tokens)
            tl.store(Lse + part, tl.full([BLOCK_M], float("-inf"), tl.float32), mask=part_mask)
            tl.store(Out + part[:, None] * D + offs_d[None, :], z, mask=part_mask[:, None])
        else:
            tl.store(
                Out + token[:, None] * stride_out_t + head[:, None] * D + offs_d[None, :],
                z.to(Out.dtype.element_ty),
                mask=row_store[:, None],
            )
        return

    prefix = tl.maximum(seqlen_k - BLOCK_SIZE, 0)
    num_live = seqlen_k - prefix
    n_prefix_tiles = tl.cdiv(prefix, BLOCK_N)
    do_live = split == (num_splits - 1)

    # Balanced contiguous partition of prefix tiles across splits.
    base_tiles = n_prefix_tiles // num_splits
    rem_tiles = n_prefix_tiles % num_splits
    tile_start = split * base_tiles + tl.minimum(split, rem_tiles)
    tile_count = base_tiles + tl.where(split < rem_tiles, 1, 0)
    tile_end = tile_start + tile_count

    if (tile_count == 0) & (not do_live):
        z = tl.zeros([BLOCK_M, D], tl.float32)
        if MODE_PARTIAL:
            part = (split.to(tl.int64) * num_tokens + token) * H + head
            part_mask = row_store & (token < num_tokens)
            tl.store(Lse + part, tl.full([BLOCK_M], float("-inf"), tl.float32), mask=part_mask)
            tl.store(Out + part[:, None] * D + offs_d[None, :], z, mask=part_mask[:, None])
        else:
            tl.store(
                Out + token[:, None] * stride_out_t + head[:, None] * D + offs_d[None, :],
                z.to(Out.dtype.element_ty),
                mask=row_store[:, None],
            )
        return

    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * D + offs_d[None, :],
        mask=row_store[:, None],
        other=0.0,
    )

    qk_scale_old = softmax_scale * k_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    bt = BlockTable + seq.to(tl.int64) * stride_bt
    offs_n = tl.arange(0, BLOCK_N)
    full_prefix_tiles = prefix // BLOCK_N
    full_end = tl.maximum(tile_start, tl.minimum(tile_end, full_prefix_tiles))
    has_partial = (tile_start <= full_prefix_tiles) & (full_prefix_tiles < tile_end)

    kv_slot_s = tl.multiple_of(kv_slot_stride, 16)
    kv_page_s = tl.multiple_of(kv_page_stride, 16)
    kv_v_off = tl.multiple_of(kv_v_offset, 16)
    kv_head_off = tl.multiple_of(kv_layer_offset + kv_head.to(tl.int64) * kv_head_stride, 16)

    # Hot loop: 100% full single-page FP8 tiles, coalesced [BLOCK_N, D] loads, zero per-key masks.
    for n in range(tile_start, full_end):
        key_base = n * BLOCK_N
        page_idx = key_base // PAGE
        page_off = key_base % PAGE
        page = tl.load(bt + page_idx).to(tl.int64)
        kv_base = tl.multiple_of(kv_head_off + page * kv_page_s + page_off * kv_slot_s, 16)

        k = tl.load(
            Cache + kv_base + offs_n[:, None] * kv_slot_s + offs_d[None, :],
            eviction_policy="evict_first",
        ).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k))
        m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale_old)
        alpha = _ex2(m_i - m_new)
        p = _ex2(s * qk_scale_old - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            Cache + kv_base + kv_v_off + offs_n[:, None] * kv_slot_s + offs_d[None, :],
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    # At most one partial prefix tile at the end of the prefix.
    if has_partial:
        key_base = full_prefix_tiles * BLOCK_N
        keys = key_base + offs_n
        is_old = keys < prefix
        page_idx = key_base // PAGE
        page_off = key_base % PAGE
        page = tl.load(bt + page_idx).to(tl.int64)
        kv_base = tl.multiple_of(kv_head_off + page * kv_page_s + page_off * kv_slot_s, 16)

        k = tl.load(
            Cache + kv_base + offs_n[:, None] * kv_slot_s + offs_d[None, :],
            mask=is_old[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * qk_scale_old
        s = tl.where(is_old[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = _ex2(m_i - m_safe)
        p = _ex2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            Cache + kv_base + kv_v_off + offs_n[:, None] * kv_slot_s + offs_d[None, :],
            mask=is_old[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc * v_scale

    # Live BF16 draft block (1..BLOCK_SIZE tokens) handled once by the final split.
    if do_live:
        LIVE_N: tl.constexpr = 16
        offs_live = tl.arange(0, LIVE_N)
        is_live = offs_live < num_live
        live_tok = (seq * BLOCK_SIZE + offs_live).to(tl.int64)

        k_live = tl.load(
            K + live_tok[:, None] * stride_k_t + kv_head * D + offs_d[None, :],
            mask=is_live[:, None],
            other=0.0,
        )
        s_live = tl.dot(q, tl.trans(k_live)) * (softmax_scale * LOG2E)
        s_live = tl.where(is_live[None, :], s_live, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s_live, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = _ex2(m_i - m_safe)
        p_live = _ex2(s_live - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p_live, 1)

        v_live = tl.load(
            V + live_tok[:, None] * stride_v_t + kv_head * D + offs_d[None, :],
            mask=is_live[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p_live.to(tl.bfloat16), v_live)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE_PARTIAL:
        part = (split.to(tl.int64) * num_tokens + token) * H + head
        part_mask = row_store & (token < num_tokens)
        tl.store(Lse + part, lse2, mask=part_mask)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=part_mask[:, None])
    else:
        tl.store(
            Out + token[:, None] * stride_out_t + head[:, None] * D + offs_d[None, :],
            o.to(Out.dtype.element_ty),
            mask=row_store[:, None],
        )


@triton.jit
def _dspark_attn_combine_kernel(
    OPart,
    LsePart,
    Y,
    num_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge split-KV FP32 partials and log2 LSEs for DSpark layer 1 into BF16 `Y`."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_ok = row < num_tokens
    row_safe = tl.where(row_ok, row, 0)
    y_ptrs = Y + row_safe * (H * D) + head * D + offs_d
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row_safe) * H + head
    lse_s = tl.load(LsePart + part, mask=(splits < NUM_SPLITS) & row_ok, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=((w > 0.0) & row_ok)[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    tl.store(y_ptrs, o.to(Y.dtype.element_ty), mask=(offs_d < H * D) & row_ok)
