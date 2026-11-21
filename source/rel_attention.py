#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SM120 Relative-Bias Attention, SWA Ring Attention, Split-KV Combine & Log-Scaling Tau (Triton 3.8 AOT)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2

LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)

_EX2_ASM = tl.constexpr("ex2.approx.ftz.f32 $0, $1;")
_LG2_ASM = tl.constexpr("lg2.approx.ftz.f32 $0, $1;")


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * LOG2E))


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise(
        _EX2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _lg2(x):
    return tl.inline_asm_elementwise(
        _LG2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, mask=i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tl.log(ratio)
    tl.store(Out + i, tau, mask=i < n)


@triton.jit
def rel_attention_full_kernel(
    Q,
    KVState,
    Rel,
    Gate,
    Sink,
    BlockTable,
    SeqLens,
    Out,
    Lse,
    num_tokens,
    table_width,
    stride_gate,
    stride_sink,
    sm_scale,
    v_scale,
    H: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    QLEN_CONST: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Paged FP8 E4M3 full attention with unsheared relative bias [tokens, H, E].

    KVState points to page 0 of the active full layer (`states.kv + full_idx * 131072`).
    Page layout: `[pages, 9, 2 (K/V), HKV=8, PAGE=64, D=128]` fp8e4nv.
    """
    TOKENS: tl.constexpr = 1 if QLEN_CONST == 1 else (8 if BLOCK_M == 32 else 16)
    STRIDE_KV_BLK: tl.constexpr = 9 * 2 * HKV * PAGE * D
    STRIDE_KV_HALF: tl.constexpr = HKV * PAGE * D
    STRIDE_KV_H: tl.constexpr = PAGE * D

    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // NUM_SPLITS
    split = tl.program_id(1) % NUM_SPLITS
    req = tl.program_id(2)

    q_start = 0 if QLEN_CONST == 0 else req * QLEN_CONST
    q_len = num_tokens if QLEN_CONST == 0 else QLEN_CONST
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return

    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)

    pos = past + tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    n_hi = tl.cdiv(tl.minimum(pos_hi + 1, seqlen_k), BLOCK_N)
    num_iters = tl.maximum(n_hi - split + NUM_SPLITS - 1, 0) // NUM_SPLITS

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    tl.extra.cuda.gdc_wait()

    q = tl.load(
        Q + token[:, None] * (H * D) + head[:, None] * D + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    qk_scale = sm_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    bt_base = BlockTable + req.to(tl.int64) * table_width
    rel_row_i32 = (q_start + tok) * (H * E) + head * E

    for i in range(num_iters):
        n = split + i * NUM_SPLITS
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page_idx = (n * BLOCK_N) // PAGE
        page_off = (n * BLOCK_N) % PAGE
        page = tl.load(bt_base + page_idx, mask=n * BLOCK_N < seqlen_k, other=0)
        kv_valid = key_valid & (page >= 0)
        kv_base = page.to(tl.int64) * STRIDE_KV_BLK + kv_head * STRIDE_KV_H + page_off * D

        k = tl.load(
            KVState + kv_base + offs_n[:, None] * D + offs_d[None, :],
            mask=kv_valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & kv_valid[None, :]
        in_extent = visible & (dist < E) & row_valid[:, None]
        safe_dist = tl.where(in_extent, dist, 0)
        bias = tl.load(
            Rel + (rel_row_i32[:, None] + safe_dist),
            mask=in_extent,
            other=0.0,
        )
        s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            KVState + kv_base + STRIDE_KV_HALF + offs_n[:, None] * D + offs_d[None, :],
            mask=kv_valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    tl.extra.cuda.gdc_launch_dependents()

    empty = l_i == 0.0
    o = (acc / tl.where(empty, 1.0, l_i)[:, None]) * v_scale
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))

    if MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        gate = tl.load(
            Gate + token[:, None] * stride_gate + cols,
            mask=row_valid[:, None],
            other=0.0,
        )
        sink = tl.load(
            Sink + token * stride_sink + head,
            mask=row_valid,
            other=0.0,
        )
        o = o * _sigmoid(gate.to(tl.float32))
        o = o * _sigmoid(lse2 * LN2 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * (H * D) + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


@triton.jit
def _rel_attention_swa_impl(
    Q,
    SeqState,
    KCur,
    VCur,
    Rel,
    Gate,
    Sink,
    Lines,
    SeqLens,
    Out,
    Lse,
    num_tokens,
    stride_gate,
    stride_sink,
    sm_scale,
    v_scale,
    H: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    USE_CUR_KV: tl.constexpr,
    QLEN_CONST: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sliding-window (W=1024) FP8 E4M3 attention over states.seq_state ring + optional KCur/VCur.

    SeqState points to slot 0 of the active SWA visit (`states.seq_state + 1916928 + swa_idx * 2097152`).
    Per-visit ring layout: `[2 (K/V), HKV=8, RING=1024, D=128]` fp8e4nv.
    """
    TOKENS: tl.constexpr = 1 if QLEN_CONST == 1 else (8 if BLOCK_M == 32 else 16)
    RING_V_OFF: tl.constexpr = HKV * RING * D

    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // NUM_SPLITS
    split = tl.program_id(1) % NUM_SPLITS
    req = tl.program_id(2)

    q_start = 0 if QLEN_CONST == 0 else req * QLEN_CONST
    q_len = num_tokens if QLEN_CONST == 0 else QLEN_CONST
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    offs_d = tl.arange(0, D)

    line = tl.load(Lines + req).to(tl.int64)
    if line <= 0:
        if MODE == 1:
            tl.store(
                Out + token[:, None] * (H * D) + head[:, None] * D + offs_d[None, :],
                tl.zeros([BLOCK_M, D], tl.bfloat16),
                mask=row_valid[:, None],
            )
        else:
            part = (split * num_tokens + token) * H + head
            tl.store(Lse + part, tl.full([BLOCK_M], float("-inf"), tl.float32), mask=row_valid)
            tl.store(
                Out + part[:, None] * D + offs_d[None, :],
                tl.zeros([BLOCK_M, D], tl.float32),
                mask=row_valid[:, None],
            )
        return

    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    pos = past + tok
    pos_lo = past + first_tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.cdiv(tl.minimum(pos_hi + 1, seqlen_k), BLOCK_N)
    num_iters = tl.maximum(n_hi - n_lo - split + NUM_SPLITS - 1, 0) // NUM_SPLITS

    offs_n = tl.arange(0, BLOCK_N)

    tl.extra.cuda.gdc_wait()

    q = tl.load(
        Q + token[:, None] * (H * D) + head[:, None] * D + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    qk_scale = sm_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    ring_k_base = line * SEQ_STRIDE + kv_head * (RING * D)
    ring_v_base = ring_k_base + RING_V_OFF
    rel_row_i32 = (q_start + tok) * (H * E) + head * E

    for i in range(num_iters):
        n = n_lo + split + i * NUM_SPLITS
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        ring_slots_i32 = (keys % RING) * D

        if USE_CUR_KV:
            is_cur = (keys >= past) & key_valid
            is_old = (keys < past) & key_valid
            k_old = tl.load(
                SeqState + ring_k_base + ring_slots_i32[:, None] + offs_d[None, :],
                mask=is_old[:, None],
                other=0.0,
            )
            cur_idx_i32 = (q_start + tl.maximum(keys - past, 0)) * (HKV * D) + kv_head * D
            k_cur = tl.load(
                KCur + cur_idx_i32[:, None] + offs_d[None, :],
                mask=is_cur[:, None],
                other=0.0,
            )
            k = tl.where(is_cur[:, None], k_cur, k_old).to(tl.bfloat16)
        else:
            k = tl.load(
                SeqState + ring_k_base + ring_slots_i32[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (dist < WINDOW) & key_valid[None, :]
        in_extent = visible & (dist < E) & row_valid[:, None]
        safe_dist = tl.where(in_extent, dist, 0)
        bias = tl.load(Rel + (rel_row_i32[:, None] + safe_dist), mask=in_extent, other=0.0)
        s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        if USE_CUR_KV:
            is_cur2 = (keys >= past) & key_valid
            is_old2 = (keys < past) & key_valid
            ring_slots2_i32 = (keys % RING) * D
            cur_idx2_i32 = (q_start + tl.maximum(keys - past, 0)) * (HKV * D) + kv_head * D
            v_old = tl.load(
                SeqState + ring_v_base + ring_slots2_i32[:, None] + offs_d[None, :],
                mask=is_old2[:, None],
                other=0.0,
            )
            v_cur = tl.load(
                VCur + cur_idx2_i32[:, None] + offs_d[None, :],
                mask=is_cur2[:, None],
                other=0.0,
            )
            v = tl.where(is_cur2[:, None], v_cur, v_old).to(tl.bfloat16)
        else:
            v = tl.load(
                SeqState + ring_v_base + ring_slots_i32[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    tl.extra.cuda.gdc_launch_dependents()

    empty = l_i == 0.0
    o = (acc / tl.where(empty, 1.0, l_i)[:, None]) * v_scale
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))

    if MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        gate = tl.load(
            Gate + token[:, None] * stride_gate + cols,
            mask=row_valid[:, None],
            other=0.0,
        )
        sink = tl.load(
            Sink + token * stride_sink + head,
            mask=row_valid,
            other=0.0,
        )
        o = o * _sigmoid(gate.to(tl.float32))
        o = o * _sigmoid(lse2 * LN2 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * (H * D) + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


@triton.jit
def rel_attention_swa_cur_kernel(
    Q,
    SeqState,
    KCur,
    VCur,
    Rel,
    Gate,
    Sink,
    Lines,
    SeqLens,
    Out,
    Lse,
    num_tokens,
    stride_gate,
    stride_sink,
    sm_scale,
    v_scale,
    H: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    QLEN_CONST: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _rel_attention_swa_impl(
        Q, SeqState, KCur, VCur, Rel, Gate, Sink, Lines, SeqLens, Out, Lse,
        num_tokens, stride_gate, stride_sink, sm_scale, v_scale,
        H, HKV, G, D, E, RING, WINDOW, SEQ_STRIDE, True,
        QLEN_CONST, NUM_SPLITS, MODE, BLOCK_M, BLOCK_N,
    )


@triton.jit
def rel_attention_swa_ring_kernel(
    Q,
    SeqState,
    Rel,
    Gate,
    Sink,
    Lines,
    SeqLens,
    Out,
    Lse,
    num_tokens,
    stride_gate,
    stride_sink,
    sm_scale,
    v_scale,
    H: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    QLEN_CONST: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _rel_attention_swa_impl(
        Q, SeqState, SeqState, SeqState, Rel, Gate, Sink, Lines, SeqLens, Out, Lse,
        num_tokens, stride_gate, stride_sink, sm_scale, v_scale,
        H, HKV, G, D, E, RING, WINDOW, SEQ_STRIDE, False,
        QLEN_CONST, NUM_SPLITS, MODE, BLOCK_M, BLOCK_N,
    )


@triton.jit
def rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge split-KV partials (log2 LSE), then apply gate and natural-log sink."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
    sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)

    tl.extra.cuda.gdc_wait()

    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = _ex2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    lse_nat = (_lg2(w_sum) + lse_max) * LN2
    o = o * _sigmoid(gate) * _sigmoid(lse_nat - sink)
    tl.store(Y + row * (H * D) + head * D + offs_d, o.to(Y.dtype.element_ty))


def get_dsl_specs() -> list[tuple]:
    """Return M3 SM120 Triton AOT kernel specs shared by build_dsl() and scripts/attn_build.py."""
    SEQ_STATE_BYTES = 90_005_376
    specs = [
        (
            "attn_log_tau",
            log_scaling_tau_kernel,
            {"Pos": "*i64", "Out": "*fp32", "n": "i32"},
            {"FLOOR": 8192, "ALPHA": 0.1, "BLOCK": 256},
            4,
            2,
            False,
        ),
        # Full attention (paged FP8 KV)
        (
            "attn_full_prefill",
            rel_attention_full_kernel,
            {
                "Q": "*bf16", "KVState": "*fp8e4nv", "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16",
                "BlockTable": "*i32", "SeqLens": "*i32", "Out": "*bf16", "Lse": "*fp32",
                "num_tokens": "i32", "table_width": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "PAGE": 64,
                "QLEN_CONST": 0, "NUM_SPLITS": 1, "MODE": MODE_GATED, "BLOCK_M": 32, "BLOCK_N": 32,
            },
            4,
            2,
            True,
        ),
        (
            "attn_full_prefill_split",
            rel_attention_full_kernel,
            {
                "Q": "*bf16", "KVState": "*fp8e4nv", "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16",
                "BlockTable": "*i32", "SeqLens": "*i32", "Out": "*fp32", "Lse": "*fp32",
                "num_tokens": "i32", "table_width": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "PAGE": 64,
                "QLEN_CONST": 0, "NUM_SPLITS": 16, "MODE": MODE_PARTIAL, "BLOCK_M": 32, "BLOCK_N": 32,
            },
            4,
            2,
            True,
        ),
        (
            "attn_full_decode1",
            rel_attention_full_kernel,
            {
                "Q": "*bf16", "KVState": "*fp8e4nv", "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16",
                "BlockTable": "*i32", "SeqLens": "*i32", "Out": "*fp32", "Lse": "*fp32",
                "num_tokens": "i32", "table_width": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "PAGE": 64,
                "QLEN_CONST": 1, "NUM_SPLITS": 64, "MODE": MODE_PARTIAL, "BLOCK_M": 16, "BLOCK_N": 64,
            },
            4,
            2,
            True,
        ),
        (
            "attn_full_decode8",
            rel_attention_full_kernel,
            {
                "Q": "*bf16", "KVState": "*fp8e4nv", "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16",
                "BlockTable": "*i32", "SeqLens": "*i32", "Out": "*fp32", "Lse": "*fp32",
                "num_tokens": "i32", "table_width": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "PAGE": 64,
                "QLEN_CONST": 1, "NUM_SPLITS": 32, "MODE": MODE_PARTIAL, "BLOCK_M": 16, "BLOCK_N": 64,
            },
            4,
            2,
            True,
        ),
        (
            "attn_full_verify",
            rel_attention_full_kernel,
            {
                "Q": "*bf16", "KVState": "*fp8e4nv", "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16",
                "BlockTable": "*i32", "SeqLens": "*i32", "Out": "*fp32", "Lse": "*fp32",
                "num_tokens": "i32", "table_width": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "PAGE": 64,
                "QLEN_CONST": 8, "NUM_SPLITS": 32, "MODE": MODE_PARTIAL, "BLOCK_M": 32, "BLOCK_N": 32,
            },
            4,
            2,
            True,
        ),
        # SWA attention (ring-buffered FP8 KV in states.seq_state)
        (
            "attn_swa_prefill",
            rel_attention_swa_cur_kernel,
            {
                "Q": "*bf16", "SeqState": "*fp8e4nv", "KCur": "*fp8e4nv", "VCur": "*fp8e4nv",
                "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16", "Lines": "*i32", "SeqLens": "*i32",
                "Out": "*bf16", "Lse": "*fp32",
                "num_tokens": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "RING": 1024, "WINDOW": 1024,
                "SEQ_STRIDE": SEQ_STATE_BYTES,
                "QLEN_CONST": 0, "NUM_SPLITS": 1, "MODE": MODE_GATED, "BLOCK_M": 32, "BLOCK_N": 32,
            },
            8,
            2,
            True,
        ),
        (
            "attn_swa_decode1",
            rel_attention_swa_ring_kernel,
            {
                "Q": "*bf16", "SeqState": "*fp8e4nv",
                "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16", "Lines": "*i32", "SeqLens": "*i32",
                "Out": "*fp32", "Lse": "*fp32",
                "num_tokens": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "RING": 1024, "WINDOW": 1024,
                "SEQ_STRIDE": SEQ_STATE_BYTES,
                "QLEN_CONST": 1, "NUM_SPLITS": 8, "MODE": MODE_PARTIAL, "BLOCK_M": 16, "BLOCK_N": 32,
            },
            4,
            2,
            True,
        ),
        (
            "attn_swa_decode8",
            rel_attention_swa_ring_kernel,
            {
                "Q": "*bf16", "SeqState": "*fp8e4nv",
                "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16", "Lines": "*i32", "SeqLens": "*i32",
                "Out": "*bf16", "Lse": "*fp32",
                "num_tokens": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "RING": 1024, "WINDOW": 1024,
                "SEQ_STRIDE": SEQ_STATE_BYTES,
                "QLEN_CONST": 1, "NUM_SPLITS": 1, "MODE": MODE_GATED, "BLOCK_M": 16, "BLOCK_N": 32,
            },
            4,
            2,
            True,
        ),
        (
            "attn_swa_verify",
            rel_attention_swa_cur_kernel,
            {
                "Q": "*bf16", "SeqState": "*fp8e4nv", "KCur": "*fp8e4nv", "VCur": "*fp8e4nv",
                "Rel": "*bf16", "Gate": "*bf16", "Sink": "*bf16", "Lines": "*i32", "SeqLens": "*i32",
                "Out": "*bf16", "Lse": "*fp32",
                "num_tokens": "i32", "stride_gate": "i32", "stride_sink": "i32",
                "sm_scale": "fp32", "v_scale": "fp32",
            },
            {
                "H": 24, "HKV": 8, "G": 3, "D": 128, "E": 1024, "RING": 1024, "WINDOW": 1024,
                "SEQ_STRIDE": SEQ_STATE_BYTES,
                "QLEN_CONST": 8, "NUM_SPLITS": 1, "MODE": MODE_GATED, "BLOCK_M": 32, "BLOCK_N": 32,
            },
            8,
            2,
            True,
        ),
    ]

    for num_splits in (8, 16, 32, 64):
        specs.append(
            (
                f"attn_combine_s{num_splits}",
                rel_decode_combine_kernel,
                {
                    "OPart": "*fp32", "LsePart": "*fp32", "Gate": "*bf16", "Sink": "*bf16", "Y": "*bf16",
                    "num_tokens": "i32", "stride_gate": "i32", "stride_sink": "i32",
                },
                {"H": 24, "D": 128, "NUM_SPLITS": num_splits, "BLOCK_S": num_splits, "BLOCK_D": 32},
                4,
                2,
                True,
            )
        )
    return specs


def build_dsl(out_dir: Path | None = None) -> dict:
    """Compile all SM120 Triton AOT attention kernels on CPU and emit attn_dsl.json."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[1] / "build/sm_120"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = get_dsl_specs()

    records = {}
    for name, kernel, sig, constants, warps, stages, fp_fusion in specs:
        compiled = triton.compile(
            ASTSource(kernel, signature=sig, constexprs=constants, attrs={}),
            target=GPUTarget("cuda", 120, 32),
            options={"num_warps": warps, "num_stages": stages, "enable_fp_fusion": fp_fusion},
        )
        if (
            compiled.metadata.global_scratch_size
            or compiled.metadata.profile_scratch_size
            or compiled.metadata.num_ctas != 1
        ):
            raise ValueError(f"{name}: unsupported scratch or cluster launch requirements")
        cubin_path = out_dir / f"{name}.cubin"
        cubin_path.write_bytes(compiled.asm["cubin"])
        (out_dir / f"{name}.ptx").write_text(compiled.asm["ptx"])
        records[name] = {
            "source": cubin_path.name,
            "sha256": hashlib.sha256(cubin_path.read_bytes()).hexdigest(),
            "entry": compiled.name,
            "shared_mem": compiled.metadata.shared,
            "block": [32 * warps, 1, 1],
            "signature": list(sig.values()),
            "arguments": list(sig.keys()),
            "abi_tail": ["i64", "i64"],
        }
        print(f"Built {name}: shared={compiled.metadata.shared} bytes", flush=True)

    meta = {"triton": triton.__version__, "target": "sm_120", "modules": records}
    (out_dir / "attn_dsl.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    build_dsl()
