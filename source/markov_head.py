# SPDX-License-Identifier: Apache-2.0
"""Inkling2MarkovHead Triton kernels for SM120 AOT."""

import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as tldevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as tldevice


@triton.jit
def _round_bf16(x):
    """Exact FP32 -> BF16 -> FP32 round-to-nearest-even without FMA folding."""
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _skip_anchor_reorder_kernel(
    X,
    Y,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    """Gather rows `1..BLOCK_SIZE-1` from `[seqs * BLOCK_SIZE, H]` into `[seqs * (BLOCK_SIZE - 1), H]`."""
    r = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * B + tl.arange(0, B)
    mask = d < H
    seq = r // (BLOCK_SIZE - 1)
    step = r % (BLOCK_SIZE - 1)
    src = seq * BLOCK_SIZE + 1 + step
    val = tl.load(X + src * H + d, mask=mask, other=0.0)
    tl.store(Y + r * H + d, val, mask=mask)


@triton.jit
def _markov_ctx_concat_kernel(
    StepHidden,
    FRnn,
    Out,
    H: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    B: tl.constexpr,
):
    """Concatenate `[step_hidden[seq, step], f_rnn[seq]]` -> `[seqs * NUM_STEPS, 2 * H]`."""
    r = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * B + tl.arange(0, B)
    mask = d < H
    seq = r // NUM_STEPS
    h_val = tl.load(StepHidden + r * H + d, mask=mask, other=0.0)
    f_val = tl.load(FRnn + seq * H + d, mask=mask, other=0.0)
    tl.store(Out + r * (2 * H) + d, h_val, mask=mask)
    tl.store(Out + r * (2 * H) + H + d, f_val, mask=mask)


@triton.jit
def _markov_add_bias_kernel(
    Ctx,
    Bias,
    D3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place FP32 `Ctx[r, :] += Bias[:]` for `[seqs * 7, 768]`."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    mask = c < D3
    x = tl.load(Ctx + r * D3 + c, mask=mask, other=0.0)
    b = tl.load(Bias + c, mask=mask, other=0.0)
    tl.store(Ctx + r * D3 + c, x + b, mask=mask)


@triton.jit
def _markov_state_concat_kernel(
    State,
    Prev,
    Table,
    Out,
    step_off,
    id_stride,
    RANK: tl.constexpr,
):
    """Gather `Table[Prev[seq * id_stride + step_off]]` and concat `[State[seq], emb]` -> `Out [seqs, 2 * RANK]`."""
    seq = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    tok = tl.load(Prev + seq * id_stride + step_off).to(tl.int64)
    s_val = tl.load(State + seq * RANK + c)
    e_val = tl.load(Table + tok * RANK + c)
    tl.store(Out + seq * (2 * RANK) + c, s_val)
    tl.store(Out + seq * (2 * RANK) + RANK + c, e_val)


@triton.jit
def _markov_gate_add_kernel(
    Ctx,
    StatePart,
    State,
    OutState,
    Latent,
    step,
    NUM_STEPS: tl.constexpr,
    RANK: tl.constexpr,
):
    """Fused FP32 `Ctx[seq, step] + StatePart[seq]` + bitwise-exact BF16 Markov gate."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    ctx_base = Ctx + (row * NUM_STEPS + step) * (3 * RANK) + c
    sp_base = StatePart + row * (3 * RANK) + c

    gate_raw = _round_bf16(tl.load(ctx_base) + tl.load(sp_base))
    cand_raw = _round_bf16(tl.load(ctx_base + RANK) + tl.load(sp_base + RANK))
    out_raw = _round_bf16(tl.load(ctx_base + 2 * RANK) + tl.load(sp_base + 2 * RANK))

    state = tl.load(State + row * RANK + c).to(tl.float32)
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate_raw)))
    cand = _round_bf16(tldevice.tanh(cand_raw))
    keep = _round_bf16(gate * state)
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    tl.store(OutState + row * RANK + c, (keep + update).to(tl.bfloat16))
    tl.store(Latent + row * RANK + c, tldevice.tanh(out_raw).to(tl.bfloat16))


@triton.jit
def _add_softcap_mask_kernel(
    Base,
    Bias,
    FoldBias,
    Out,
    step,
    softcap,
    NUM_STEPS: tl.constexpr,
    N: tl.constexpr,
    UNPADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute `bf16(softcap * tanh(bf16(base[s, step] + bf16(bias[s] + fold_bias)) / softcap))`,
    masking `col >= UNPADDED` to `-inf`."""
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(Base + (row * NUM_STEPS + step) * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bias + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    fb = tl.load(FoldBias + cols, mask=mask, other=0.0).to(tl.float32)
    bias_bf16 = (b + fb).to(tl.bfloat16).to(tl.float32)
    x = (a + bias_bf16).to(tl.bfloat16).to(tl.float32)
    x = x / softcap
    x = tldevice.tanh(x)
    x = (x * softcap).to(tl.bfloat16)
    x = tl.where(cols < UNPADDED, x, float("-inf"))
    tl.store(Out + row * N + cols, x.to(tl.bfloat16), mask=mask)
