#!/usr/bin/env python3
"""Calibrate per-visit (0..47) FP8 E4M3 KV dequantization scales for Inkling-2-Mini.

Supports:
  --analytical-fallback : CPU-only dry-run emitting analytical L2-clip / RMSNorm upper bounds.
  --mode torch_loop     : GPU/CPU empirical calibration over multi-domain + long (>10K) prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Analytical upper bounds guaranteed by Inkling-2-Mini architecture:
#   - K is L2-clipped per KV head to ||k||_2 <= 22.627416997969522 (16 * sqrt(2))
#   - V is RMS-normalized without gamma (||v||_2 <= sqrt(128) = 11.313708498984761)
K_MAX_BOUND = 22.627416997969522
V_MAX_BOUND = 11.313708498984761
FP8_E4M3_MAX = 448.0
NUM_VISITS = 48

CALIBRATION_PROMPTS = [
    # English prose
    (
        "The history of numerical computing is closely tied to the evolution of floating-point "
        "representations. Modern accelerators rely on low-precision formats such as FP8 E4M3 "
        "and NVFP4 with block-level scaling to maximize memory bandwidth utilization while "
        "preserving model accuracy across long context windows."
    ),
    # Python / C++ code
    (
        "def binary_search(arr: list[int], target: int) -> int:\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        if arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
    # Math / reasoning
    (
        "Let x_1, x_2, ..., x_n be positive real numbers such that sum(x_i) = n. "
        "By the AM-GM inequality, prod(x_i) <= 1, with equality if and only if x_i = 1 for all i. "
        "Consider the integral of exp(-t^2) from 0 to infinity, which equals sqrt(pi) / 2."
    ),
    # Chinese text
    (
        "大语言模型在超长上下文推理场景中，KV Cache 往往成为显存容量的主要瓶颈。"
        "通过结合滑动窗口注意力（SWA）、短卷积状态缓存（sconv）以及 FP8 E4M3 量化，"
        "可以在单卡 72GB 显存内稳定支持 1M 上下文与多路并发推理。"
    ),
    # JSON / tool-calling format
    (
        '{"tool_call": {"name": "search_database", "arguments": {"query": "inkling2 loop '
        'architecture", "limit": 10, "filters": {"Quantization": ["NVFP4", "MXFP8"]}}}}'
    ),
]


def compute_analytical_scales() -> dict[str, dict[str, float]]:
    k_scale = K_MAX_BOUND / FP8_E4M3_MAX
    v_scale = V_MAX_BOUND / FP8_E4M3_MAX
    return {str(v): {"k": k_scale, "v": v_scale} for v in range(NUM_VISITS)}


def scales_from_maxima(
    maxima: dict[str, dict[str, float]],
    headroom: float,
) -> dict[str, dict[str, float]]:
    out = {}
    for v in range(NUM_VISITS):
        key = str(v)
        m = maxima[key]
        k_val = min(max(m["k"] * headroom, 1e-6), K_MAX_BOUND) / FP8_E4M3_MAX
        v_val = min(max(m["v"] * headroom, 1e-6), V_MAX_BOUND) / FP8_E4M3_MAX
        out[key] = {"k": max(k_val, 1e-8), "v": max(v_val, 1e-8)}
    return out


def _rms_norm(x, weight, eps: float, out_dtype=None):
    import torch

    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x_f * rstd
    if weight is not None:
        y = y * weight.float()
    return y if out_dtype is None else y.to(out_dtype)


def _l2_clip(x, scale, threshold: float, eps: float, tau=None):
    import torch

    x_f = x.float()
    if scale is not None:
        x_f = x_f * scale.float()
    sum_sq = x_f.pow(2).sum(dim=-1, keepdim=True)
    norm = torch.where(sum_sq > 0, sum_sq * torch.rsqrt(sum_sq.clamp(min=1e-30)), torch.zeros_like(sum_sq))
    coeff = torch.clamp(threshold / (norm + eps), max=1.0)
    y = (x_f * coeff).to(x.dtype)
    if tau is not None:
        y = (y.float() * tau).to(x.dtype)
    return y


def _sconv_torch(x, w_static, dyn=None, is_dynamic: bool = False):
    import torch

    t_len, c_dim = x.shape
    x_f = x.float()
    acc = torch.zeros_like(x_f)
    if is_dynamic:
        dyn_v = dyn.float().view(t_len, 8, 4)
        for lag in range(4):
            w_l = w_static[lag].float().unsqueeze(0) + dyn_v[:, :, lag].repeat_interleave(128, dim=-1)
            if lag == 0:
                acc = x_f * w_l
            else:
                x_prev = torch.zeros_like(x_f)
                if t_len > lag:
                    x_prev[lag:] = x_f[:-lag]
                acc = acc + x_prev * w_l
        scale = 9.0**-0.5
    else:
        w_s = w_static.view(c_dim, 4).float()
        for k in range(4):
            lag = 3 - k
            if lag == 0:
                x_prev = x_f
            else:
                x_prev = torch.zeros_like(x_f)
                if t_len > lag:
                    x_prev[lag:] = x_f[:-lag]
            acc = acc + x_prev * w_s[:, k].unsqueeze(0)
        scale = 5.0**-0.5
    return ((acc + x_f) * scale).to(x.dtype)


_NVFP4_LUT_VALS = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_NVFP4_LUT_CACHE: dict[str, object] = {}


def _get_nvfp4_lut(device):
    import torch

    key = str(device)
    lut = _NVFP4_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(_NVFP4_LUT_VALS, device=device, dtype=torch.float32)
        _NVFP4_LUT_CACHE[key] = lut
    return lut


def _dequant_nvfp4(q_u8, scale_fp8, amax_f32, denom_f32, block_size: int = 16):
    import torch

    n_rows, half_k = q_u8.shape
    lo = (q_u8 & 0x0F).to(torch.int32)
    hi = ((q_u8 >> 4) & 0x0F).to(torch.int32)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(n_rows, half_k * 2)
    lut = _get_nvfp4_lut(q_u8.device)
    vals = lut[unpacked].view(n_rows, -1, block_size)
    sf = scale_fp8.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    outer = amax_f32.float() / denom_f32.float()
    return (vals * sf).reshape(n_rows, half_k * 2) * outer


def _dequant_mxfp8(q_fp8, scale_ue8m0, block_size: int = 32):
    import torch

    n_rows, k_dim = q_fp8.shape
    vals = q_fp8.view(torch.float8_e4m3fn).float().view(n_rows, k_dim // block_size, block_size)
    exp = scale_ue8m0.to(torch.int32) & 0xFF
    # IEEE-754 FP32 bitcast matching moe_sm120.py / mxfp8.py:
    #   0   -> 0x00400000 (subnormal FP32: 0.5 * 2^-126 == 2^-127, NOT 2^-126)
    #   255 -> 0x7FC00000 (quiet NaN)
    #   else -> exp << 23
    bits = torch.where(
        exp == 0,
        torch.full_like(exp, 0x00400000),
        torch.where(exp == 255, torch.full_like(exp, 0x7FC00000), exp << 23),
    )
    sf = bits.view(torch.float32).unsqueeze(-1)
    return (vals * sf).reshape(n_rows, k_dim)


def calibrate_torch_loop(
    checkpoint: Path,
    headroom: float,
    device: str,
    max_tokens: int | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.loads((checkpoint / "config.json").read_text())["text_config"]
    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    local_ids = set(cfg["local_layer_ids"])
    core_start = cfg["loop_config"]["core_start"]
    core_end = cfg["loop_config"]["core_end"]
    num_passes = cfg["loop_config"]["num_passes"]
    block_size = cfg["n_layers_per_attn_res_block"]
    eps = cfg["rms_norm_eps"]
    qk_thresh = cfg["qk_l2_clip_threshold"]
    rel_thresh = cfg["rel_l2_clip_threshold"]
    n_floor = cfg["log_scaling_n_floor"]
    tau_alpha = cfg["log_scaling_alpha"]
    gate_cap = cfg["mlp_gate_softcap"]
    up_cap = cfg["mlp_up_softcap"]
    route_scale = cfg["route_scale"]
    topk = cfg["num_experts_per_tok"]

    weights = {}
    target_dev = torch.device(device)
    for sf_path in sorted(checkpoint.glob("model*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual.") or k in ("model.llm.unembed.weight", "model.llm.norm.weight"):
                    continue
                t = f.get_tensor(k)
                weights[k] = t.to(target_dev) if target_dev.type != "cpu" else t
                del t

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    long_text = (CALIBRATION_PROMPTS[0] + "\n" + CALIBRATION_PROMPTS[1] + "\n") * 220
    prompt_ids = []
    for text in [*CALIBRATION_PROMPTS, long_text]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if max_tokens is not None:
            ids = ids[:max_tokens]
        if len(ids) > 0:
            prompt_ids.append(torch.tensor(ids, dtype=torch.long, device=device))

    visits: list[tuple[int, int]] = []
    for p in range(core_start):
        visits.append((p, 0))
    for cp in range(num_passes):
        for p in range(core_start, core_end):
            visits.append((p, cp))
    for p in range(core_end, num_layers):
        visits.append((p, 0))
    assert len(visits) == NUM_VISITS

    maxima = {str(v): {"k": 0.0, "v": 0.0} for v in range(NUM_VISITS)}
    attn_res_scale = hidden_size**-0.5

    def get_w(name: str):
        return weights[name]

    def attn_res_mix(stack, partial, q_w, norm_w, addend=None, out_dtype=torch.bfloat16):
        # stack: [B, T, H], partial: [T, H], q_w: [Q, H] (already scaled by H^-0.5)
        q_w = q_w.view(-1, hidden_size).float()
        blocks = [stack[b] for b in range(stack.shape[0])]
        blocks.append(partial if addend is None else (partial.float() + addend.float()).to(partial.dtype))
        stacked = torch.stack(blocks, dim=0)  # [B+1, T, H]
        k_norm = _rms_norm(stacked, None, eps, out_dtype=None)
        logits = torch.einsum("qh,bth->qbt", q_w, k_norm)
        probs = torch.softmax(logits, dim=1)
        mixed = torch.einsum("qbt,bth->qth", probs, k_norm)
        if norm_w is None:
            return [mixed[j] for j in range(mixed.shape[0])]
        return [
            _rms_norm(mixed[j], norm_w[j] if norm_w.dim() == 2 else norm_w, eps, out_dtype=out_dtype)
            for j in range(mixed.shape[0])
        ]

    # Precompute scaled cross-pass message queries [core_len, 4, H] matching loop.py:168-176
    cp_q = get_w("model.llm.middle_cycle_cross_pass_queries").float().view(-1, 2, hidden_size)
    cp_kv = get_w("model.llm.middle_cycle_cross_pass_kv_queries").float().view(-1, 2, hidden_size)
    msg_queries = torch.stack((cp_q[:, 0], cp_kv[:, 0], cp_kv[:, 1], cp_q[:, 1]), dim=1) * attn_res_scale

    with torch.inference_mode():
        embed_w = get_w("model.llm.embed.weight")
        embed_norm_w = get_w("model.llm.embed_norm.weight")

        for ids in prompt_ids:
            t_len = ids.shape[0]
            pos = torch.arange(t_len, device=device, dtype=torch.float32)
            tau_all = 1.0 + tau_alpha * torch.log(torch.clamp((pos + 1.0) / n_floor, min=1.0))
            tau_all = tau_all.unsqueeze(-1).unsqueeze(-1)

            h0 = _rms_norm(embed_w[ids], embed_norm_w, eps, out_dtype=torch.bfloat16)
            max_blocks = (NUM_VISITS + block_size - 1) // block_size
            stack = torch.empty((max_blocks + 1, t_len, hidden_size), dtype=torch.bfloat16, device=device)
            stack[0].copy_(h0)
            num_blocks = 0
            bank = None
            prelude_blocks = None
            prelude_partial = None
            prelude_dests = None
            inputs = None

            for v_idx, (phys, cp) in enumerate(visits):
                pfx = f"model.llm.layers.{phys}"
                is_local = phys in local_ids
                looped = cp > 0

                if phys == core_start and not looped:
                    prelude_blocks = num_blocks
                    prelude_partial = stack[num_blocks].clone()
                elif phys == core_start and looped:
                    assert prelude_blocks is not None and prelude_partial is not None
                    bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                    num_blocks = prelude_blocks
                    stack[num_blocks].copy_(prelude_partial)
                    if prelude_dests is None:
                        q_pre = get_w(f"model.llm.attn_res.attn_queries.{phys - 1}").float() * attn_res_scale
                        prelude_dests = attn_res_mix(stack[:num_blocks], prelude_partial, q_pre, None, out_dtype=None)
                    inputs = prelude_dests

                partial = stack[num_blocks]
                if is_local:
                    attn_nw = get_w(f"{pfx}.attn_norm.weight")
                else:
                    attn_nw = torch.stack([
                        get_w(f"{pfx}.q_input_norm.weight"),
                        get_w(f"{pfx}.k_input_norm.weight"),
                        get_w(f"{pfx}.v_input_norm.weight"),
                    ])

                messages = None
                if looped:
                    assert bank is not None and inputs is not None
                    mq = msg_queries[phys - core_start]
                    messages = attn_res_mix(bank[:1], bank[1], mq, None, out_dtype=None)
                    dests = [inputs[0]] * 3 if len(inputs) == 1 else inputs[:3]
                    attn_inputs = []
                    for s_idx, s_name in enumerate(("attn", "k", "v")):
                        dest = dests[s_idx]
                        msg = messages[s_idx]
                        pair = torch.stack((dest, msg), dim=1)
                        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
                        norm_pair = pair.to(torch.bfloat16).flatten(1)
                        h_ctrl = norm_pair @ get_w(f"{pfx}.controllers.{s_name}.down_weight").t()
                        gain = h_ctrl.float() @ get_w(f"{pfx}.controllers.{s_name}.gain_weight").t().float()
                        forget = h_ctrl.float() @ get_w(f"{pfx}.controllers.{s_name}.forget_weight").t().float()
                        blended = (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * msg
                        nw = attn_nw if attn_nw.dim() == 1 else attn_nw[s_idx]
                        attn_inputs.append(_rms_norm(blended, nw, eps, out_dtype=torch.bfloat16))
                    x_q, x_k, x_v = attn_inputs
                else:
                    if inputs is None:
                        attn_inputs = [
                            _rms_norm(partial, attn_nw if attn_nw.dim() == 1 else attn_nw[j], eps, out_dtype=torch.bfloat16)
                            for j in range(1 if is_local else 3)
                        ]
                    else:
                        attn_inputs = inputs
                    x_q = attn_inputs[0]
                    x_k = attn_inputs[0] if is_local else attn_inputs[1]
                    x_v = attn_inputs[0] if is_local else attn_inputs[2]

                wq = get_w(f"{pfx}.attn.wq_du.weight")
                wk = get_w(f"{pfx}.attn.wk_dv.weight")
                wv = get_w(f"{pfx}.attn.wv_dv.weight")
                wr = get_w(f"{pfx}.attn.wr_du.weight")
                wg = get_w(f"{pfx}.attn.gate_attn.weight")
                ws = get_w(f"{pfx}.attn.attn_sink_proj.weight")
                wo = get_w(f"{pfx}.attn.wo_ud.weight")

                q_proj = (x_q @ wq.t()).view(t_len, 24, 128)
                k_proj = x_k @ wk.t()
                v_proj = x_v @ wv.t()
                r_proj = (x_q @ wr.t()).view(t_len, 24, 16)
                gate = (x_q @ wg.t()).view(t_len, 24, 128)
                sink = x_q @ ws.t()

                if is_local:
                    k_conv = _sconv_torch(k_proj, get_w(f"{pfx}.attn.k_sconv.weight"), is_dynamic=False)
                    v_conv = _sconv_torch(v_proj, get_w(f"{pfx}.attn.v_sconv.weight"), is_dynamic=False)
                else:
                    k_dyn = x_k @ get_w(f"{pfx}.attn.k_sconv.weight_proj").t()
                    v_dyn = x_v @ get_w(f"{pfx}.attn.v_sconv.weight_proj").t()
                    k_conv = _sconv_torch(k_proj, get_w(f"{pfx}.attn.k_sconv.static_w"), k_dyn, is_dynamic=True)
                    v_conv = _sconv_torch(v_proj, get_w(f"{pfx}.attn.v_sconv.static_w"), v_dyn, is_dynamic=True)

                # Clip / norm exactly as attn_prologue
                kf = k_conv.view(t_len, 8, 128).float()
                kn = kf.pow(2).sum(dim=-1, keepdim=True).sqrt()
                k_post = kf * torch.clamp(qk_thresh / (kn + eps), max=1.0)
                v_post = _rms_norm(v_conv.view(t_len, 8, 128), None, eps, out_dtype=None)

                maxima[str(v_idx)]["k"] = max(maxima[str(v_idx)]["k"], k_post.abs().max().item())
                maxima[str(v_idx)]["v"] = max(maxima[str(v_idx)]["v"], v_post.abs().max().item())

                tau_use = None if is_local else tau_all
                q_post = _l2_clip(q_proj, get_w(f"{pfx}.attn.q_l2_clip.scale"), qk_thresh, eps, tau=tau_use)
                r_post = _l2_clip(r_proj, get_w(f"{pfx}.attn.rel_l2_clip.scale"), rel_thresh, eps, tau=tau_use)

                rel_proj = get_w(f"{pfx}.attn.rel_logits_proj.proj")
                rel_bias = (r_post.float() @ rel_proj.float()).to(torch.bfloat16)
                k_rep = k_post.repeat_interleave(3, dim=1)
                v_rep = v_post.repeat_interleave(3, dim=1)
                q_f = q_post.float() * (128.0**-0.5)

                attn_out = torch.empty((t_len, 24, 128), device=device, dtype=torch.float32)
                lse_nat = torch.empty((t_len, 24), device=device, dtype=torch.float32)
                blk = 512
                for q0 in range(0, t_len, blk):
                    q1 = min(q0 + blk, t_len)
                    k0 = max(0, q0 - 1023) if is_local else 0
                    scores = torch.einsum("qhd,khd->hqk", q_f[q0:q1], k_rep[k0:q1])
                    q_pos = torch.arange(q0, q1, device=device)[:, None]
                    k_pos = torch.arange(k0, q1, device=device)[None, :]
                    dist = q_pos - k_pos
                    vis = dist >= 0
                    if is_local:
                        vis = vis & (dist < 1024)
                    in_ext = vis & (dist < 1024)
                    safe_d = torch.where(in_ext, dist, torch.zeros_like(dist))
                    b_slice = rel_bias[q0:q1].permute(1, 0, 2)
                    bias_vals = torch.gather(b_slice, 2, safe_d.unsqueeze(0).expand(24, -1, -1)).float()
                    scores = scores + torch.where(in_ext.unsqueeze(0), bias_vals, torch.zeros_like(bias_vals))
                    scores = torch.where(vis.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
                    m_max = scores.max(dim=-1, keepdim=True).values
                    exp_s = torch.exp(scores - m_max)
                    sum_s = exp_s.sum(dim=-1, keepdim=True)
                    probs = exp_s / sum_s
                    attn_out[q0:q1] = torch.einsum("hqk,khd->qhd", probs, v_rep[k0:q1])
                    lse_nat[q0:q1] = (m_max.squeeze(-1) + torch.log(sum_s.squeeze(-1))).t()

                gated = (attn_out * torch.sigmoid(gate.float()) * torch.sigmoid(lse_nat - sink.float()).unsqueeze(-1)).to(
                    torch.bfloat16
                )
                attn_partial = gated.view(t_len, 3072) @ wo.t()
                attn_out_h = _sconv_torch(attn_partial, get_w(f"{pfx}.attn_sconv.weight"), is_dynamic=False)

                first = (v_idx % block_size == 0)
                mlp_q = get_w(f"model.llm.attn_res.mlp_queries.{phys}").float() * attn_res_scale
                mlp_nw = get_w(f"{pfx}.mlp_norm.weight")
                if messages is not None:
                    mlp_dest = attn_res_mix(stack[: num_blocks + first], partial if not first else attn_out_h, mlp_q, None, None if first else attn_out_h, out_dtype=None)[0]
                    pair = torch.stack((mlp_dest, messages[3]), dim=1)
                    pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
                    h_ctrl = pair.to(torch.bfloat16).flatten(1) @ get_w(f"{pfx}.controllers.mlp.down_weight").t()
                    gain = h_ctrl.float() @ get_w(f"{pfx}.controllers.mlp.gain_weight").t().float()
                    forget = h_ctrl.float() @ get_w(f"{pfx}.controllers.mlp.forget_weight").t().float()
                    blended = (1.0 - torch.tanh(forget)) * mlp_dest + torch.tanh(gain) * messages[3]
                    mlp_in = _rms_norm(blended, mlp_nw, eps, out_dtype=torch.bfloat16)
                elif first:
                    mlp_in = attn_res_mix(stack[: num_blocks + 1], attn_out_h, mlp_q, mlp_nw)[0]
                else:
                    mlp_in = attn_res_mix(stack[:num_blocks], partial, mlp_q, mlp_nw, attn_out_h)[0]

                # MoE forward (stacked w13/w2 NVFP4 or MXFP8)
                r_logits = mlp_in.float() @ get_w(f"{pfx}.mlp.gate.weight").float().t()
                r_bias = get_w(f"{pfx}.mlp.gate.bias").float()
                sel = torch.sigmoid(r_logits) + r_bias
                topk_idx = torch.topk(sel, topk, dim=-1).indices
                topk_logits = torch.gather(r_logits, 1, topk_idx)
                topk_log_sig = F.logsigmoid(topk_logits)
                topk_w = torch.softmax(topk_log_sig, dim=-1) * route_scale

                w13_w = get_w(f"{pfx}.mlp.experts.w13_weight")
                w13_s = get_w(f"{pfx}.mlp.experts.w13_weight_scale")
                w2_w = get_w(f"{pfx}.mlp.experts.w2_weight")
                w2_s = get_w(f"{pfx}.mlp.experts.w2_weight_scale")
                is_nvfp4 = f"{pfx}.mlp.experts.w13_weight_amax" in weights
                if is_nvfp4:
                    w13_amax = get_w(f"{pfx}.mlp.experts.w13_weight_amax")
                    w13_den = get_w(f"{pfx}.mlp.experts.w13_weight_outer_scale_denominator")
                    w2_amax = get_w(f"{pfx}.mlp.experts.w2_weight_amax")
                    w2_den = get_w(f"{pfx}.mlp.experts.w2_weight_outer_scale_denominator")

                moe_acc = torch.zeros((t_len, hidden_size), device=device, dtype=torch.float32)
                for eid in topk_idx.unique().tolist():
                    tok_rows, slot_idx = (topk_idx == eid).nonzero(as_tuple=True)
                    if tok_rows.numel() == 0:
                        continue
                    x_e = mlp_in[tok_rows].float()
                    if is_nvfp4:
                        w13_e = _dequant_nvfp4(w13_w[eid], w13_s[eid], w13_amax[eid], w13_den)
                        w2_e = _dequant_nvfp4(w2_w[eid], w2_s[eid], w2_amax[eid], w2_den)
                    else:
                        w13_e = _dequant_mxfp8(w13_w[eid], w13_s[eid])
                        w2_e = _dequant_mxfp8(w2_w[eid], w2_s[eid])
                    gu = x_e @ w13_e.t()
                    gate_e = gu[:, 0::2]
                    up_e = gu[:, 1::2]
                    sig_g = torch.sigmoid(gate_e)
                    g_cap = gate_cap * torch.tanh(gate_e / gate_cap)
                    u_cap = up_cap * torch.tanh(up_e / up_cap)
                    h_act = ((g_cap * sig_g) * u_cap * topk_w[tok_rows, slot_idx].unsqueeze(-1)).to(torch.bfloat16).float()
                    moe_acc.index_add_(0, tok_rows, h_act @ w2_e.t())

                moe_out = moe_acc.to(torch.bfloat16)
                mlp_conv = _sconv_torch(moe_out, get_w(f"{pfx}.mlp_sconv.weight"), is_dynamic=False)
                residual = None if first else partial
                if residual is None:
                    num_blocks += 1
                    next_h = (attn_out_h.float() + mlp_conv.float()).to(torch.bfloat16)
                else:
                    next_h = ((residual.float() + attn_out_h.float()) + mlp_conv.float()).to(torch.bfloat16)
                stack[num_blocks].copy_(next_h)

                if v_idx + 1 < NUM_VISITS:
                    next_phys, next_pass = visits[v_idx + 1]
                    next_q = get_w(f"model.llm.attn_res.attn_queries.{next_phys - 1}").float() * attn_res_scale
                    if next_pass > 0:
                        inputs = attn_res_mix(stack[:num_blocks], stack[num_blocks], next_q, None, out_dtype=None)
                    else:
                        if next_phys in local_ids:
                            nnw = get_w(f"model.llm.layers.{next_phys}.attn_norm.weight")
                        else:
                            nnw = torch.stack([
                                get_w(f"model.llm.layers.{next_phys}.q_input_norm.weight"),
                                get_w(f"model.llm.layers.{next_phys}.k_input_norm.weight"),
                                get_w(f"model.llm.layers.{next_phys}.v_input_norm.weight"),
                            ])
                        inputs = attn_res_mix(stack[:num_blocks], stack[num_blocks], next_q, nnw)

    return scales_from_maxima(maxima, headroom), maxima


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("KERN_WEIGHTS", Path(__file__).resolve().parents[2] / "weights" / "Inkling-2-Mini")),
    )
    parser.add_argument("--out", type=Path, default=Path("results/kv-scales.json"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--headroom", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--mode", choices=("torch_loop",), default="torch_loop")
    parser.add_argument("--analytical-fallback", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.analytical_fallback:
        scales = compute_analytical_scales()
        args.out.write_text(json.dumps(scales, indent=2) + "\n")
        print(f"Wrote analytical fallback KV scales (48 visits) to {args.out}")
        return

    scales, maxima = calibrate_torch_loop(args.checkpoint, args.headroom, args.device, args.max_tokens)

    args.out.write_text(json.dumps(scales, indent=2) + "\n")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"headroom": args.headroom, "maxima": maxima, "scales": scales}, indent=2) + "\n")
    print(f"Wrote calibrated KV scales (48 visits) to {args.out}")


if __name__ == "__main__":
    main()
