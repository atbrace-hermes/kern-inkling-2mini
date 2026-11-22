# Inkling-2-Mini SM120 Optimization History

Workload-weighted GPU time measured by `kern bench` (`workload.toml`, CUPTI HES kernel timestamps) on a single **NVIDIA RTX PRO 5000 72GB Blackwell** (`SM120 / sm_120a`, 110 SMs, 1.2 TB/s GDDR7): **`33,093.2 → 11,032.1 GPU s` (`-66.66%` / `3.000×` speedup)**.

Final 76-prompt `standard_reference.parquet` decode-replay audit (BF16 reference vs FP8 KV):
- `kern-inkling-2mini` (`main@e6df2eb`): **mean KL = `1.07e-2`**, **p99 = `1.08e-1`** in **181.3 s**
- Official vLLM FP8 KV reference: mean KL = `1.118e-2`, p99 = `1.145e-1` in 463.0 s (`2.55×` faster at lower KL)

| # | Commit | Area | Cost (GPU s) | Delta | Guard Worst | e2e_smoke KL | Description |
|---|--------|------|-------------:|------:|------------:|-------------:|-------------|
| 0 | `39204d0` | baseline | 33093.2 | — | — | `1.08e-2` (PASS) | Baseline bundle (0 dropped) |
| 1 | `d7e022f` | dspark-attn | 26737.1 | -19.28% (-6385.5 s) | +0.05% | `1.08e-2` (PASS) | `dspark_attn_paged_partial` coalesced 16B `cp.async`, 0 spills, split prefix/live |
| 2 | `6dc5731` | rel-attn | 21283.4 | -20.44% (-5467.2 s) | +1.30% | `1.02e-2` (PASS) | `attn_full_prefill` 2-phase bulk+tail paged FP8 + 16B alignment |
| 3 | `49ca02f` | dspark-attn | 21009.0 | -1.27% (-269.8 s) | +0.13% | `1.02e-2` (PASS) | `dspark_attn_ring` coalesced 16B `cp.async`, 0 spills, split full/edge/live |
| 4 | `c7adb0e` | moe-gemm | 20783.7 | -0.81% (-169.1 s) | -0.03% | `1.02e-2` (PASS) | NVFP4 EVT-BF16 fused scaling, `n128` prefill tile, BF16 SwiGLU/combine |
| 5 | `fbb9e91` | moe-gemm | 20641.6 | -0.62% (-128.7 s) | +0.03% | `1.02e-2` (PASS) | 8x-deduplicated `moe_quant_*_v2`, hoisted MAE scales, warp-compacted `zero_tails` |
| 6 | `1057983` | norm-attnres | 18935.5 | -8.21% (-1694.1 s) | -8.70% | `1.02e-2` (PASS) | `norm_attnres.cu` bit-identical single-pass register-cached `block=256` AttnRes/CrossPass |
| 7 | `098080f` | dspark | 18812.5 | -0.59% (-110.9 s) | +0.19% | `1.02e-2` (PASS) | Fused SM120 TC `dspark_markov_fold_argmax` with L2 `evict_last` pinning |
| 8 | `1a8936c` | rel-attn | 16769.1 | -10.79% (-2028.7 s) | +0.15% | `1.05e-2` (PASS) | SM120 CUDA BF16 MMA `attn_swa_prefill` 16B `cp.async` double-buffered |
| 9 | `d02b375` | verify-attn | 15234.3 | -9.03% (-1511.8 s) | +0.06% | `1.05e-2` (PASS) | SM120 CUDA transposed-MMA split-KV `attn_swa_verify` 1-instr FP8->BF16 (`12.6×`) |
| 10 | `ba1c538` | bf16-gemm | 14898.3 | -1.96% (-297.6 s) | -0.09% | `1.04e-2` (PASS) | SM120 256-bit `v8.b32` direct-reg MMA/GEMV + PDL for `m3_gemm_bf16` (`M<=32`) |
| 11 | `51b66ec` | loop-ctrl | 14669.6 | -1.25% (-186.3 s) | -1.43% | `1.04e-2` (PASS) | Bit-identical single-pass `loop_controller` s1/s4 + 2-stream small-T RMSNorm |
| 12 | `41def25` | rel-attn | 13232.8 | -9.61% (-1407.4 s) | +1.87% | `1.04e-2` (STRICT PASS) | Exact-BF16 SM120 CUDA `attn_full_prefill_bulk` 3-stage FP8 SMEM + 1-instr cvt (246 TFLOP/s) |
| 13 | `8178c9d` | moe-gemm | 13102.5 | -0.82% (-108.7 s) | +0.09% | `1.04e-2` (STRICT PASS) | SM120 persistent low-M NVFP4 & MXFP8 grouped GEMMs (`T<=32`) |
| 14 | `d095a4f` | bf16-gemm | 12931.2 | -0.98% (-127.8 s) | +0.21% | `1.04e-2` (STRICT PASS) | Fused `wqrgskv`/`wqrgs` weight concat + dual-stream pair GEMM (`decode`/`round`, 352->132 GEMMs/step) |
| 15 | `a22f89b` | verify-attn | 11420.6 | -11.18% (-1436.9 s) | +0.03% | `1.04e-2` (STRICT PASS) | SM120 CUDA 2-phase bulk+tail split-KV `attn_full_verify` wave-matched 1-wave `S=41/20/10/5` |
| 16 | `d782f09` | moe-gemm | 11262.5 | -1.05% (-120.0 s) | +0.07% | `1.04e-2` (STRICT PASS) | NVFP4 `256x128x128` prefill tile, bit-exact fused MoE router, nozero low-M quant & streaming combine |
| 17 | `5ecf7cc` | dspark-attn | 11109.2 | -1.08% (-121.7 s) | +1.44% | `1.04e-2` (STRICT PASS) | M4 bundle: SM120 split-KV paged+ring attn, `v8.b32` MMA+GateUpSiLU (`M<=32`), fused bit-exact `verify_softcap_sample` (`10.8×`) |
| 18 | `e6df2eb` | norm-attnres | 11032.1 | -0.45% (-50.2 s) | +0.07% | `1.04e-2` (STRICT PASS) | Bit-identical 4-stream small-T `cross_pass_attn_res_f32` (`2.0×`) + full Blackwell PDL overlap |
