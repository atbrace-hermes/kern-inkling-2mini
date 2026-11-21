# kern-inkling-2mini

Day-0 text-only + DSpark MTP support for **Inkling-2-Mini** (`inkling2_loop`) on a single `NVIDIA RTX PRO 5000 72GB Blackwell` (**SM120 / `sm_120a`**) using `kern` (`schema_version = 7`). Vision (`model.visual.*`) is excluded.

## Architecture Summary

- **48 Logical Layer Visits from 24 Physical Layers** (`H = 2304`, `n_layers_per_attn_res_block = 6`):
  - Prelude: physical `0..5` (1x, visits `0..5`)
  - Core: physical `6..17` (3 passes, visits `6..17`, `18..29`, `30..41`) with Cross-Pass AttnRes reads over the previous pass's 2 core blocks and rank-64 FP32 no-FMA gated loop controllers across `(attn, k, v, mlp)`
  - Coda: physical `18..23` (1x, visits `42..47`)
- **Hybrid Attention & `sconv`**:
  - 40 local SWA visits (`W = 1024`) + 2 MTP SWA layers in per-sequence FP8 ring buffers (`bytes_per_seq`)
  - 8 full relative-bias GQA visits (physical `5, 11, 17, 23`) + 1 MTP full layer in paged FP8 KV cache (`18 KiB/token`, supporting `1M` / `1048576` context and `max_seqs = 8` in 72 GB VRAM)
  - 4-stream short convolution (`k, v, attn, mlp`, `W = 4`) across 48 visits
- **Mixed MoE Quantization**:
  - Layers `0..19`: NVFP4 (`fouroversix_mae`, block=16 + FP32 outer scales), 128 experts, top-8, `inter = 1152`
  - Layers `20..23`: MXFP8 (`mxfp8_e4m3`, block=32 `ue8m0`)
- **Unembed & DSpark MTP**:
  - `extern:cublas_bf16_tn_f32` -> FP32 `30.0 * tanh(x / 30.0)` + vocab mask `>= 200058 -> -inf`
  - 3-layer BF16 bidirectional DSpark drafter (`block_size = 8`, `target_layer_ids = [5, 23, 41, 48]`, `Inkling2MarkovHead` rank-256 `rnn_hidden`)

## Build & Verify Commands

Toolchain paths:
- Python: `/workspace/kern-qwen38/.venv/bin/python` (Python 3.12, `torch==2.14.0`, `triton==3.8.0`, `safetensors==0.8.0`)
- `kern`: `/workspace/kern/target/release/kern` (and `kern-serve`)

```sh
PY=/workspace/kern-qwen38/.venv/bin/python
KERN=/workspace/kern/target/release/kern

# 1. Inspect checkpoint (CPU-only)
"$PY" scripts/inspect_checkpoint.py --checkpoint /workspace/weights/Inkling-2-Mini --strict

# 2. AOT-compile SM120 CUDA & Triton kernels (CPU-only, per milestone or all)
"$PY" scripts/build.py --milestone M1 --update
"$PY" scripts/build_dsl.py --milestone M1

# 3. Generate bundle & static verify (M5)
"$PY" scripts/gen.py --checkpoint /workspace/weights/Inkling-2-Mini --out out/inkling-2-mini
"$KERN" verify out/inkling-2-mini/model.json

# 4. Run unit tests (GPU)
"$PY" -m unittest discover -s tests/unit -v

# 5. Serve (1M context, FP8 KV, max_seqs=8)
"$KERN" server --config out/inkling-2-mini/kern.toml inkling-2-mini --port 8000 --max-seqs 8 --seq-slots 32
```

## Precision & Memory Budget

See `precision.toml`:
- Text + MTP weights: `~18.18 GiB`
- Paged FP8 full KV (`9` full layers): `18 KiB/token` (`18.00 GiB` @ `1048576` tokens)
- Per-sequence SWA + `sconv` ring states: `~86 MiB/seq` (`< 0.7 GiB` for `max_seqs = 8`)
