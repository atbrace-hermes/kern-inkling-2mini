# kern-inkling-2mini

Text-only + DSpark MTP (`k=7`) inference bundle and custom SM120 (`sm_120a`) CUDA/Triton kernels for **Inkling-2-Mini** (`inkling2_loop`) on a single **NVIDIA RTX PRO 5000 72GB Blackwell** using [`kern`](https://github.com/pegainfer-project/kern) (`schema_version = 7`). Vision (`model.visual.*`) is excluded.

## Highlights (Single RTX PRO 5000 72GB, SM120)

- **3.000× workload kernel speedup** across 18 verified optimization landings (`33,093.2 → 11,032.1 GPU s`, `-66.66%` on `workload.toml`; see [`results/optimization-history.md`](results/optimization-history.md)).
- **Gold-standard numerical parity** (76-prompt `standard_reference.parquet`, BF16 reference vs FP8 KV, zero extra weight quantization):
  - `kern-inkling-2mini`: **mean KL = `1.07e-2`**, **p99 = `1.08e-1`** (`181.3 s`)
  - Official vLLM FP8 KV reference: mean KL = `1.118e-2`, p99 = `1.145e-1` (`463.0 s`)
- **Full 1M (`1,048,576`) context + `max_seqs = 8`** inside 72 GB VRAM with static per-visit FP8 E4M3 KV cache (`18 KiB/token` across 9 full-attention layers).

## End-to-End Serving Benchmarks (Single RTX PRO 5000 72GB, SM120)

Measured with NVIDIA AIPerf (`chunk=2048`, `max_seqs=8`, DSpark `k=7`, CUDA Graph ON, explicit KV cache reset per cell):
- `kern`: FP8 E4M3 KV cache (`main`)
- `vLLM`: FP8 KV cache (`vllm/vllm-openai:inkling-2-mini` + SM120 `_C_stable_libtorch.abi3.so` from `vllm/vllm-openai:nightly`)
- `SGLang`: BF16 KV cache (`lmsysorg/sglang:inkling-2-mini`, SM120 `fa4` requires `K.dtype == Q.dtype`)

| Workload | Metric | **kern** | **SGLang** | **vLLM** |
|---|---|---:|---:|---:|
| Cold Prefill `ISL=4K, c=1` (0% hit) | TTFT p50 | **267.1 ms** | 370.8 ms | 395.9 ms |
| Cold Prefill `ISL=32K, c=1` (0% hit) | TTFT p50 | **2,191 ms** | 3,083 ms | 3,107 ms |
| Cold Prefill `ISL=128K, c=1` (0% hit) | TTFT p50 | **12.46 s** | 17.74 s | 18.94 s |
| Decode `ISL=512, OSL=512, c=1 / c=8` | Total tok/s | **177.3 / 693.3** | 133.8 / 548.9 | 142.4 / 477.9 |
| Decode `ISL=8K, OSL=512, c=1 / c=8` | Total tok/s | **149.4 / 383.5** | 124.7 / 318.2 | 118.3 / 261.9 |
| Decode `ISL=32K, OSL=512, c=1 / c=8` | Total tok/s | **94.8 / 161.7** | 78.1 / 120.3 | 77.9 / 117.6 |
| `SpecBench` `temp=0, c=1 / c=8` (80 reqs) | Total tok/s (AL) | **193.1 / 742.3** (2.67 / 2.74) | 166.1 / 636.7 (2.72 / 2.83) | 147.7 / 437.8 (2.71 / 2.79) |
| `HumanEval` `temp=0, c=1 / c=4` (80 reqs) | Total tok/s (AL) | **255.8 / 615.9** (3.59 / 3.63) | 221.8 / 542.8 (3.67 / 3.68) | 152.4 / 406.7 (3.67 / 3.58) |

## Architecture Summary

- **48 Logical Layer Visits from 24 Physical Layers** (`H = 2304`, `n_layers_per_attn_res_block = 6`):
  - Prelude: physical `0..5` (1x, visits `0..5`)
  - Core: physical `6..17` (3 passes, visits `6..17`, `18..29`, `30..41`) with Cross-Pass AttnRes reads over the previous pass's 2 core blocks and rank-64 FP32 no-FMA gated loop controllers across `(attn, k, v, mlp)`
  - Coda: physical `18..23` (1x, visits `42..47`)
- **Hybrid Attention & `sconv`**:
  - 40 local SWA visits (`W = 1024`) + 2 MTP SWA layers in per-sequence FP8 ring buffers (`bytes_per_seq`)
  - 8 full relative-bias GQA visits (physical `5, 11, 17, 23`) + 1 MTP full layer in paged FP8 KV cache
  - 4-stream short convolution (`k, v, attn, mlp`, `W = 4`) across 48 visits
- **Native Mixed MoE Quantization** (zero extra weight quantization):
  - Layers `0..19`: NVFP4 (`fouroversix_mae`, block=16 + FP32 outer scales), 128 experts, top-8, `inter = 1152`
  - Layers `20..23`: MXFP8 (`mxfp8_e4m3`, block=32 `ue8m0`)
- **Unembed & DSpark MTP**:
  - `extern:cublas_bf16_tn_f32` -> FP32 `30.0 * tanh(x / 30.0)` + vocab mask `>= 200058 -> -inf`
  - 3-layer BF16 bidirectional DSpark drafter (`block_size = 8`, `target_layer_ids = [5, 23, 41, 48]`, `Inkling2MarkovHead` rank-256 `rnn_hidden`)

## Build & Verify

### Prerequisites

- CUDA 13.x toolchain (`nvcc` supporting `sm_120a`)
- Python 3.12 (`uv sync` from `pyproject.toml` / `uv.lock`)
- [`kern`](https://github.com/pegainfer-project/kern) binary (`KERN_BIN` or on `PATH`)
- Local Inkling-2-Mini weights directory (`KERN_WEIGHTS`, e.g. downloaded from `thinkingmachines/Inkling-2-Mini`)
- Submodule initialized: `git submodule update --init --recursive`

```sh
export KERN_WEIGHTS=/path/to/Inkling-2-Mini
export KERN_BIN=/path/to/kern

# 1. Inspect checkpoint (CPU-only)
uv run python scripts/inspect_checkpoint.py --checkpoint "$KERN_WEIGHTS" --strict

# 2. AOT-compile SM120 CUDA & Triton kernels (CPU-only)
uv run python scripts/build.py --update
uv run python scripts/build_dsl.py

# 3. Generate bundle & static verify
uv run python scripts/gen.py --checkpoint "$KERN_WEIGHTS" --out out/inkling-2-mini
"$KERN_BIN" verify out/inkling-2-mini/model.json

# 4. Run unit tests (GPU)
KERN_RUN_GPU=1 uv run python -m unittest discover -s tests/unit -v

# 5. Serve (1M context, FP8 KV, DSpark k=7, max_seqs=8)
"$KERN_BIN" server --config out/inkling-2-mini/kern.toml inkling_2_mini \
  --port 8000 --chunk 2048 --max-seqs 8 --seq-slots 32
```

## Optimization Loop (`loop/`)

The repository preserves the hill-climbing verification and A/B harness used to achieve the `3.000×` SM120 speedup:

- `loop/env.sh`: path & environment configuration (`KERN`, `KERN_LOOP_WEIGHTS`, `KERN_LOOP_SMOKE_REF`, `KERN_LOOP_STD_REF`)
- `loop/make`: incremental kernel rebuild + `gen.py` + `kern verify`
- `loop/check` & `loop/e2e_smoke`: deterministic span parity & `e2e-logits` KL gate
- `loop/ab` & `loop/bench`: paired A/B CUPTI benchmarking + `loop/guard.toml` regression guard (`loop/score.py`)
- `loop/onmodal` & `loop/modal_runner.py`: optional remote SM120 GPU runner on Modal

## Precision & Memory Budget

See [`precision.toml`](precision.toml):
- Text + MTP weights: `~18.18 GiB`
- Paged FP8 full KV (`9` full layers): `18 KiB/token` (`18.00 GiB` @ `1,048,576` tokens)
- Per-sequence SWA + `sconv` ring states: `~86 MiB/seq` (`< 0.7 GiB` for `max_seqs = 8`)

## License

Apache-2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`upstreams.json`](upstreams.json).
