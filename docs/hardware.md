# Our GPUs, and which tasks run where

**Short answer.** We have one RTX 5090 workstation and six RTX 4090s on two servers that share storage. Every task in this repo runs on the 4090. Fewer run on the 5090: the older vLLM/SGLang fixes pin builds with no code for its architecture. Where a task runs on both, the speedups differ in both directions, so each GPU needs its own reference score.

## What hardware is this?

| | RTX 4090 servers | RTX 5090 workstation |
|---|---|---|
| GPUs | 6 × RTX 4090 (24 GB, sm_89): a 4-GPU server and a 2-GPU server | 1 × RTX 5090 (32 GB, sm_120) |
| Usable GPU memory | about 22 GiB per GPU | about 30 GiB (the desktop holds about 0.8 GiB) |
| GPU links | none: no peer-to-peer between 4090s; multi-GPU traffic goes over PCIe and shared memory | single GPU |
| CPU | AMD Threadripper PRO 5975WX: 32 cores / 64 threads, 4 CCDs, each with its own L3 cache (both servers are the same type) | AMD Ryzen 9 3950X: 16 cores / 32 threads, 2 CCDs |
| Host memory | 251 GB (most of it page cache, so model weights stay cached) | 62 GB |
| NVIDIA driver | 550.127.08 (supports CUDA up to 12.4) | 595.99 (CUDA 13.2) |
| Storage | local NVMe under `/tmp` (about 5 GB/s), wiped on restart; a shared network home directory (270-340 MB/s cold reads, slower under load) | local disk |

**What the 4090 driver means.** CUDA 12.8 and 12.9 wheels run on driver 550 through CUDA's minor-version compatibility. CUDA 13 wheels do not. vLLM moved to CUDA 13 wheels in about mid-2026 (around v0.26), and SGLang's kernel package at 0.4.7, so engine versions newer than that cannot run on our 4090s.

**Keep work on local disk.** Put venvs, model weights, compile caches (`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`) and `TMPDIR` on local NVMe, not on network storage:
- torch.compile costs 28-44 s per cold start on network storage, against 4.7 s from a warm local cache;
- Triton temporary files on network storage cost about 100 ms per request, and moved Apple MM's score from 1.39× to 1.65×;
- keep `TMPDIR` short: vLLM's zmq socket path has a 107-character limit, and a deep path crashed one engine ("File name too long").

## How are CPU cores pinned to GPUs?

Every timed job runs pinned to one CCD, so jobs on different GPUs share no physical core and no L3 cache. Logical CPUs k and k+32 (4090 servers) or k and k+16 (5090) are the two threads of one physical core.

| Machine | GPU | CPUs (`taskset -c`) |
|---|---|---|
| 4-GPU 4090 server | 0 | 0-7,32-39 |
| | 1 | 8-15,40-47 |
| | 2 | 16-23,48-55 |
| | 3 | 24-31,56-63 |
| 2-GPU 4090 server | 0 | 0-7,32-39 |
| | 1 | 24-31,56-63 |
| RTX 5090 workstation | 0 | 0-7,16-23 (builds and staging go on 8-15,24-31 at low priority) |

- `nvidia-smi topo -m` reports CPU affinity 0-63 for every GPU on the 4090 servers, so it cannot be used to choose a CPU set. Use the table.
- An early table that gave the two threads of one core to different GPUs caused interference between GPUs. Don't split a core.
- Don't build, copy or hash on a CCD whose GPU is running timed jobs; it showed up as single bad pairs and stretched jobs (see [findings.md](findings.md)).

## Which tasks run where?

REF is the human fix's (or Apple oracle's) calibrated speedup on that GPU. A/A is the largest deviation from 1.0 when the unmodified code is scored against itself. Minutes are per scoring job.

### Apple tasks and our scaled task

| Task | RTX 4090 | RTX 5090 | Notes |
|---|---|---|---|
| Apple MM (`regression_multimodal_prefill_ttft`) | 1.389 (1.373-1.401); A/A within 2.1%; 20-27 min | 1.549-1.560 (independent tester 1.542); A/A 0.997-1.003; about 18 min | `--attention-backend triton` on both; memory fraction 0.50 (4090) / 0.60 (5090) |
| Apple ST, adapted (`regression_sglang_stream_decode_tps_c1`, Qwen2.5-1.5B) | 1.142-1.146; A/A within 0.3%; 18-20 min | Triton attention: 1.191-1.195; FlashInfer with CUDA 12.9: 1.222-1.224; A/A within 0.8%; 16-20 min | 5090 needs one of two backend changes (below) and memory fraction 0.60 |
| Task C (`optimize_vllm_qwen35_4b_high_concurrency`) | about 1.32 (1.312-1.314 on one server, 1.324-1.330 on the other) | 1.317-1.324; about 21 min | 5090 limited to 70% of GPU memory |
| Φ-Bench vLLM scheduler | authors' answer 1.393; best configuration 1.712 | authors' answer 1.363 (tester 1.344); best configuration 1.609 (tester 1.55); 20-25 min | see [selection/other-benchmarks.md](selection/other-benchmarks.md) |

### Tasks from vLLM and SGLang history

| Task | RTX 4090 REF (band) | RTX 5090 | Why it does or doesn't run on the 5090 |
|---|---|---|---|
| vLLM #7822 | 1.480 (±0.5%) | not run | pins builds of the same age or older than those below |
| vLLM #16436 | 1.0855 (±1.0%) | not run | same |
| vLLM #12380 | 1.507 (±1.1%) | **does not run** | torch 2.5.1+cu124 has no sm_120 code and no PTX: "no kernel image is available" on the first tensor operation |
| vLLM #14930 | 1.0909 (±0.5%) | **does not run** | same (torch 2.6.0+cu124) |
| vLLM #13365 | 1.0934 (±0.5%) | **does not run** | same (torch 2.5.1+cu124) |
| SGLang #6994 | 1.474 (±1.1%) | **does not run** | torch 2.7.1+cu126, and sgl-kernel 0.1.7 has machine code only up to sm_90 |
| vLLM #14857 (+ #14778) | 1.0648 (±0.5%); #14778 1.0365 | **does not run** | torch 2.6.0 / 2.5.1 +cu124 |
| SGLang #16521 | 3.83 (3.779-3.875) | **3.524** (jobs 3.531, 3.491); A/A 0.53%; about 8.5 min | runs as shipped |
| SGLang #20972 | 1.142 (±0.6%) | **1.226** (jobs 1.224-1.235); A/A 0.72%; about 9 min | runs as shipped; the 4090 band is too tight here, use at least ±1% |
| SGLang #9460 | 1.203 (±1.5%) | **does not run** | sgl-kernel 0.3.8 has no sm_120 code (RMSNorm fails during CUDA-graph capture) |
| vLLM #37338 (cold start) | 1.29 (±2%) | **1.176** (jobs 1.178, 1.183); A/A 0.57%; about 36 min | runs as shipped; its re-read pass peaks at 30.1 GB, only about 0.6 GB under our limit |
| SGLang #27868 | 1.0276 (±0.5%) | **1.058** (jobs 1.053-1.061); A/A 0.56%; about 8.4 min | needs a CUDA 12.9 compiler for FlashInfer (below) |
| SGLang #14956 | 1.31 (±2%) | **1.487** (calibration), 1.511 and 1.489 (scoring); A/A job 1.007; about 14 min | runs as shipped |
| vLLM #35442 | 1.037 (±1%) | smoke test passes; scoring still running | memory utilization 0.80 → 0.60 (at 0.80 the card reached 31.1 GB during CUDA-graph capture) |
| vLLM #27418 | 1.69 (±3%) | smoke test passes; scoring still running | memory utilization 0.60 and re-read pass 0.38, as precautions |
| vLLM #30056 | 1.79 (±3%) | smoke test passes; scoring still running | memory utilization 0.60, as a precaution |
| SGLang #20503 | 1.08 (±4%) | not tried | |

Why the speedups differ between the GPUs, and which way, is discussed in [findings.md](findings.md), section 4. In short: a fixed host cost per step weighs more on the faster 5090, and a cold-start task is set by the host CPU, which is slower on the 5090 workstation (every cold run of #37338 is about 50% longer there; it saves about the same 48 s, so the ratio drops from 1.29 to 1.18).

On the 5090, broken variants are still rejected wherever we tried them (#16521, #20972, #37338, #27868, #14956, and Apple ST's two fault patches).

## What does the RTX 5090 need?

**sm_120 support.** The 5090 needs builds made with CUDA 12.8 or newer.
- Builds made with CUDA 12.4 or 12.6 contain no sm_120 code. `torch.cuda.get_arch_list()` stops at sm_90, and even `torch.ones(1024, device="cuda") + 1` fails with "no kernel image is available".
- Swapping in a newer torch is not enough: the engines' own CUDA libraries (vLLM `_C`, `_moe_C`, flash-attention; SGLang's sgl-kernel) also stop at sm_90. Running these tasks would mean rebuilding the old engines, which changes the task.
- A torch architecture check alone does not tell you whether a package runs: #9460 has an sm_120 torch but fails in sgl-kernel, and #27868 fails in FlashInfer's compiler step.
- Libraries with no sm_120 code that our tasks don't use on this card: FlashMLA, FA3 and SGLang's `flash_ops`.
- FlashAttention 2 in older vLLM wheels ships sm_80 PTX, which the driver JIT-compiles to sm_120 (slow the first time: a 192 s CUDA-graph capture in one test). FA3 is sm_90a only.
- On the 5090, SGLang picks the FlashInfer MXFP4 MoE backend for gpt-oss (its Blackwell check covers compute capability 12), and vLLM with an fp8 KV cache picks FlashInfer attention. Both change which code a task exercises.

**CUDA 12.9 for FlashInfer's JIT.** FlashInfer 0.6.11 and 0.6.12 compile some kernels at first use, and for sm_120 they need nvcc 12.9 or newer. With the CUDA 12.8 toolkit that a task image or package ships, they refuse, and the server dies during CUDA-graph capture.
- The error message is misleading: "FlashInfer requires GPUs with sm75 or higher". The real cause ("SM 12.x requires CUDA >= 12.9") is logged one line earlier as a warning.
- Fix A: point `CUDA_HOME` at NVIDIA's CUDA 12.9.1 redistributable (nvcc 12.9.86, cudart, cccl, checksum-verified) plus the library headers FlashInfer's kernels include (cuBLAS, cuRAND, and so on). FlashInfer then compiles its kernels for sm_120. This keeps the default backends and changes only the compiler version.
- Fix B: avoid FlashInfer: `--attention-backend triton --sampling-backend pytorch` (SGLang). This changes two code paths on both arms.
- For Apple ST the two fixes give different scores (1.22 vs 1.19), so a 5090 number must name which was used.

**Memory.** Our watchdog kills a job at about 31,000 MiB (30 GiB usable). Lower the engine memory fraction where a task's default would overshoot: about 0.60 on 32 GB gives roughly the KV pool that 0.80 gives on 24 GB.

## Why should setup run with `TZ=UTC`?

The package setup scripts give uv a dependency cutoff date with `--exclude-newer`, and uv reads a bare date in the **local** time zone. The locks were resolved on machines running in UTC. On a machine east of UTC the cutoff moves earlier by the time-zone offset.
- Our 5090 workstation is at UTC+4. For #14857, `--exclude-newer 2025-03-23` then excluded fastapi 0.115.12, which was uploaded at 22:55 UTC on the cutoff day, and setup failed.
- `export TZ=UTC` before setup avoids it. Packages whose locks are fully hashed are not affected.
- Any machine east of UTC can hit this. The packages were left unchanged so their file manifests stay valid; a future version should pass a full UTC timestamp.

## Other practical notes

- **Hugging Face rate limits.** Anonymous downloads from a shared IP got HTTP 429. The download scripts retry with back-off; setting `HF_TOKEN` avoids it.
- **Long downloads stall** on some networks. Our scripts download each pinned wheel in its own short process and resume large files with a fresh connection; everything is checksum-verified before use.
- **Deleted pins.** Some pinned packages were later removed from PyPI (xgrammar 0.1.15/0.1.16, an `apache-tvm-ffi` pre-release). The affected packages carry a documented override, applied to both arms.
- **Cold first runs.** The first run after a GPU has been idle can be about 1.7× slower. Always do an unscored warm-up.
- **Rare slow runs.** About 3% of runs are 3% or more slower, with the GPU waiting on the host (memory reclaim stalls). Pair runs (ABBA) and use medians.
- **VRAM above the engine's budget.** vLLM's logits and sampler temporaries can take a 24 GB card to 97-98% even with a lowered memory fraction. Limit `--max-num-seqs` if needed.
- **Never kill processes by interpreter or pattern** on a shared host. A job may kill only its own process tree. Several of our tools match their own processes by a tag in the environment and the job's root directory.
