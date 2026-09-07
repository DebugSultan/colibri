# Qwen3.8-Flash-Next (`qwen38` engine) — GPU reconnaissance

Date: 2026-09-07 · Status: **reconnaissance only** — no implementation plan, no
strategy recommendation. Every `file:line` reference below was read directly
from the tree (branch `main`); items derived by arithmetic are marked
*(derived)*.

Target machine for this analysis: Ryzen 9 3900X (12c/24t), 64 GB DDR4,
990 PRO 1 TB, RTX 5070 Ti 16 GB + RTX 5060 Ti 16 GB (Blackwell, sm_120,
requires CUDA toolkit ≥ 12.8).

---

## 1. Scope and method

Question: *why does the `qwen38` engine have no CUDA/GPU path while `qwen36`
does, and what exactly would a GPU path have to cover?*

Method: (a) build-system forensics in `c/Makefile`; (b) full forward-pass
compute map of `c/qwen38_core.h` / `c/qwen38.c`; (c) inventory of the GPU
assets already in the repo (`backend_cuda.cu`, `backend_cuda_dsv4.cu`,
`backend_vulkan.c`, `qwen36_tier.c`, the FP8 kernel family and its oracles);
(d) pinned external references (llama.cpp PR #27742, issue #27763, vLLM
recipe).

---

## 2. TL;DR

1. **The CPU-only status is a deliberate, documented build decision, not an
   omission.** `c/Makefile:1032-1036` says so in a comment and compiles the
   target with the NOCUDA flag filters, no tier file, no CUDA object:

   ```make
   # Qwen3.8-Flash-Next text-only sibling. This target is intentionally CPU-only;
   # qwen38.c owns the direct checkpoint loader and SERVE=1 protocol, with no
   # CUDA/Metal tier advertised by the control plane.
   qwen38$(EXE): qwen38.c cli_args.h qwen38_core.h ...
   	$(CC) $(NOCUDA_CFLAGS) qwen38.c -o qwen38$(EXE) $(NOCUDA_LDFLAGS)
   ```

   `NOCUDA_CFLAGS = $(filter-out -DCOLI_CUDA -DCOLI_ANS,$(CFLAGS))`
   (Makefile:1097). Contrast the `qwen36` target (Makefile:1020-1030), which
   under `CUDA=1` compiles `qwen36_tier.c` and links `$(CUDA_OBJ)`.
   `docs/qwen38.md:69` states the same: *"There is currently no Qwen3.8 GPU
   backend."*

2. **The kernel for qwen38's exact expert format already exists in the repo,
   is the default decode path of the fmt=8 grouped family, and is
   oracle-tested.** `grouped_hidden_f8w_dual` / `grouped_down_f8w`
   (`backend_cuda.cu:932,968`; `COLI_CUDA_F8_WARP`, default on on CUDA; `=2`
   is a `cuda_fp8.h` HW variant, CUDA-only). Decode E4M3 through a 256-entry
   LUT, scale per 128×128 block, and accumulate with the *same* convention as
   the CPU reference `matmul_fp8` (`quant.h:523`): f32 partial inside the
   block, scale applied once per block, double across blocks. Oracles:
   `tests/test_fp8_warp_cuda.cu` (old-vs-new, tails, misalignment) and
   `tests/test_fp8_cuda.cu`; bench: `tests/bench_fp8_cuda.cu` +
   `tools/run_f8_bench.sh` (Makefile:905, 992).

3. **The main colibri engine already runs native-FP8 MoE experts through the
   CUDA *and* Vulkan expert-group tier** (`colibri.c:5786-6198`:
   `coli_cuda_expert_group_issue/take/pinned/resident_issue`;
   `backend_vulkan.c:807-838,1148-1169`: `coli_vk_expert_group_issue/take/
   issue2/take2`, used at `colibri.c:5025,5933-5989`). fmt=8 is the main
   engine's "native FP8-e4m3 passthrough" tensor format (loader-seam tests:
   `tests/test_fp8_load.c`, `tests/test_fp8_passthrough.c`,
   `tests/test_fp8_e2e_loader.c`). So "expert matmul on GPU for qwen38's
   weights" is not new technology here — it is production technology of the
   sibling engine.

4. **What qwen38 lacks is not the FP8 GEMM.** It lacks: (a) a tier file that
   wires its per-layer, disk-backed LRU expert flow to an expert-group tier
   (the `qwen36_tier.c` pattern: resident set + async issue/take + CPU
   fallback on misses); (b) GPU or hybrid treatment of the sequence mixers —
   QSA (sparse attention with indexer) and Gated DeltaNet (linear attention
   with a per-layer recurrent state) — and of PLE (a 51 B hashed n-gram table
   that lives on disk) and the 4-branch gated residuals; (c) the build
   switch. These mixers have no kernel in `backend_cuda.cu` today (its
   attention family is MLA-style "absorb", gated to fmt 0-4 and to
   K≤512/T≤4096, `backend_cuda.cu:2198-2228`); the DeepSeek-V4 dedicated
   backend *does* have sparse-attention + indexer + hyper-connection kernels
   (`backend_cuda_dsv4.h`), which is the closest in-repo precedent. The
   architecture is also young upstream: llama.cpp merged it 2026-08-27
   (PR #27742) and still carries an open Blackwell correctness bug
   (#27763, §7.2).

5. **The dominant cost is disk, not FLOPs.** On the reference i9-14900K box,
   of a 140 s median request, 96 s is synchronous expert disk service, 17 s
   expert matmul, 14 s attention, 2.6 s LM head (`docs/qwen38.md`). Decode is
   ~1.1 tok/s at cap 32. A GPU path changes the arithmetic only if the
   resident expert set fits in VRAM; the fit math for the target machine is
   in §8.

---

## 3. Why qwen38 is CPU-only: the evidence

### 3.1 Build system

| item | location | fact |
|---|---|---|
| qwen38 target | `c/Makefile:1032-1036` | intentional CPU-only comment; `$(NOCUDA_CFLAGS) qwen38.c … $(NOCUDA_LDFLAGS)`; no tier source, no `$(CUDA_OBJ)` |
| NOCUDA filters | `c/Makefile:1097-1098` | strip `-DCOLI_CUDA -DCOLI_ANS` from CFLAGS; strip `-lcudart -lstdc++ -lcuda -L$(CUDA_HOME)/lib64…` from LDFLAGS |
| qwen36 target | `c/Makefile:1020-1030` | `CUDA=1` → `QWEN36_TIER_SRC=qwen36_tier.c`, full `$(CFLAGS)/$(LDFLAGS)`, links `$(CUDA_OBJ)`; otherwise NOCUDA (comment 1015-1019 explains the switch) |
| `CUDA_OBJ` | `c/Makefile:521` (and 576) | `backend_cuda.o` |
| `DSV4_CUDA_OBJ` | `c/Makefile:549` | `backend_cuda_dsv4.o` |
| `COLI_V4_SUPPORTED` | `c/Makefile:449-459` | x86-64 (Win/Linux) or aarch64 (Linux/macOS) |
| `CUDA_ARCH` | `c/Makefile:250-278` | `native` default; `portable` = sm_80/86/89/90/**120** SASS + compute_120 PTX (+ sm_121 if nvcc ≥ 12.9). sm_120 ⇒ CUDA ≥ 12.8 |
| DeepGEMM block | `c/Makefile:224-231, 539-560, 797-837` | opt-in `DEEPGEMM=1`, **sm_120a (consumer Blackwell)** only, pinned commit `39fb4447a062b418fd08ce17cd308adb28559417`, fetched by `c/tools/fetch_deepgemm.sh` into gitignored `c/third_party/deepgemm`, `patches-deepgemm-sm120-msvc.patch`, C++20, `cuda-dsv4-dg-dll` builds the sm_120a DLL with the TMA driver API (`-lcuda`) |
| qwen38 segment object | `c/Makefile:1460-1465` | `qwen38.o` compiled with `-DQWEN38_NO_MAIN` for the segment runtime; qwen38 is in the `install` target |
| qwen38 test targets | `c/Makefile:1038+` | `qwen38-tiny-generate`, `qwen38-tiny-check`, `qwen38-ple-prefetch-check` (A/B `Q38_PLE_PREFETCH=0/1` must emit identical tokens), `qwen38-vision-check`, `qwen38-vision-serve-check` |
| fmt=8 test/bench targets | `c/Makefile:905, 992` | decode sweep + census + tail; old-vs-warp kernel bench |
| flake.nix | — | **no `cuda` match**: the toolkit is a host dependency, not pinned in the flake |

There is no `qwen38_tier.c`/`.h` in the tree, and zero occurrences of
`cuda`/`gpu` in `qwen38.c`, `qwen38_core.h`, `qwen38_nfc.h`,
`qwen38_vision.h`.

### 3.2 Design-level differences vs qwen36

The qwen36 tier pattern is: experts resident in **RAM** (int4 + scales), hot
experts *promoted* to VRAM, misses served from RAM. qwen38 inverts the
hierarchy: experts live on **disk** (official E4M3 shards), a per-layer LRU of
a handful of slots holds *native FP8* slabs in RAM, and the scales live in a
small resident bank. The tier problem is therefore "resident VRAM set +
streaming misses", not "promotion from RAM". The sequence mixers are also
different: qwen36 has plain (grouped) attention that the main engine's CUDA
path already covers; qwen38 has QSA (sparse, index-driven, third cache) and
Gated DeltaNet (recurrent state), which have no kernel in the generic
backend.

---

## 4. Model and checkpoint fact sheet

From `docs/qwen38.md` (authoritative) and `qwen38_core.h`:

- Upstream: `Qwen/Qwen3.8-Flash-Next-FP8`, pinned revision
  `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`, ~185.5 GB decimal (~173 GiB).
  colibri loads it **directly from the official safetensors shards** — no
  conversion, no second copy. Text-only by design (vision tower implemented,
  MTP not used).
- Architecture: 125 B ordinary parameters, **6 B activated per token**, plus
  a **51 B hashed n-gram embedding (PLE)**. **48 layers = 12 × {3 Gated
  DeltaNet + 1 QSA}**. Every layer: 4-branch gated residual ("hyper
  connection") + 512-expert top-10 MoE + 1 gated shared expert.
- Geometry (derived): per-expert matrices are 3 × (hidden × inter) bytes in
  FP8 (1 byte/weight). The docs triple-confirm **4.7 MiB per expert**:
  routed-expert cache table (cap 16 ⇒ 3.5 GiB over 48 layers), disk total
  (120.8 GB / 24,576 experts), and decode arithmetic (10 × 48 experts ⇒
  2.2 GiB cold). Hidden = **2560** (the vision tower "projecting to 2560"),
  hence inter = 1,638,400 / 2560 = **640** *(derived; consistent with all
  three doc figures to <0.1%)*.
- QSA: 24 query heads, **2 KV heads**, rotary; the indexer pools **complete
  4-token blocks** (ratio 4), scores them, keeps the **best 512 blocks**
  (budget 2048 tokens) plus a causal tail of up to 3 tokens; 24-head
  attention runs only on those original tokens with a per-head sigmoid gate
  from the packed q.
- PLE: per token, bigram+trigram history is hashed, **16 rows × 160 bytes
  (FP8)** are read from the 51 B-parameter sharded tables, scaled by the
  checkpoint's scalar PLE scale; never materialized in RAM.
- Native context: 262,144 tokens; `Q38_MAXT` default 8,192 (context state
  54 KiB/token, allocated for the whole ceiling before `READY`).
- Correctness reference: tiny oracle generated from
  `Qwen4ExpForCausalLM` in `transformers==5.16.1`; the gate checks greedy
  token IDs **and** the final upstream logit vector, in native-BF16 and
  expanded-FP32 modes, at cache caps 1 and 4; CI repeats cap 1 under
  ASan/UBSan.

### 4.1 Memory accounting (docs table, reference checkpoint)

| item | size |
|---|---|
| resident dense set (native BF16: DeltaNet/QSA projections, norms, gated-residual mixers, embedding, shared experts, LM head) | **9.2 GiB, fixed** |
| routed-expert cache | **4.7 MiB per slot per layer** × 48: cap 16 = 3.5 GiB, cap 32 = 7.0 GiB, cap 64 = 14.1 GiB |
| FP8 scale bank (every expert's 128×128 block scales resident, so a miss is one FP8 read) | 28 MiB, fixed |
| context state (QSA K/V/index) | 54 KiB/token; 432 MiB at 8,192 |
| recurrent + PLE state, prefix snapshot, cached logits | 226 MiB, fixed |
| prompt/decode workspace | ≤ 1.1 GiB peak |
| PLE table (51 B) | 0 RAM — 16 × 160 B row reads per token |
| routed experts on disk | 120.8 GB, streamed |

Launcher default: **one expert slot per layer** (cap 1). `coli plan
--model <dir> --ram <GB> --ctx 8192 --gpu none` prints this accounting and
chooses the cap — note the planner already has a `--gpu` dimension.

### 4.2 Measured CPU performance (docs; i9-14900K 24c / 61 GiB / 990 EVO, cap 32)

| phase | prompt | request s | TTFT s | decode tok/s | hit |
|---|---:|---:|---:|---:|---:|
| cold | 31 | 129.5 | 20.6 | 1.17 | 58.8% |
| warm-identical (exact prefix reuse) | 31 | 107.7 | 0.01 | 1.18 | 63.8% |
| rotating prompts (median of 4) | 35-40 | 140.2 | 23.8 | 1.09 | 53.8% |

Breakdown of the 140 s median: **96 s synchronous expert disk service,
17 s expert matmul, 14 s attention, 2.6 s LM head** → about two thirds of
every request is waiting on disk. Per decode token: 10 of 512 experts × 48
layers × 4.7 MiB = 2.2 GiB of expert weights uncached (~half at cap-32 hit
rate).

---

## 5. Engine compute map (`c/qwen38_core.h` unless noted)

Line numbers verified in this pass. The forward is `q38_step` (1955-1991):
PLE prefetch first, then per token: embedding (or vision row) broadcast to
the 4 residual branches; per layer: PLE injection at `ple_layer`, `attn_gr`
read → QSA or DeltaNet → apply; `mlp_gr` read → MoE → apply; finally
`final_gr` read → LM head. `q38_layers_forward_range` (1932-1953) is the
segment-adapter slice of the same loop.

### 5.1 Core structures

- `Cfg` 19-34: hidden, layers, vocab, max_positions, eos, eps, theta,
  hc_count/hc_rank/hc_width, q_heads/kv_heads/head_dim/rotary_dim,
  idx_qheads/idx_kheads/idx_dim/idx_budget/idx_ratio, experts/topk/inter/
  shared_inter/norm_topk, dn_kheads/dn_vheads/dn_kdim/dn_vdim/dn_convk/
  dn_conv_dim, ple_layer/ple_dim/ple_convk/ngram_size/heads_per_ngram,
  ngram_heads/ngram_head_dim/ngram_parts, is_attn[512], vision fields.
- `Q38Weight` 43-50: `data`, `scales` (block-FP8 only), rows/cols,
  `scale_count`, `kind ∈ {F32, BF16, FP8}`, ownership flags.
- `Layer` 78-92: attn_gr/mlp_gr (GatedResidual), router, shared expert
  (sh_g/sh_u/sh_d + sh_gate), q/k/v/o, qn/kn, idx_qk + idx norms, DeltaNet
  projections + conv + A_log/dt_bias + norm, PLE key/value + norms + conv.
- `Slot` 94-100 (per-expert LRU entry: eid, gate/up/down, `fp8_slab`),
  `LCache` 101 (slots, by_expert, cap), `Q38ExpertScaleCache` 103-107
  (resident `[expert][3][scale_count]` f32 bank; `ready` 0/1/-1).
- Timers (60-71, `COLI_TIMERS=1`): expert-read, fp8-expand, routed-expert,
  shared-expert, dense-matmul (resident-mm), deltanet, qsa-index,
  qsa-attention, ple, lm-head.

### 5.2 Matmul core

- `q38_matmul` 189-200 (f32, OMP over output rows); `q38_matmul_bf16`
  248-259 (BF16 storage, FP32 math — documented: storage parity with
  `st_read_f32`, no BF16 dot); `q38_weight_matmul` 261-275 (dispatch on
  `kind`); `q38_dense_matmul` 290-295 (+timer).
- `matmul_fp8` (`quant.h:523-543`) — the CPU reference for native experts:
  scalar, OMP over `o`, per 128×128 block: `float acc` inside the block,
  `a += (double)acc * sc` across blocks; scale index `scl = bscale +
  blkO*nblkI; sc = scl[bi]`. Its own comment (quant.h:517-522) is worth
  quoting: *"Scalar reference path (no SIMD in v1 -- BW-bound like the Metal
  kernel, and this format's hot path is the GPU one; a vectorized CPU kernel
  is future work if measured needed."* — the code declares the GPU path as
  the intended hot path for exactly this format.
- `E4M3_LUT` (`quant.h:478-511`): static 256-entry decode table (compile-
  time, deliberately — it runs under `#pragma omp parallel for` and a lazy
  init would race), byte-for-byte cross-checked against
  `torch.float8_e4m3fn`; NaN policy: NaN decodes to IEEE NaN and propagates;
  the sampler has a tested safety net (`tests/test_logit_nan.c`).
  `FP8_BLOCK 128`, `fp8_nblk` at quant.h:514-515.

### 5.3 Gated residuals ("hyper connections", 4 branches)

- `q38_gr_read` 890-914: per branch b: `norm_b = rms0(hyper_b)` (zero-
  centered RMSNorm, `1+w`, `q38_rms0` 305-309); `low = silu(norm_b @ down)`
  (W→R), `mix = low @ up` (R→W); `mixed = Σ_b sigmoid(mix_b)·norm_b / C`;
  `inject = 2·sigmoid(norm_b @ inject)`.
- `q38_gr_apply` 916-922: `hyper_b += inject_b · block`.
- Two read+apply pairs per layer (around the mixer and around the MoE) plus
  `final_gr` before the LM head. Per layer: 2 low-rank down + 2 up + 4
  inject projections and 4 zero-centered RMSNorms.

### 5.4 QSA (`q38_attention`, 1599-1648)

- Projections: q packed with its gate (H → QH×2D), k/v (H → KVH×D),
  indexer q/k (H → (IQ+idx_kheads)×ID).
- Cache update (1606-1614, per token, sequential): per KV head
  `k = rms0(k, kn)` + RoPE(pos) → `m->K[layer][h][pos]`; v raw → `m->V`;
  indexer key → `m->IK[layer][pos]`. Caches: `K,V [KVH][kv_cap][D]` f32 +
  `IK [kv_cap][ID]` f32 per QSA layer, `kv_cap = Q38_MAXT`
  (`ensure_kv` 1919-1927) — this is the 54 KiB/token.
- Selection (1618-1633, per token): pool each complete 4-token block
  (mean of indexer keys → rms0(idx_kn) + RoPE(block start)); score =
  Σ_heads max(0, qidx_h·pool)/√ID; sort; take `min(blocks, budget/4)`
  blocks; selected tokens = take×4 + causal tail.
- Attention (1634-1644, per query head, sequential over heads):
  `qh = rms0(qraw[0:D])` + RoPE; scores over selected tokens; softmax;
  `out_h = Σ a_j V[kh][selected_j]`; **`out_h *= sigmoid(qraw[D:2D])`**
  (the packed gate). No OMP inside — each s reads the cache rows written by
  earlier s in the same call, so the token loop is inherently sequential.
- Output projection: heads @ o (QH×D → H), batched over the window.

### 5.5 Gated DeltaNet (`q38_deltanet`, 1489-1591)

- Batched projections over the window: dn_qkv (H→CD), dn_z (H→V), dn_b, dn_a
  (H→VH); window rows capped by `Q38_PREFILL_BATCH_ROWS` (32) and a 64 MiB
  workspace.
- Per token (sequential): depthwise conv over the qkv ring (CK taps,
  `DN_conv[layer]`) + silu; split q/k/v; per vhead: row-normalize q and k,
  then **OMP over vheads**: `alpha = exp(-exp(A_log)·softplus(a+dt_bias))`,
  `beta = sigmoid(b)`; state update `state *= alpha; delta =
  (v − k·state[:,v])·beta; state += k⊗delta` on a float KD×VD matrix
  (`DN_rec[layer]`); `core[v] = q·state[:,v]`; per-head `q38_rmsg`
  (RMSNormGated with **sigmoid** gate from z — 312+; the sigmoid matches the
  upstream note that this GDN variant gates with sigmoid, not silu); batched
  output projection (V→H).
- Comment 1485-1488: *"The convolution and recurrent update remain strictly
  token-causal inside each chunk, so chunk boundaries cannot change state or
  floating-point order."* — the chunking is provably numerically safe.

### 5.6 MoE and the expert-streaming subsystem (the core)

**Routing** (both paths): dense router H→E, softmax, top-K by repeated
argmax + `rt_router_pick` deterministic fallback, optional top-k gate
renormalization (`norm_topk`), `rt_route` trace.

**Decode** (`q38_moe_decode`, 1653-1686; comment 1650-1652: *"The single-row
path is intentionally kept separate from prefill"*): prefetch (1667) → gated
shared expert (1671, computed while experts load) → `q38_expert_get_batch`
(1674) or per-expert `q38_expert_get` (1676) → **the per-expert loop
(1675-1680)**:

```c
for(int z=0;z<K;){
    Slot *ex=loaded_batch?selected[z]:q38_expert_get(m,layer,idx[z]);
    q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
    for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
    for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
}
```

This is **the** GPU hook point: `xs` (1×H), the expert's `Slot` (three
`Q38Weight` of kind FP8 = raw E4M3 slab pointers + scale pointers into the
resident bank), `route_gates[z]`, residual `ys`. The data in scope is
exactly what `grouped_hidden_f8w_dual`/`grouped_down_f8w` consume (gate+up+
down descriptors, per-expert 128×128 scales, f32 x).

**Prefill** (`q38_moe_prefill`, 1698-1903): batched router (≤32 rows per
window; `q38_moe_prefill_rows` 1698-1712 sizes the workspace), per-row
top-K, **expert-major regrouping** (1799-1811: group offsets, assignments
reordered by expert, inverse map), prefetch of unique experts (1814),
batched shared expert (1818-1833), then expert groups **sized to the LRU
cap** (`load_limit = min(cap, Q38_MAX_TOPK)`, "cache-sized parallel load
groups", 1845-1878): per group `q38_expert_get_batch` loads the demand set
in parallel, per expert the input rows are gathered and batched gate/up
(+silu·mul) and down run over that expert's row span into a scratch, and the
reduction (1883-1894) replays the original per-row top-k order — documented
**bit-identical** to the single-row path across F32/BF16/FP8. Dispatch
`q38_moe` 1905-1908: `S≤1 || !prefill_batch` → decode, else prefill.

**Expert storage/loading:**

- Per-layer LRU `q38_expert_get` (1241-1251, O(cap) scan); miss →
  `q38_load_expert` (1229-1239) → first `q38_try_load_native_fp8_expert`
  (1122-1131), else the generic per-matrix loader.
- `q38_native_fp8_expert_tensors` (1070-1090): gate and up must be
  **contiguous in the shard** (`weight[1]->off == weight[0]->off + nbytes`)
  and dtype==4 with exact shapes — enables a 2-pread pair (gate+up) + down.
- `q38_prepare_expert_scale_bank` (967-1035): once per layer, all gate/up
  `weight_scale_inv` tensors are one compact disk range and all down another
  (invariant verified by fd/offset/byte checks); both ranges are read with 2
  preads and scattered into the resident f32 bank
  `values[(expert*3+proj) · scale_count]` **by numeric expert id** (the
  checkpoint order is lexical: 0, 1, 10, …). `scale_count =
  fp8_nblk(inter)·fp8_nblk(hidden)`. Incompatible layouts fall back
  (`ready=-1`) to the per-matrix loader. This is why a miss is one FP8 read:
  the scales are already resident (28 MiB total).
- `q38_bind_fp8_slot` (1046-1068): slab = 3·hidden·inter bytes; gate at
  `raw`, up at `raw+mb`, down at `raw+2mb`; scales borrowed from the bank
  (`scales + k·scale_count`). `q38_bind_borrowed_fp8` (1037-1044) sets the
  FP8 `Q38Weight` on the borrowed pointers.
- **`COLI_MAP_EXPERTS=1`** (opt-in in `st.h:995-1031`, per-shard mmap cache,
  issue #1325): if the three shard ranges are mappable, the slot **points at
  the mmap** instead of copying the slab — `q38_load_native_fp8_ranges`
  1092-1120, comment 1097-1098: *"Se i tre intervalli sono mappati
  (COLI_MAP_EXPERTS=1), lo slot li PUNTA invece di copiarli: niente slab,
  niente 14 MB per miss."* (Note: 14 MB ≈ 3× the 4.7 MiB per-expert figure
  the docs triple-confirm — see §10.)
- `q38_prefetch_native_fp8_experts` (1133-1148): `posix_fadvise(...,
  POSIX_FADV_WILLNEED)` on the two ranges of each missing expert, called
  before the expert loop (decode 1667, prefill 1814); counter
  `expert_prefetch_ranges`.
- `q38_expert_get_batch` (1264-1342): parallel demand-set load
  (`Q38ExpertLoadJob` 1253+). Preconditions: `expert_parallel_reads`, count
  ≥ 2 and ≤ `Q38_MAX_TOPK`, cap ≥ count, scale bank ready, native FP8
  tensors available, no duplicates. The main thread reserves the slots
  (protecting them from concurrent eviction), OMP workers fill in parallel
  (one `q38_load_native_fp8_ranges` per job), the main thread publishes the
  `by_expert` indices. Any unmet precondition → 0 → serial fallback.
- Per-matrix fallback `q38_load_fp8_expert_weight` (1150-1185+): native FP8
  → reserve FP8 + raw read + f32 scale read; non-native → decode to F32
  (`e4m3_decode · scale`, OMP) for A/B (`Q38_NATIVE_FP8=0`).

### 5.7 PLE (51 B hashed n-gram embedding)

- `q38_ple_row` (1344-1356): binary search over ≤512 PLE shards
  (`ple_part_start`), then one raw pread of the 160-byte FP8 row (dtype==4)
  with `e4m3_decode · ple_weight_scale`, or an f32 slice read.
- `q38_hash_row` (1358-1364): `x = cur·mult[0] ^ p1·mult[1] (^ p2·mult[2]
  for trigrams)` in u64 (multipliers ~2.4e13 — 64-bit, cf. llama.cpp's
  UINT64 GGUF fix), then `sx % ple_head_vocab[head]` (signed-adjusted) +
  `ple_head_offset[head]`.
- `q38_ple` (1429-1468): per token: history p1/p2; embedding rows (prefetch
  buffer or inline); `keys = emb@ple_keyᵀ`, `value = emb@ple_valueᵀ`; per
  branch b (4): `kn = rms0(keys_b)`, `qn = rms0(hyper_b)`,
  `dot = kn·qn/√H`, `shaped = copysign(√|dot|, dot)`, `g = sigmoid(shaped)`,
  `gated = g·value`, `norm = rms0(gated)`; output = gated + depthwise conv
  (ple_convk taps) over a per-head ring state; ring shift
  (`state_len = (ple_convk−1)·ngram_size`); history update, reset on EOS.
- `q38_ple_prefetch` (1388-1427): at forward start (before any layer),
  simulates the two-token history window (without mutating it) and reads all
  16 rows × S tokens in parallel (OMP). Rationale in the comment: PLE sits
  at layer 2 of 48, giving ~2 full layers — the expensive expert-streaming
  part — of lead time; *"16 righe da 160 byte"*; a pure reordering, enforced
  by `qwen38-ple-prefetch-check`.

### 5.8 Environment variables (definitive inventory)

Strict-bool via `q38_env_bool` (core.h:578, accepts only "0"/"1"):

| var | default | line | effect |
|---|---|---|---|
| `Q38_NATIVE_FP8` | 1 | core.h:817 / qwen38.c:2388 | experts stay E4M3 in cache (else F32-expanded) |
| `Q38_NATIVE_BF16` | 1 | core.h:818 / qwen38.c:2387 | resident matrices stay BF16 (else F32) |
| `Q38_EXPERT_PREFETCH` | 1 | core.h:819 | `posix_fadvise` on upcoming expert ranges |
| `Q38_EXPERT_PARALLEL_READS` | 1 | core.h:820 | `q38_expert_get_batch` parallel demand loads |
| `Q38_PREFILL_BATCH` | 1 | core.h:821 | expert-major batched prompt path |
| `Q38_VISION` | 1 | core.h:877 | image path |
| `Q38_PLE_PREFETCH` | 1 | core.h:1392 | early parallel PLE row reads |

Free-form: `Q38_MAXT` (qwen38.c:50, default 8192), `Q38_EOS` (1241),
`Q38_PREFIX_LOG` (1518/1520), `SNAP` (1628), `OPENAI` (1630), `MODEL`
(1631), `SERVE` (1641/1743), `TOK` (1659), `N_NEW` (1722), `ENC_DEBUG`
(1731), `DUMP` (1736/1802), `PPL` (1753), `NOSTREAM` (1782), `COLI_TIMERS`
(core.h:1994), `COLI_USAGE` (qwen38.c:933), `COLI_MAP_EXPERTS` (st.h:1007),
plus `OMP_NUM_THREADS` and the `coli plan` budget flags.

Constants (core.h:12-17): `Q38_MAX_LAYERS 512`, `Q38_MAX_EXPERTS 1024`,
`Q38_MAX_TOPK 256`, `Q38_MAX_PLE_PARTS 512`, `Q38_PREFILL_BATCH_ROWS 32`,
`Q38_PREFILL_WORKSPACE_BYTES (64u<<20)`.

---

## 6. GPU asset inventory (what already exists and could be reused)

### 6.1 Generic CUDA backend ABI (`c/backend_cuda.h`)

One exported surface, `coli_cuda_*` (up to `COLI_CUDA_MAX_DEVICES 16`, line 22),
shared by every engine that opts into `COLI_CUDA`:

- Init/device: `coli_cuda_init(devices, count)` 57; shutdown 58;
  `device_count`/`device_at`/`mem_info`/`device_integrated`/`stats` 61-68;
  `group_stats` + per-device variant 68-71.
- **Format guards**: `coli_cuda_e8_set_grid` 78 (fmt=6, must precede any
  fmt=6 upload); `coli_cuda_fp8_set_lut(const float *lut)` 83 — *"Must be
  called after coli_cuda_init; fmt=8 uploads are refused until it"* — i.e. the
  backend already has a first-class fmt=8 (E4M3) ingestion path gated on a
  host-provided decode LUT.
- Tensor upload: `tensor_upload_g`/`tensor_upload`/`tensor_upload_compressed`
  86-96; `tensor_free`/`bytes`/`device`/`update` 188-193.
- Matmul: `matmul_mxfp4` 113, `matmul` 118, `expert_mlp` 126,
  `shared_mlp_w4a16` 133 (TC int4, *"does not quantize the activation"*).
- **Expert groups** (the MoE hook): `expert_group_issue` 143 + `take` 147
  (async pair), `expert_group` 149 (sync), `expert_group_pinned` 157, and
  `expert_group_resident_issue`/`_take(home_device, devices, …)` 216-220
  (multi-device resident variant).
- **Attention**: `attention_absorb` 164, `attention_absorb_batch` 171,
  `attention_project_batch` 178, `attention_project_ragged` 183, plus device
  variants 228-242 (`absorb_batch_dev`, `absorb_kvdev`,
  `project_batch_dev`, `project_batch_dev_out`).
- **Pipe ops** (elementwise/norm plumbing over device buffers): `pipe_scratch`
  197, `pipe_alloc/free` 198-199, `pipe_upload/download` 200-201, `pipe_rmsnorm`
  202, `pipe_rope` 204, `pipe_silu_mul` 206, `pipe_add` 207, `pipe_rows_add`
  208, `pipe_gemm` 210, `pipe_rmsnorm_s` 211, `pipe_rope_base` 214,
  `pipe_router` 222, `pipe_copy2d` 226, `pipe_peer_copy` 237, `pipe_sync` 242.

### 6.2 Kernels in `c/backend_cuda.cu`

Grouped-expert kernels (the direct counterparts of the qwen38 expert loop):

| kernel | line | input format | notes |
|---|---|---|---|
| `grouped_hidden_g4_dual` | 851 | fmt=4 packed int4 | gate+up in one launch (`dual`), `GroupDesc` per expert |
| `grouped_hidden_f8_dual` | 892 | fmt=8 E4M3 + 128×128 scales | baseline FP8 grouped |
| `grouped_down_f8` | 911 | fmt=8 | down projection |
| `grouped_hidden_f8w_dual` | 933 | fmt=8, warp rework | `COLI_CUDA_F8_WARP` (env read at 1656; CUDA default on, HIP default 0 per `tests/test_fp8_warp_cuda.cu:135-139`) |
| `grouped_down_f8w` | 968 | fmt=8, warp rework | template `<0>`/`<1>` variants benched side-by-side |

Dispatch inside the backend: `1825-1836` picks `f8w<1>/<0>` when enabled and
falls back to `grouped_hidden_f8_dual`/`grouped_down_f8`; `1986` and `2141`
launch `grouped_hidden_g4_dual` on the sync and async (`issue`) paths.
Tensor-core gates: `COLI_CUDA_TC_INT4` (1905, `TC_MIN_ROWS` default 8 at
1911), `COLI_CUDA_TC_W4A16` (1932-1945, requires `COLI_GPU_HAS_WMMA`,
compute ≥ 7, min rows default 16), `COLI_CUDA_W4_PACKED` (1972, 2117),
`COLI_CUDA_DUAL_PROJ` (1974, 2121, default on). Async default on via
`COLI_CUDA_ASYNC` (1887); profiling via `COLI_CUDA_PROFILE` (1894).
`expert_group_issue` implemented at 2061, `take` at 2175.

Other backend env vars found: `COLI_ANS_PROFILE` 1152, `CUDA_RAW_EXPERTS`
1262, `COLI_ANS_SIDECAR`/`COLI_ANS_PACK`/`COLI_ANS_DIRECT` 1460-1466,
`COLI_GPU_FAIL_AFTER` 1640.

The grid shape used by the FP8 grouped kernels (from
`tests/bench_fp8_cuda.cu:69-85`) is `dim3(I or I/8, S, E)` — i.e. the kernel
is already parametrized for **E experts × S rows**, so a decode call
(S=1, E=K≤topk) is a legal configuration of the existing kernel.

### 6.3 Oracles and benches (tests/)

- `test_fp8_cuda.cu:196-197` — async `issue`/`take` checked against the CPU
  reference (this is the exact call pair a qwen38 decode path would use).
- `test_fp8_warp_cuda.cu` — warp rework: decode sweep, census + tail, and an
  old-vs-new rerun with `COLI_CUDA_F8_WARP=0` vs `=1` (254-256); unaligned
  rows at 367-369.
- `test_grouped_g4_cuda.cu:144-145` — fmt=4 grouped path, sync + async.
- `bench_fp8_cuda.cu:69-85` — all five grouped kernels launched per decode
  shape (Makefile target `fp8-bench`, 995-999; comment at 905 and 992).
- `test_cuda_fmt_guard.c` — tensors in wrong formats must be refused by the
  absorb/grouped kernels.
- `test_weights_owned_cuda.cu`, `test_absorb_determinism.cu` — ownership and
  determinism of the absorb path.

### 6.4 Vulkan sibling (`c/backend_vulkan.h/.c`)

Same conceptual surface: `coli_vk_expert_group_issue`/`take` 63/66,
`issue2`/`take2` 83/86 (implemented 807/815, 1148/1154). The main engine
already issues expert groups to Vulkan (colibri.c 5025, 5933-5935, 5970,
5989) and can run the MLA absorb attention core on Vulkan
(`COLI_VK_ATTN=1`, colibri.c 633).

### 6.5 Main-engine multi-GPU precedent (colibri.c)

The GLM engine shards MoE groups across several CUDA devices and takes each
device's result back: issue per device at 5786 (`pd_g[di]`,
`g_cuda_devices[di]`) with takes at 5808 and 6089; a second pattern at
6147-6164 (`dev_g[di]`). So "N devices, issue in parallel, take and reduce"
is an existing, exercised pattern in this repo — not hypothetical.

---

## 7. The working pattern: qwen36's VRAM expert tier

- `qwen36_tier.h:16-18`: *"Enable with `COLI_CUDA=1`
  `[COLI_GPUS=0,1]` `[CUDA_EXPERT_GB=<G>|auto]` … -DCOLI_CUDA (CUDA=1);
  otherwise the inline stubs below keep the engine CPU-only"* (stubs at
  68, `#ifdef` at 24).
- `qwen36_tier.c:446/461`: the tier's MoE hot path is exactly
  `coli_cuda_expert_group_issue(tg[di], tu[di], td[di], rows, c, xr)` →
  `coli_cuda_expert_group_take(G.dev[di])` — issue the K-expert group to the
  device, take the reduced output, no per-expert round trips.
- Makefile 1010-1030 documents the wiring: `CUDA=1` compiles
  `qwen36_tier.c` and reuses the shared CUDA backend (`$(CUDA_OBJ)`);
  without it the engine is CPU-only via the tier header's stubs. The comment
  (1015-1019) explicitly justifies keeping `NOCUDA_*` flags off by default so
  `make qwen36` has no toolkit dependency.
- Testability without a GPU: `tests/qwen36_fake_cuda.h` fakes
  `coli_cuda_init`/`expert_group_issue`/`take` (72-86), and
  `test_qwen36_tier_shutdown.c` / `test_qwen36_tier_int8.c` drive the tier
  purely via `setenv("COLI_CUDA","1")` + `setenv("COLI_GPUS","0")`.
  `tests/test_efficiency_report.py:87-88` shows the runtime knob set in use:
  `COLI_CUDA=1, COLI_GPU=0, CUDA_DENSE=1, CUDA_EXPERT_GB=<vram_gb>`.
- `CUDA_DENSE` additionally moves the dense (non-expert) weights to VRAM
  (referenced by the same efficiency-report overlay and by
  `GPU_BACKENDS.md`).

Shape of the pattern: **engine keeps ownership of scheduling and LRU; the
tier is a thin adapter that (a) decides residency against a byte budget,
(b) uploads evicted-in/evicted-out experts, (c) issues groups through the
shared ABI, (d) takes results.** That adapter is the unit a qwen38 GPU path
would be measured against.

---

## 8. The dedicated precedent: DeepSeek V4 backend

When an architecture doesn't fit the generic surface, this repo's answer was
a **second, self-contained CUDA backend with its own ABI**:

- `c/backend_cuda_dsv4.cu` — `dsv4_cuda_*` symbols, init for up to 16
  devices with peer-access enablement, cuBLASLt, 32 MiB TC workspace,
  per-device streams/events (1218); its own E4M3 scale table uploaded at
  init (`cudaMemcpyToSymbol(e8_table, scale, …)`, same line); tensor uploads
  for fp8 / fp8+bf16 / fp4 / bf16 / f32 with 128-block scales (1276-1300);
  kernels for: `kv_ring_append`/`kv_comp_append` 723/744,
  `sparse_attn_batch[_cached]` 806/874/1073/1555, `indexer_score_batch` 943,
  `fp8_ref_matmul` 1023, `matmul_bf16_batch` 1119, routing 1627/2049/2122/
  2151/2168/2247 (incl. `ep2` expert-parallel variant), `expert_group` 1714,
  `expert_fp8` 1756, `moe` 1772, `moe_activation` 1803, expert bank
  upload + hash table 1995-2043 (with `tp2` tensor-parallel variant 2008),
  hyper-connection `mhc_pre`/`mhc_post` family 1301-1418, `rmsnorm` 1796,
  graph capture 1252-1265, profiler 1250-1251, `decode_state_set` 1249,
  `stream_drain` 1949.
- `c/backend_loader_dsv4.c` — Windows runtime loader, and the **arch-fit
  handshake**: two builds ship side by side, `coli_cuda_dsv4_dg.dll`
  (DeepGEMM, sm_120a tensor-core paths) and `coli_cuda_dsv4.dll` (portable,
  sm_80+); the loader tries DeepGEMM first and calls
  `dsv4_cuda_backend_arch_ok(device)` — *"does not fit this GPU; trying the
  next backend"* (383-387); `COLI_DSV4_DLL` forces one; when no DLL fits,
  *"GPU tier disabled (CPU path remains active)"* (396-397). The comment
  (11-13) states the two loaders are separate files on purpose: *"the dsv4
  ABI (dsv4_cuda_*) is entirely different from the GLM ABI (coli_cuda_*)".*
- Makefile: `CUDA_ARCH ?= native` (260) with a `portable` fat binary covering
  sm_80/86/89/90/120 (254-261); the BUILD_CONFIG stamp hashes `CUDA_ARCH`
  (713) so arch changes force rebuilds; pinned DeepGEMM sm120 checkout via
  `tools/fetch_deepgemm.sh` (226-228, 828-832); `cuda-dsv4-dll` 820-826
  (nvcc + MSVC host, `-DEF:dsv4.def`); `cuda-dsv4-dg-dll` 832-840
  (`-gencode arch=compute_120a,code=sm_120a`); oracle tests
  `dsv4-cuda-test` 959-965 (dense_batch / attention_batch / moe_batch) and
  `dsv4-cuda-loader-test` 845-846 (proves the MinGW-host ↔ MSVC-DLL ABI
  boundary).
- `c/hybrid_split.h:6`: `DSV4_HYBRID=1` splits decode experts between GPU
  bank and CPU misses — i.e. the DSV4 engine already runs the same
  "GPU-resident bank + streamed misses" hybrid the qwen38 engine needs, for
  its own architecture.

Note the deliberate overlap with qwen38's problem shape: hybrid attention
(sparse/indexer), hyper-connection pre/post norms, FP8 128-block grouped
experts, expert-parallel and tensor-parallel variants, arch-fit fallback.
DSV4 is the in-repo proof that a second dedicated backend is the accepted
way to give a hybrid architecture real GPU kernels here.

---

## 9. Build matrix (Makefile, verified)

- Default: no GPU objects at all (`CUDA_OBJ =`, 440); the default build is
  CPU (209-213: *"CUDA=1 adds an opt-in backend for resident tensors. The
  default build remains …"*).
- `CUDA=1` (Linux-only, 510): `-DCOLI_CUDA` + `CUDA_OBJ = backend_cuda.o`
  (519-521) + `INK_CUDA_OBJ = backend_cuda_ink.o` (522); Windows/HIP use
  runtime DLLs via `backend_loader.o` (494-503, 743-754:
  `COLI_CUDA_BUILDING_DLL` keeps the same `coli_cuda_*` surface in the DLL).
- `DEEPGEMM=1`: adds `COLI_DSV4_DEEPGEMM`, DeepGEMM/cutlass includes,
  relaxed-constexpr, and `COLI_DSV4_NCCL` (545-565).
- **qwen36 switch** (1020-1030): `CUDA=1` → `QWEN36_TIER_SRC = qwen36_tier.c`
  + real flags; otherwise empty tier + `NOCUDA_*` flags.
- **qwen38 (1032-1036)** — the decisive lines, verbatim:
  > *"Qwen3.8-Flash-Next text-only sibling. This target is intentionally
  > CPU-only; qwen38.c owns the direct checkpoint loader and SERVE=1
  > protocol, with no CUDA/Metal tier advertised by the control plane."*
  with the recipe `$(CC) $(NOCUDA_CFLAGS) qwen38.c -o qwen38$(EXE)
  $(NOCUDA_LDFLAGS)`.
- `NOCUDA_CFLAGS/LDFLAGS` (1097-1098) are simply the global flags with
  `-DCOLI_CUDA -DCOLI_ANS` and the cudart libs filtered out; the same guard
  is asserted by `tests/test_makefile_cuda_scope.py` (comment 1001-1006).
- qwen38 test gates that a GPU change must keep green: `qwen38-tiny-check`
  (1038-1058, matrix over `Q38_PREFILL_BATCH` × `Q38_NATIVE_BF16`),
  `qwen38-ple-prefetch-check` (1046-1050), `qwen38-vision-check` (1175-1178),
  `qwen38-vision-serve-check` (1183-1187), `test_qwen38_native_weights`
  (1195), `test_qwen38_serve_framing` (1167), `test_qwen38_tokenizer`
  (1161), `test_qwen38_config` (1164), `test_qwen38_metrics` (1192),
  `test_qwen38_prefix` (1189).
- `install` ships `qwen38` (1849, 1861); `CUDA_ARCH` is part of the
  build-config stamp (713), so `native` vs `portable` rebuilds are tracked.

---

## 10. External references (verified 2026-09-07)

- **llama.cpp PR #27742** (danielhanchen, Unsloth) — *"model: add
  Qwen3.8-Flash-Next (qwen4exp)"*: the decode graph for the architecture —
  hyper-connection residual stream, gated DeltaNet layers, MoE block with
  gated shared expert, dense full attention; converter, text graph, sparse
  attention, vision. Merged into main; the llama.cpp 0.4.0 release notes
  carry *"Added initial Qwen3.8-Flash-Next (qwen4exp) architecture support;
  optimization improvements are still pending (#27742)"*. Feature request:
  issue #27741; known follow-up: issue #27797 (multi-segment prompts
  degrade). It is the reference implementation of this exact architecture.
- **vLLM day-0 support** — official recipe
  `recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next`: FP8 checkpoint
  (`Qwen/Qwen3.8-Flash-Next-FP8`), `VLLM_PLE_CPU_OFFLOAD=1`,
  `--moe-backend triton`, TP=4, 262K native context (1M via YaRN). Two
  facts worth keeping: PLE offload is **host-RAM only** (no disk in vLLM),
  and issue #54173 documents `CUBLAS_STATUS_INTERNAL_ERROR` / illegal
  memory access in the **GDN path with prefix caching on GB10 (sm_121)** —
  i.e. even on Blackwell-class silicon the Gated DeltaNet kernels are the
  fragile part, not the MoE.
- **16 GB single-GPU field data** — `solarkyle/qwen38-flashnext-16gb`:
  measured llama.cpp tuning for UD-IQ4_XS (93.7 GB) on one 16 GB consumer
  GPU — speculative-decoding variant sweep, expert-residency slope, context
  scaling, and explicit negative results. Directly relevant to the target
  machine's two 16 GB cards. Unsloth GGUF thread (3090 Ti 24 GB, Q4_K_XL
  111.3 GB): ~15 t/s decode / ~340 t/s prefill at 30k prompt;
  `Qwen3.8-Flash-Next-UD-Q4_K_XL -ngl 999 -ot "per_layer_token_embd=CPU"`
  keeps the PLE table off VRAM.
- **Kernel-level community forks** — `Aristo94/EngramHalo.cpp` (RDNA3.5
  tuning: *"true QSA sparse gather, working MTP"*), Qiita write-up on
  placing the n-gram table on SSD with mmap (llama.cpp default is mmap;
  vLLM offloads to RAM only). Both confirm the split the colibri engine
  already made: PLE stays 0-RAM/disk, experts stream.

---

## 11. Target machine fit (facts only)

Machine: Ryzen 9 3900X (12c/24t), 64 GB DDR4, 990 PRO 1 TB NVMe,
RTX 5070 Ti 16 GB + RTX 5060 Ti 16 GB — both Blackwell `sm_120` (needs CUDA
≥ 12.8).

Against the model's memory table (`docs/qwen38.md`):

| item | size | where it can live on 16 GB |
|---|---|---|
| dense BF16 (48 layers) | 9.2 GiB | resident (fits with headroom) |
| FP8 scale bank | 28 MiB | resident |
| workspace | ≤ 1.1 GiB | resident |
| GDN recurrent state | 226 MiB | resident |
| KV + IK context | 54 KiB/token | resident, caps context length |
| experts | 120.8 GB disk, 4.7 MiB/matrix/layer | streamed (16 GB holds a small resident bank) |
| PLE | 51 GB, 160 B/token | disk, 0 RAM (by design) |

So a single 16 GB card can hold the entire non-expert state plus KV for a
useful context, leaving VRAM for a small resident expert bank — the same
shape qwen36's `CUDA_EXPERT_GB` tier parameterizes and DSV4's
`DSV4_HYBRID=1` runs. The expert bandwidth problem (96 s of the measured
140 s median on the reference CPU box) is a disk→VRAM problem; the two
existing CPU-side levers (`q38_expert_get_batch` parallel loads,
`posix_fadvise` prefetch, `COLI_MAP_EXPERTS` mmap) reduce it before any
kernel is written. Multi-device precedent: `COLI_CUDA_MAX_DEVICES 16`,
`pipe_peer_copy`, `expert_group_resident_issue/_take` and the colibri.c
per-device issue/take loops (§6.5) show how the repo already spreads work
over several cards. Toolchain fact: the repo already builds a
`sm_120a` (compute_120a) DeepGEMM artifact (`cuda-dsv4-dg-dll`, Makefile
832-840) — sm_120/120a support is proven inside this build system, and
`CUDA_ARCH=portable` fat binaries include sm_120 (254-261).

---

## 12. Landmines and open questions (recon-level observations)

1. **Token causality is a hard invariant.** The DeltaNet state update and
   the QSA cache append are sequential per token inside a call (core.h
   1485-1488, 1606); the prefill chunking is provably order-preserving
   (1485-1488 comment), which means kernels may reorder *within* a token's
   head/expert dimensions but never across tokens for the recurrent state.
   Any attention kernel must keep the "s reads cache rows written by earlier
   s in the same call" property or split prefill/decode like the engine
   already does.
2. **QSA is not absorb.** The existing `attention_absorb*` kernels compute
   attention over a contiguous KV buffer (the MLA usage in colibri). QSA
   needs: 4-token block pooling, block scoring with per-head max-relu,
   block selection with causal tail, then attention over the *gathered*
   rows plus the packed sigmoid gate on the query. No existing kernel does
   the gather/selection; composing gather→absorb through a scratch buffer
   is an open question, not an established path.
3. **The E4M3 decode table has to cross the boundary.** `coli_cuda_fp8_set_lut`
   (backend_cuda.h:83) must be called before any fmt=8 upload; the host
   table exists (`E4M3_LUT`, quant.h:478-511, cross-checked against
   `torch.float8_e4m3fn`), and the DSV4 backend uploads its own at init
   (`cudaMemcpyToSymbol(e8_table, …)`, backend_cuda_dsv4.cu:1218). A qwen38
   GPU path must do the equivalent once per device.
4. **The expert loop is the seam, and it is per-expert today.**
   `q38_moe_decode` (core.h 1675-1680) calls CPU matmul per expert with
   `Slot` descriptors that are already raw FP8 slab + scale-bank pointers —
   the exact input shape `grouped_hidden_f8_dual`/`_f8w` consume. The
   grouped kernels accept E experts × S rows (bench grid `dim3(I,S,E)`), so
   decode (S=1, E=K) is a legal existing kernel configuration; prefill's
   expert-major regrouping (1799-1811) produces exactly the grouped layout
   the kernels want.
5. **Prefill batching is a separate contract.** `q38_moe_prefill`
   (1698-1903) is documented bit-identical to the single-row path across
   F32/BF16/FP8, with cache-sized parallel load groups (1845-1878). A GPU
   prefill path must preserve that bit-identity or re-baseline the tiny
   model gate (`qwen38-tiny-check` matrix) deliberately.
6. **Residency policy is engine-owned.** LRU + scale bank + mmap opt-in
   live in `qwen38_core.h`, not in any backend; the tier precedent
   (qwen36) keeps that split — the adapter decides bytes, the engine
   decides which expert. A GPU path that re-implements expert selection
   inside the backend would break `q38_expert_get_batch`,
   `COLI_MAP_EXPERTS`, and the trace (`rt_route`) invariants.
7. **Dedicated-backend precedent sets the cost.** DSV4 is a second .cu, a
   second ABI, a second loader, an arch-fit handshake, and its own oracle
   suite. The generic-backend route (qwen36 tier style) is the cheaper
   precedent. Which one fits is the main architectural open question this
   recon deliberately does not answer.
8. **`sm_120` is untested territory for the generic backend.** All generic
   FP8 grouped kernels were benched/oracled on whatever cards this project
   has used so far; the warp rework's old-vs-new harness
   (`test_fp8_warp_cuda.cu:254-256`) is the tool to measure it on Blackwell
   before trusting either variant. vLLM issue #54173 shows GDN kernels are
   the known-fragile area on sm_12x silicon generally.
9. **The Makefile comment and the code disagree on vision.** The qwen38
   target comment (Makefile 1032) says *"text-only sibling"*, yet the
   target compiles `qwen38_vision.h` (1035), `Q38_VISION` defaults to 1
   (core.h 877), and `qwen38-vision-check` / `qwen38-vision-serve-check`
   (1175-1187) are live gates. Any GPU work must keep the vision row path
   (broadcast into the 4 residual branches, `q38_step` 1955-1991) green;
   the "text-only" phrasing should be treated as stale.
10. **MTP parity is out of scope for the engine today.** The checkpoint
    family ships a ~4B MTP head (vLLM recipes/community); colibri's engine
    does not use it (`docs/qwen38.md`). A GPU path should not be expected
    to change that, but the llama.cpp reference (MTP in the PR branch) is
    the comparison point if parity is ever requested.
11. **Control plane knows qwen38 has no GPU tier.** `docs/qwen38.md`
    (line 69) says there is currently no Qwen3.8 GPU backend and documents
    `coli plan --gpu none`; `doctor.py:455` bakes the engine's `COLI_CUDA`
    marker, and `COLI_CUDA_PIPE` appears in the plan's `tune` section
    (test_doctor.py:165). Whichever route is chosen, the plan/doctor
    surface has to advertise it honestly.
12. **Number reconciliation.** Docs say 4.7 MiB per expert slot per layer;
    the mmap comment in core.h says "14 MB per miss" — 3 matrices × 4.7 MiB
    ≈ 14.1 MiB, so both are right at different granularity (per-matrix vs
    per-slot-of-three). No discrepancy.

---

## 13. Source index

| file | role in this recon | key lines |
|---|---|---|
| `c/Makefile` | build matrix: qwen38 CPU-only, qwen36 CUDA switch, DSV4/DeepGEMM, FP8 bench/oracle targets, NOCUDA guards | 209-213, 254-277, 440, 494-522, 545-565, 713, 743-754, 797-847, 905, 959-972, 995-1036, 1046-1058, 1097-1098, 1161-1196, 1849, 1861, 1889-1896 |
| `c/qwen38_core.h` | the qwen38 engine: structures, matmul, GR, QSA, DeltaNet, MoE + expert streaming, PLE, env inventory | 12-34, 43-50, 60-71, 78-107, 189-309, 877-922, 967-1090, 1092-1148, 1150-1185, 1229-1342, 1344-1468, 1489-1648, 1653-1908, 1919-1994 |
| `c/qwen38.c` | checkpoint loader, SERVE, direct `getenv` knobs (`Q38_MAXT`, `Q38_EOS`, …) | 50, 2387-2388, 1241, 1518-1802, 1841-1843 |
| `c/quant.h` | `E4M3_LUT`, `FP8_BLOCK`, `matmul_fp8` CPU reference ("hot path is the GPU one") | 478-515, 517-543 |
| `c/st.h` | shard store + `COLI_MAP_EXPERTS` mmap cache (issue #1325) | 995-1031 |
| `c/backend_cuda.h` | generic GPU ABI: init/devices, fmt guards, uploads, expert groups, attention, pipes | 22, 57-242 |
| `c/backend_cuda.cu` | kernels: grouped f8/f8w/g4, TC gates, dispatch, env knobs, `issue`/`take` | 851, 892, 911, 933, 968, 1152, 1262, 1391, 1460-1466, 1640, 1656, 1825-1945, 1972-2141, 2061, 2175 |
| `c/backend_gpu_compat.h` | CUDA/HIP shim, `COLI_GPU_HAS_WMMA` | 9-103 |
| `c/backend_loader.c` | Windows DLL resolution of the `coli_cuda_*` ABI | 86-90, 167-168, 1411-1412, 1565-1575 |
| `c/qwen36_tier.c/.h` | working VRAM-expert-tier pattern (issue/take, byte budget, stubs) | tier.h 16-24, 68; tier.c 446, 461 |
| `c/backend_cuda_dsv4.cu/.h` | dedicated second backend: full hybrid-architecture kernel set + `dsv4_cuda_*` ABI | 699-723, 806-1119, 1218-1265, 1276-1421, 1555-1592, 1627-1803, 1949-2168, 2247-2369 |
| `c/backend_loader_dsv4.c` | DLL loader + arch-fit handshake (DeepGEMM sm_120a first, CPU fallback) | 1-19, 39-47, 349-397, 504-591 |
| `c/hybrid_split.h` | `DSV4_HYBRID=1` GPU-bank + CPU-miss expert split | 6 |
| `c/backend_vulkan.c/.h` | Vulkan expert-group sibling (`issue`/`take`, `issue2`/`take2`) | vulkan.h 63-86; vulkan.c 807, 815, 1148, 1154, 1857 |
| `c/colibri.c` | main engine multi-device MoE issue/take, Vulkan group usage, `COLI_VK_ATTN` | 633, 5025, 5786, 5808, 5933-5989, 6089, 6147-6164 |
| `c/telemetry.h` | per-device VRAM telemetry under `COLI_CUDA` | 159-241 |
| `c/doctor.py` | plan/doctor surface that advertises GPU tier state | 455 |
| `c/tests/test_fp8_cuda.cu` | FP8 grouped oracle incl. async issue/take | 4, 114-116, 152, 196-197 |
| `c/tests/test_fp8_warp_cuda.cu` | warp rework oracle + old-vs-new `COLI_CUDA_F8_WARP` comparison | 1-14, 135-141, 210-211, 254-256, 320-369 |
| `c/tests/bench_fp8_cuda.cu` | all five grouped kernels per decode shape | 69-85 |
| `c/tests/test_grouped_g4_cuda.cu` | fmt=4 grouped oracle | 4, 90, 144-145 |
| `c/tests/test_cuda_fmt_guard.c` | wrong-format refusal | 6 |
| `c/tests/test_backend_cuda.cu` | generic backend oracle incl. async group | 391-392 |
| `c/tests/qwen36_fake_cuda.h` | GPU-less test harness for the tier | 72-86 |
| `c/tests/test_qwen36_tier_shutdown.c`, `test_qwen36_tier_int8.c` | tier driven via env + fake CUDA | 41-42, 70-71 |
| `c/tests/test_efficiency_report.py` | runtime knob set in use (`COLI_GPU`, `CUDA_DENSE`, `CUDA_EXPERT_GB`) | 5-88 |
| `c/tests/test_doctor.py`, `test_resource_plan.py` | plan surface assertions | 150, 165; 345 |
| `c/tests/test_makefile_cuda_scope.py` | asserts the NOCUDA build shape | (cited from Makefile 1001-1006) |
| `docs/qwen38.md` | authoritative model facts, memory table, perf breakdown, "no GPU backend" | 69 + tables |
| `GPU_BACKENDS.md` | backend architecture, CUDA/HIP model, runtime config, DLL loader | whole file |
| external | llama.cpp PR #27742 (+ issues #27741, #27797), vLLM recipe + issue #54173, `solarkyle/qwen38-flashnext-16gb`, `Aristo94/EngramHalo.cpp`, unsloth GGUF threads, Qiita PLE-on-SSD | §10 |

*End of reconnaissance. No implementation plan is included, by request.*



