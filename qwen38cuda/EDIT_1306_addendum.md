# Integrazione da appendere al NOSTRO commento su JustVugg/colibri#1306

Commento da **modificare** (mai postarne un secondo, regola di campagna):
https://github.com/JustVugg/colibri/issues/1306#issuecomment-5736198159

Il PATCH via API restituisce `403 Resource not accessible by personal access
token`: l'account `DebugSultan` ha solo `pull` su `JustVugg/colibri` e il PAT
fine-grained non porta `Issues: write` sui repo di terzi. Serve un token classico
con scope `repo`/`public_repo`, oppure l'incolla manuale del testo qui sotto in
coda al commento esistente.

Testo gia' pronto anche come JSON pronto da inviare:
`gh api --method PATCH repos/JustVugg/colibri/issues/comments/5736198159 --input body-1306.json`
(il JSON contiene il corpo COMPLETO, vecchio + integrazione).

---
---

**EDIT 2026-09-19 — the rebaseline I promised above is done, and it corrects two things I said. Same comment edited rather than a new one, so the thread stays readable.**

**1. Fixed memory regime, and this time the regime is part of the measurement.** Prod stopped (GPUs at 2 MiB), `swapoff -a`, preflight verifying `SwapTotal=0 kB`, and per-arm evidence: `pswpin`/`pswpout` delta **0 pages** on all four arms, `read_bytes` 2.4–3.8 MB per run (i.e. no disk I/O during the measurement — 34.70 GB of experts wired with mlock, 0 lock failures), storage probe 5.73 → 5.68 GB/s across the battery, and `parallel-batches` = 1258/1258/2056/1258, all **> 0**, so no arm sat in the `cap < topk` cliff.

**2. The decode numbers I asked you to treat as pending did not survive.** 64 tokens, `CACHE=8192 BITS=8 OMP=20`, 2k prompt, same read-only heat file for every arm: `ours` **0.2193 tok/s** — `hybrid` (`Q38_TIER_TRUNK=2`) **+0.69%** — `hybrid_pf` (`+Q38_TRUNK_PREFILL=1`) **+0.17%** — `trunk` (`Q38_TIER_TRUNK=1`) **−12.39%**. The first two are noise, and the counters say so unambiguously: across `ours`, `hybrid` and `hybrid_pf`, `weight-ranges`, `coalesced-gate-up`, `prefetched`, `parallel-batches`, cache hit rate (92.1%) and the GPU/CPU split (70.1% on GPU) are **identical digit for digit**. So: **the +40%/+11% hybrid gains were an artifact of the contaminated regime — please do not carry them forward.**

**3. The part that concerns your tier directly, and it is my mistake to report.** In that battery your VRAM expert tier showed `VRAM hit rate 0.0% | LFRU swaps 0` in every trunk arm — 553 tensors and ~4.0 GB resident in `trunk`, 16590+16707 tensors and 14.24+14.34 GB in `hybrid_pf`, and **zero hits** in all of them. It was not a policy that never promotes. It was unreachable by construction on my side: my battery sets `Q38_INT4_SNAP`, `q38_load_expert` opens with `if (q38_try_load_int4_expert(...)) return;`, so every expert lands in `i4_slab` and `fp8_slab` stays NULL — and my note hook requires a *native FP8* slot (`ex->fp8_slab && ex->gate.data == ex->fp8_slab`), so `qt_note()` was never called once. `q_skips 0` rules out queue congestion. Positive control, same arm with only `Q38_INT4_SNAP` removed: **`resident 115/24576`, `uploads 115`, hit rate 4.9%** — the tier wakes up exactly as designed.

**Consequence, stated plainly: that −12.39% measures your *dense* int8 trunk (`553 of 553 offered matrices resident as int8`, genuinely active in every arm) with your *expert tier* switched off. On experts, the two designs were never compared. Please do not read it as a verdict on your expert tier — it isn't one, and I would rather correct that here than have it quoted somewhere as if it were.**

For what it is worth, the dense int8 trunk is the best dense path measured on this rig (42.4+33.2 vs 100.6+69.9 ms/fwd), and that result is unaffected by any of the above.

**4. Method note, since it cost me an arm.** I first hypothesised the starving guard came from the mapping branch in `q38_load_native_fp8_ranges` and predicted `COLI_MAP_EXPERTS=0` would wake the tier. Control run: **no change, `miss(CPU) 3360` identical bit for bit**. The chain was coherent and simply not the one being taken. A coherent code path is not a measurement, and the control has to be run on the flag you believe is causal.

The fmt=9 (bf16) PR is ready — patch verified against `dev` @ `5a0b725` with the test green — and lands next.
