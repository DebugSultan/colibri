/* qwen38_tier.h -- optional CUDA VRAM expert tier for the qwen38 engine.
 *
 * Same concept as qwen36_tier.h ("route -> place -> overlap -> learn"): the
 * hot experts are promoted into DEVICE_LOCAL VRAM across one or more GPUs and
 * computed there through the existing expert-group API of backend_cuda.cu.
 * One home device per expert (eid % n_gpus), LFRU heat with hysteresis from
 * tier.h, uploads on a background thread through staging copies.
 *
 * THE CONSTRAINT IS INVERTED WITH RESPECT TO qwen36_tier.h.
 *
 * That tier demands cap_experts_per_layer == n_experts and keeps raw
 * pointers inside the RAM slots, which therefore must never be evicted:
 * with total residency guaranteed it can go fetch the weights itself, and
 * in fact warmstart, lookahead and LFRU swap run without ever calling back
 * into the engine. Here total residency does not exist and cannot exist:
 * the qwen38 experts live on disk (185 GB on a 64 GB machine) and the slots
 * ARE an LRU cache with eviction -- worse, with COLI_MAP_EXPERTS=1 (the
 * fastest default, Phase 0.2: 1.18 vs 1.64 s/token) the slot does not even
 * own the bytes, it POINTS into a file mapping the kernel can unmap under
 * one's feet. A pointer kept here would dangle within a few tokens.
 *
 * From which the two interface differences:
 *
 *  1. The tier OWNS its copy in VRAM and never remembers a RAM address.
 *     Between the staging copy and the upload there is no dependency on the
 *     slot that originated it: that one can be evicted right after.
 *
 *  2. The tier cannot read the disk, so it cannot promote on its own. It
 *     ASKS (q38t_plan_fill) and awaits an OFFER: the engine, which knows how
 *     to pread, loads the expert and passes it with q38t_offer(). The same
 *     holds for hot promotions -- it offers what it just loaded for a miss,
 *     which is exactly the moment the bytes are in hand.
 *
 * Since the int4 gs64 container landed the tier stages that too (fmt=4,
 * per-group scales): same three memcpy, half the weight bytes, and the
 * misses that remain read int4 from disk instead of fp8. The paragraph
 * below describes the fp8 arm, which is still the default when
 * Q38_INT4_SNAP is unset.
 *
 * In exchange the format is a gift: qwen38 is native FP8 and with
 * native_fp8 on the slot already holds raw e4m3 with a scale per 128x128
 * block (q38_load_native_fp8_ranges), i.e. **exactly** the fmt=8 that
 * backend_cuda.cu loads (:1412) and that the grouped_hidden_f8w_dual /
 * grouped_down_f8w kernels (:933,:968) chew. The staging is a memcpy, not
 * a conversion: no XOR nibble like in qwen36_tier.c:63, no expansion. The
 * BF16/F32 expansion branch of q38_load_fp8_expert_weight only concerns
 * native_fp8 off, and there the tier stays still (see q38t_init).
 *
 * The main gain is not the matmul: it is that a VRAM hit does NOT touch the
 * disk. The engine notes the routing for all K, issues the residents and
 * loads into RAM only the missing ones. With 6.02 MiB per expert at
 * cudaMalloc granularity (3*2560*640 bytes plus 300 scales charged by
 * dev_alloc_footprint, not by payload) and a ~12.5 GiB budget per 16 GB
 * card (free minus the backend reserve), about 4,300 of the 24,576 experts
 * fit in VRAM: that fraction of misses disappears from the expert-read
 * path, which Phase 0.3 showed to be the dominant cost.
 *
 * Order of use in decode (S=1):
 *
 *      q38t_note(layer, idx, K);                     // heat, all K
 *      uint32_t m = q38t_issue(layer, idx, K, xs);   // issues the residents
 *      for (k not in m) { slot = q38_expert_get(...); ...CPU...;
 *                         q38t_offer(layer, idx[k], slot...); }
 *      q38t_take(m, route_gates, K, ys);             // accumulates the residents
 *
 * The qwen38 prefill is already in expert-group form (rows grouped per
 * expert, three matmuls per group) and could use synchronous
 * coli_cuda_expert_group(), which does not have the row ceiling of the
 * asynchronous issue; it is not yet exposed here, it comes after measuring
 * decode.
 *
 * Activation: COLI_CUDA=1 [COLI_GPUS=0,1 | COLI_GPU=0] [CUDA_EXPERT_GB=<G>|auto]
 * [HEAT_FILE=<path>] [Q38T_NO_WARMSTART=1]. Compiled only when the build
 * defines -DCOLI_CUDA (CUDA=1); otherwise the inline stubs below keep the
 * engine on CPU at zero cost, as qwen36_tier.h does. */
#ifndef QWEN38_TIER_H
#define QWEN38_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init after the model load, before the first token. Returns 1 if the tier
 * is active. It does NOT ask for the RAM cache capacity: the slots can be
 * evicted as much as they want, and that is the reason this tier exists in
 * a different shape than the other. Refuses (returns 0, the engine stays on
 * CPU) if native_fp8 is off: without native FP8 the slot holds expanded F32,
 * four times larger and in a format the fmt=8 kernels do not read.
 * scale_count is how many float the scales of ONE of the three matrices are
 * worth: fp8_nblk(hidden)*fp8_nblk(inter) for fmt=8 -- the engine already
 * computed it for its per-layer scale bank -- and hidden*inter/gs for fmt=4,
 * where it is the same for all three because gs divides both axes.
 *
 * fmt is the backend's format number of the experts the engine will offer:
 * 8 for native e4m3, 4 for the int4 gs-grouped container (Q38_INT4_SNAP),
 * and gs is the group size, ignored unless fmt is 4. The tier stages ONE
 * format: the engine must not offer it fp8 experts while it is in int4 mode,
 * which is why the offer sites test the slot's kind. */
int  q38t_init(int n_layers, int n_experts, int hidden, int inter, int topk,
               int scale_count, int native_fp8, int fmt, int gs);
int  q38t_ready(void);
/* ---- dense BF16 matmul on the GPU (Q38_DENSE_GPU=1) ------------------------
 * Independent of the expert tier: the dense set is 86.9% of the per-token
 * decode byte traffic and is bf16 with no scales, so it is worth VRAM even
 * on a run where the expert tier refused to attach. The weight is uploaded
 * ONCE (fmt=9) and the device copy is cached in *slot, which the caller owns
 * and must release through q38t_dense_release.
 *
 *   slot: address of the caller's cache pointer, NULL-initialized.
 *   w:    the bf16 weight, [O,I] row-major -- exactly what the CPU kernel reads.
 *
 * Returns 1 when the GPU served it (y is written) and 0 when the caller must
 * stay on the CPU. A refusal is STICKY per weight: a tensor that would not fit
 * must not re-attempt a 1 GiB cudaMalloc on every token. */
int  q38t_dense_matmul(void **slot, float *y, const float *x, const uint16_t *w,
                       int S, int I, int O);
void q38t_dense_release(void **slot);
int  q38t_dense_enabled(void);
void q38t_dense_stats(void);

int  q38t_is_resident(int layer, int eid);
void q38t_shutdown(void);

/* Heat of the K experts routed for this token. Must be called BEFORE
 * q38t_issue and before deciding what to load: an expert resident in VRAM
 * never passes through q38_expert_get, so this is the only point where the
 * tier sees it pass. Queues nothing and does not touch the disk. */
void q38t_note(int layer, const int *eids, int K);

/* Offer: the engine has this expert's bytes in hand right now (it just
 * served a miss, or it is running a warmstart plan). The tier decides
 * whether it is worth VRAM, and if so copies to staging IMMEDIATELY -- on
 * return the caller can evict the slot, mapped or not.
 *
 *   gate,up,down: raw e4m3, [inter,hidden], [inter,hidden], [hidden,inter];
 *                 in the native cache they are the three Slot.gate/up/down
 *                 pointers (contiguous in the slab when the copy is active,
 *                 scattered in three mappings when COLI_MAP_EXPERTS=1: it is
 *                 not assumed they are).
 *   scales:       3*scale_count float [gate|up|down], i.e. this expert's
 *                 slice inside the layer's scale bank.
 *
 * planned = 1 when the expert comes from q38t_plan_fill (budget already
 * reserved at that moment); 0 for a spontaneous hot offer, which must still
 * earn its place. */
void q38t_offer(int layer, int eid,
                const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                const float *scales, int planned);

/* Issues the GPU groups for the resident subset of the K selected experts
 * (asynchronous, all devices in parallel). Returns the mask of the k taken
 * on by the GPU: those compute themselves while the CPU does the missing
 * ones. Then q38t_take().
 *
 * Note on the row ceiling: coli_cuda_expert_group_issue refuses more than 8
 * total rows per device (backend_cuda.cu:2091, "decode-scale only"). qwen38
 * has topk=10, and the home-device split (eid % n_gpus) usually leaves ~5
 * rows per card; but nothing forbids ten experts all landing on the same
 * one. Splitting the issue is not an option -- the backend admits a single
 * issue in flight per device, and a second one would require the take of
 * the first, i.e. precisely the synchronization the asynchrony was meant to
 * avoid. The extras therefore stay on the CPU via the mask, like the non
 * residents: never a wrong result, at most a slower token. The row-overflow
 * counter in q38t_stats tells whether the case actually happens. */
uint32_t q38t_issue(int layer, const int *eids, int K, const float *x);

/* Gathers the GPU results and accumulates val[k]*y_k into out[hidden]. Must
 * be called after every issue that returned a non-null mask, even if in the
 * meantime the CPU did everything else. */
void q38t_take(uint32_t mask, const float *val, int K, float *out);

/* Warmstart: the tier plans the set to fill (heat order, budget reserved
 * immediately) and hands it over; the engine loads each expert and returns
 * it with q38t_offer(..., planned=1), from as many threads as it wants.
 * q38t_fill_wait() blocks until the upload queue is empty. */
int  q38t_plan_fill(int *layers, int *eids, int max);
/* Returns the budget reserved for a planned expert that the engine then
 * decided not to load (outside its layer range, unexpected format). Without
 * it, that VRAM would stay booked by nobody forever. */
void q38t_cancel_plan(int layer, int eid);
void q38t_fill_wait(void);

/* Prefill: computes on the GPU the experts of `eids[0..count)` that are
 * resident, synchronously. Unlike the decode issue/take pair this one has no
 * row ceiling -- coli_cuda_expert_group() sizes its workspace dynamically --
 * so it takes the whole chunk of a layer in a handful of calls.
 *
 * x and y are addressed through off[]: expert c reads rows[c] consecutive
 * [hidden] rows starting at row off[c] of x and writes as many at row off[c]
 * of y -- which is exactly the group layout q38_moe_prefill already builds,
 * so no repacking happens on the engine side. done[] must come in zeroed:
 * the function sets
 * done[c]=1 for every expert it computed, and the caller keeps the others on
 * the CPU. Returns how many experts were taken.
 *
 * Numerics: the GPU arm does not reproduce the CPU accumulation order, so a
 * chunk computed here differs in the last bits from the same chunk computed
 * on the CPU. Q38_TIER_PREFILL=0 turns it off for an A/B. */
int q38t_expert_group(int layer, const int *eids, const int *rows, const int *off,
                      int count, const float *x, float *y, uint8_t *done);

/* A telemetry block on stderr: residency, hit/miss, uploads per device. */
void q38t_stats(void);

/* Host-RAM pinning hooks for total residency: the engine registers a
 * lock/unlock pair and the tier fires them at the only two instants where
 * an expert's host residency must flip -- unlock when the uploader finishes
 * a promotion (the expert computes from VRAM; its host pages are dead
 * weight), lock when a hot swap demotes a resident back to CPU fallback.
 * The tier knows nothing about files or headers: resolution stays
 * engine-side. Never registered = feature off, page-cache behavior. */
void q38t_set_host_pin(void (*lock_fn)(int,int), void (*unlock_fn)(int,int));

#else /* !COLI_CUDA: stub inline, the engine stays CPU-only */

static inline int  q38t_init(int a,int b,int c,int d,int e,int f,int g,int h,int i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;return 0;}
static inline int  q38t_ready(void){return 0;}
static inline int  q38t_dense_matmul(void**a,float*b,const float*c,const uint16_t*d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline void q38t_dense_release(void**a){(void)a;}
static inline int  q38t_dense_enabled(void){return 0;}
static inline void q38t_dense_stats(void){}
static inline int  q38t_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void q38t_shutdown(void){}
static inline void q38t_note(int a,const int*b,int c){(void)a;(void)b;(void)c;}
static inline void q38t_offer(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;}
static inline uint32_t q38t_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void q38t_take(uint32_t a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline int q38t_expert_group(int a,const int*b,const int*c,const int*d,int e,const float*f,float*g,uint8_t*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;return 0;}
static inline int  q38t_plan_fill(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void q38t_cancel_plan(int a,int b){(void)a;(void)b;}
static inline void q38t_fill_wait(void){}
static inline void q38t_stats(void){}
static inline void q38t_set_host_pin(void(*a)(int,int),void(*b)(int,int)){(void)a;(void)b;}

#endif /* COLI_CUDA */
#endif /* QWEN38_TIER_H */
