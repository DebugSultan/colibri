/* qwen38_tier.c -- CUDA VRAM expert tier for the qwen38 engine. See header.
 *
 * Close relative of qwen36_tier.c, but with the inverted constraint the
 * header explains: here the tier cannot read the disk and never keeps
 * pointers to RAM, so it promotes only what the engine OFFERS it. From this
 * follows the most visible structural difference from the other file: there
 * is no periodic LFRU tick hunting for candidates (it could there, because
 * the bytes were always reachable). The promotion decision is taken inside
 * q38t_offer, i.e. at the only instant the bytes really exist -- and it is
 * also the right instant, because an offer is born from a miss just served:
 * the expert has just proven itself useful. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "qwen38_tier.h"
#include "backend_cuda.h"
#include "tier.h"
#include "quant.h"            /* E4M3_LUT */

#define Q38T_MAX_DEV  8
/* VRAM reserve for the backend. The default is the measured value -- see the
 * comment at the budget in q38t_init; Q38T_DEV_RESERVE_MB (MiB) overrides it
 * so the breaking point is measured stepwise instead of guessed. */
#define Q38T_DEV_RESERVE ((size_t)3584*1024*1024)   /* 3.5 GiB default */
#define Q38T_QCAP     16      /* staging ~4.7 MB/item -> ~75 MB ceiling */
#define Q38T_MAX_ROWS 8       /* backend_cuda.cu:2091, "decode-scale only" */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint32_t arena_idx;                   /* slot index when the arena is on */
    uint8_t resident, queued, planned;
} Q38TSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    /* Expert format as the backend numbers it: 8 = e4m3 with a scale per
     * 128x128 block, 4 = int4 nibbles with a scale per group of gs along the
     * input axis (gs is 0 when fmt is 8). Both are staged with the same three
     * memcpy; only the byte counts and the upload argument change. */
    int fmt, gs;
    size_t sc;                            /* scale float per matrix */
    size_t mat_bytes;                     /* weight bytes per matrix */
    size_t exp_bytes;                     /* VRAM estimate per expert */
    int dev[Q38T_MAX_DEV];
    size_t budget[Q38T_MAX_DEV], used[Q38T_MAX_DEV];
    /* Expert arena: one cudaMalloc per device (pipe_alloc), slots of EXACT
     * payload at fixed offsets, LIFO free-list under G.mx. A slot is taken
     * where the budget is charged (q38t_offer / q38t_plan_fill) and given
     * back where the charge is released (cancels, upload failure); a swap
     * TRANSFERS the victim's slot to the replacement at enqueue time, so the
     * free count always mirrors used[]/exp_bytes exactly. The tensors are
     * built by coli_cuda_tensor_upload_into and own nothing, so the free
     * list -- not cudaFree -- is the only recycling, and the +26% padding
     * the six per-expert cudaMallocs paid (dev_alloc_footprint) is gone. */
    uint8_t *arena[Q38T_MAX_DEV];
    uint32_t *slot_free[Q38T_MAX_DEV];
    int slot_free_n[Q38T_MAX_DEV];
    Q38TSlot *slot;                       /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* upload queue with staging copies */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[Q38T_QCAP];
    int qh, qt_, qn;
    /* Enqueued and not yet resident. qn frees the ring slot at DEQUEUE time,
     * before the backend copy runs, so "queue empty" is not "all resident"
     * (the warmstart path waits on this one, #1360's class of defect). */
    int inflight;
    pthread_cond_t cv;
    pthread_cond_t cv_take;               /* queue space + qt_take done */
    /* statistics */
    uint64_t hits[Q38T_MAX_DEV], miss, uploads, upload_fail;
    uint64_t offers, promotions, swaps, q_full_skips, overflow_rows, take_fails;
    /* Plans the engine gave back instead of offering. Cancelling is the
     * legitimate answer for a plan it cannot execute (out-of-range layer,
     * expert in a format the tier does not stage), so it is silent -- and
     * that silence hid a mass cancellation that left the tier at 0 resident
     * with every other counter also at 0. Counted, it cannot hide again. */
    uint64_t plan_cancels;
    uint64_t tick;
    /* Hot-path stopwatches. These were added to settle one question -- is the
     * GPU branch latency-bound or bandwidth-bound? -- and the measured answer
     * is neither. Over two 128-token runs the whole hot path costs 0.944 s and
     * 1.006 s against 52.8 s and 58.0 s of routed-expert: 1.8% of it, 0.7% of
     * wall clock. Both candidate levers are therefore dead, and the numbers
     * are kept so nobody revives them:
     *  - take is 8.3 us/call on dev0 and 3.7 us/call on dev1. Since take()
     *    collects dev0 first, dev0's wait absorbs the common GPU time and
     *    dev1's figure is its EXCESS over dev0 -- NOT its absolute compute
     *    time. That excess is nil, so the 2x slower card (dev1 is a 5060 Ti
     *    at 448 GB/s against dev0's 5070 Ti at 896 GB/s) is not gating the
     *    pair: both are done while the CPU grinds the non-resident rows. A
     *    bandwidth-weighted shard instead of eid%2 would buy nothing.
     *  - issue is the dearest of the three (86 us/call on dev0, 59 on dev1)
     *    but totals 0.85 s out of 140 s, so batching the 48 per-forward
     *    issues is worth 0.6% at most.
     * What remains: the tier's whole win comes from not serving 69% of the
     * rows on the CPU, and moving them to the GPU is very nearly free. The
     * decode levers left are residency and the cost of CPU-side rows. */
    uint64_t t_issue[Q38T_MAX_DEV], t_take[Q38T_MAX_DEV];
    uint64_t n_issue[Q38T_MAX_DEV], n_take[Q38T_MAX_DEV];
    uint64_t t_acc;                       /* accumulate loop, pure CPU */
    /* issue state of the (single) decode thread */
    int is_cnt[Q38T_MAX_DEV];
    int is_k[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    float *is_x; size_t is_x_floats;
    int issue_open;                       /* no free while a group is in flight */
    /* prefill state: staging for the synchronous grouped path. Separate from
     * is_x because the two regimes never share a shape -- decode issues
     * ndev*MAX_ROWS rows of D, prefill gathers a whole device batch. */
    float *pf_x,*pf_y; size_t pf_floats;
    uint64_t pf_calls,pf_batches,pf_experts,pf_rows,pf_refused,pf_absent,t_pf;
    /* warmstart */
    int *fill_order; int fill_n, fill_cur;
    uint32_t *heat0;
} G;

static Q38TSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Host-RAM pinning hooks (residenza totale): the ENGINE owns the resolution
 * (safetensors headers + mlock), the tier only owns the instants. Unlock
 * fires when an upload completes (the expert computes from VRAM: its host
 * pages are dead weight, and leaving them locked would grow the pinned set
 * towards 64.8 GB of never-touched bytes); lock fires when a hot swap
 * demotes a resident back to CPU fallback. Unregistered = no-op: the host
 * side stays on page cache exactly as before. */
static void (*g_pin_lock)(int,int);
static void (*g_pin_unlock)(int,int);
void q38t_set_host_pin(void (*lock_fn)(int,int), void (*unlock_fn)(int,int)){
    g_pin_lock=lock_fn; g_pin_unlock=unlock_fn;
}

/* Arena free-list. Every take and give happens under G.mx, in lockstep with
 * the budget charge it mirrors; the uploader reads only the arena_idx it
 * already owns. UINT32_MAX from arena_take means "no slot" -- the budget
 * invariant (used+exp <= budget) makes it unreachable; if it ever happens
 * the offer is dropped and counted, never served silently. */
static uint32_t arena_take(int di){
    if(!G.arena[di] || G.slot_free_n[di]<=0) return UINT32_MAX;
    return G.slot_free[di][--G.slot_free_n[di]];
}
static void arena_give(int di, uint32_t idx){
    if(!G.arena[di] || idx==UINT32_MAX) return;
    G.slot_free[di][G.slot_free_n[di]++]=idx;
}

/* Reserve in bytes for one device: Q38T_DEV_RESERVE_MB (MiB), default
 * Q38T_DEV_RESERVE. Read on every q38t_init -- once per process in production
 * -- so tests can flip the env between inits without a reset hook.
 *
 * The env takes either one value, which applies to every card exactly as it
 * always did, or a comma list indexed BY DEVICE ID: "1024,6144" leaves 1 GiB
 * free on dev0 and 6 GiB on dev1. The asymmetric form exists because the cards
 * are not equal and neither is the work: the expert arena fills both cards at
 * warmstart while the dense tensors upload lazily on first use, so whatever the
 * arena takes, the dense set never sees. A larger reserve on the strong card is
 * how you hand that card to the dense path without starving the weak one of
 * experts. A device id past the end of the list falls back to the default.
 *
 * INDEX SPACE: the list is indexed by the engine's own device ids -- CUDA
 * enumeration, which is fastest-first and NOT nvidia-smi's PCI order on a mixed
 * machine. The "[q38dense] rank" line at startup names every id ("dev 0 (NVIDIA
 * GeForce RTX 5070 Ti, ...)"); write the list against THAT, or invert the intent
 * silently -- the first asymmetric run did exactly that and starved the strong
 * card of the dense reserve it was being given. */
static size_t dev_reserve_for(int device){
    const char *m=getenv("Q38T_DEV_RESERVE_MB");
    if(!m || !*m) return Q38T_DEV_RESERVE;
    if(!strchr(m,',')){
        double mb=atof(m);
        return mb>0 ? (size_t)(mb*1024.0*1024.0) : Q38T_DEV_RESERVE;
    }
    char buf[128]; snprintf(buf,sizeof buf,"%s",m);
    int idx=0;
    for(char *t=strtok(buf,","); t; t=strtok(NULL,","), idx++){
        if(idx!=device) continue;
        double mb=atof(t);
        return mb>0 ? (size_t)(mb*1024.0*1024.0) : Q38T_DEV_RESERVE;
    }
    return Q38T_DEV_RESERVE;
}

/* Local clock: the tier does not include qwen38_core.h and should not. In
 * nanoseconds because the individual calls sit in the microsecond range and a
 * double of seconds summed tens of thousands of times loses the bottom end. */
static uint64_t now_ns(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

/* Staging. In qwen36 this function had to bring the two's-complement int4
 * nibbles back to biased binary; here the format in RAM IS already the
 * backend's -- raw e4m3 for fmt=8, and for fmt=4 the offset-binary nibbles
 * the backend's own offset_to_signed_s4 kernel expects -- so it is three
 * copies and nothing else. The three matrices are NOT assumed contiguous:
 * with COLI_MAP_EXPERTS=1 the slot points at three distinct file mappings.
 * The three scale blocks ARE assumed contiguous and of equal length: fmt=8
 * bills each matrix fp8_nblk(D)*fp8_nblk(Ih) floats, and fmt=4 bills
 * D*Ih/gs for all three because q38_int4_group_size only accepts a gs that
 * divides both D and Ih. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                  const float *scales){
    memcpy(dw,                  gate, G.mat_bytes);
    memcpy(dw+G.mat_bytes,      up,   G.mat_bytes);
    memcpy(dw+2*G.mat_bytes,    down, G.mat_bytes);
    memcpy(dsc, scales, 3*G.sc*sizeof(float));
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%Q38T_QCAP; G.qn--;
        pthread_cond_broadcast(&G.cv_take);          /* queue space */
        if(ve>=0){
            /* swap: the victim is freed only when no group is in flight */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            Q38TSlot *v=qs(vl,ve);
            if(G.th_stop && G.issue_open){
                 /* Shutdown with a group still open: q38t_take, the only thing
                  * that clears issue_open, will never come. The swap is
                  * abandoned instead of freeing a tensor the group in flight
                  * can still read; the resident flag had already been turned
                  * off by the one who queued, so it is put back the way it
                  * was. */
                v->resident=1; qs(layer,eid)->queued=0; G.inflight--;
                pthread_cond_broadcast(&G.cv_take);
                pthread_mutex_unlock(&G.mx); free(w); free(sc); continue;
            }
            ColiCudaTensor *a=v->tg,*b=v->tu,*c=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            /* The tensors own nothing (arena), so this frees descriptors and
             * accounting only. The victim's slot is NOT returned: it was
             * transferred to the replacement at enqueue time. */
            if(a)coli_cuda_tensor_free(a);
            if(b)coli_cuda_tensor_free(b);
            if(c)coli_cuda_tensor_free(c);
        } else pthread_mutex_unlock(&G.mx);

        int dv=G.dev[home(eid)];
        uint32_t aidx=qs(layer,eid)->arena_idx;
        uint8_t *slotw=G.arena[home(eid)] + (size_t)aidx*G.exp_bytes;
        float *slotsc=(float*)(slotw + 3*G.mat_bytes);
        /* gate/up are [inter,hidden], down is [hidden,inter]; the signature
         * wants (I=input, O=output), not (rows, columns). The arena slot is
         * [gate|up|down][s_gate|s_up|s_down] at exact payload offsets. */
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok = coli_cuda_tensor_upload_into(&tg, w,               sc,        G.fmt, G.D,  G.Ih, dv, G.gs, slotw,                 slotsc)
              && coli_cuda_tensor_upload_into(&tu, w+G.mat_bytes,   sc+G.sc,   G.fmt, G.D,  G.Ih, dv, G.gs, slotw+G.mat_bytes,     slotsc+G.sc)
              && coli_cuda_tensor_upload_into(&td, w+2*G.mat_bytes, sc+2*G.sc, G.fmt, G.Ih, G.D,  dv, G.gs, slotw+2*G.mat_bytes,   slotsc+2*G.sc);
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        Q38TSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid);
                G.upload_fail++;
                if(G.used[hd]>=G.exp_bytes) G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* card really full: stop */
                arena_give(hd,aidx);
                if(tg)coli_cuda_tensor_free(tg);
                if(tu)coli_cuda_tensor_free(tu);
                if(td)coli_cuda_tensor_free(td); }
        s->queued=0; s->planned=0;
        G.inflight--;
        pthread_cond_broadcast(&G.cv_take);          /* this upload is complete */
        pthread_mutex_unlock(&G.mx);
        /* Promotion complete: unpin the expert's host pages. Fired outside
         * the mutex -- mlock/munlock can wait on I/O, the tier mutex never
         * must. */
        if(ok && g_pin_unlock) g_pin_unlock(layer,eid);
    }
}

/* --- init ---------------------------------------------------------------- */

/* Thread affinity around the tier's own threads (Linux).
 *
 * With OMP_PROC_BIND set, libgomp binds the initial thread to place 0 before
 * main() runs, and a pthread inherits the CPU mask of the thread that creates
 * it. The uploader thread and the CUDA runtime's own threads were therefore
 * jailed on the OpenMP master's core: every staging copy and every driver
 * call competed with the master thread's share of each expert matmul, and
 * the whole team waited for it. Measured on Qwen3.8 upstream (12 threads, one
 * card): the CPU time per remaining expert rose 64 % while the tier was on,
 * eating the whole gain of computing 45-59 % of the experts on the GPU. So
 * the tier widens the calling thread's mask to every online CPU while it
 * creates its thread and initializes CUDA, and restores the caller's mask
 * afterwards. Raw syscalls, no _GNU_SOURCE: this file is also #included by
 * tests after the engine's own headers. Same fix as qwen36_tier.c
 * (792b6f2); this branch runs 20 OpenMP threads by default. */
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#define Q38T_AFF_WORDS 64                             /* 4096 CPUs */
typedef struct { unsigned long w[Q38T_AFF_WORDS]; int len; } q38t_affmask;
static int q38t_aff_get(q38t_affmask *m){
    long r=syscall(SYS_sched_getaffinity,0,sizeof m->w,m->w);
    if(r<=0) return 0;
    m->len=(int)r; return 1;
}
static void q38t_aff_widen(const q38t_affmask *saved){
    if(!saved->len) return;
    long n=sysconf(_SC_NPROCESSORS_ONLN);
    if(n<=1) return;
    q38t_affmask all; memset(&all,0,sizeof all);
    for(long i=0;i<n && i<(long)(8*sizeof all.w);i++) all.w[i/(8*sizeof(unsigned long))] |= 1ul<<(i%(8*sizeof(unsigned long)));
    syscall(SYS_sched_setaffinity,0,(size_t)saved->len,all.w);
}
static void q38t_aff_restore(const q38t_affmask *saved){
    if(saved->len) syscall(SYS_sched_setaffinity,0,(size_t)saved->len,saved->w);
}
#else
typedef struct { int len; } q38t_affmask;
static int  q38t_aff_get(q38t_affmask *m){ (void)m; return 0; }
static void q38t_aff_widen(const q38t_affmask *m){ (void)m; }
static void q38t_aff_restore(const q38t_affmask *m){ (void)m; }
#endif

/* VRAM an allocation of `bytes` really occupies: kept as documentation of the
 * cudaMalloc granularity the arena exists to bypass. NOT compiled: the tier
 * no longer pays the rounding (every expert is one payload slot), and a dead
 * table would rot. The measured rule, driver 5xx, 256 allocations each:
 *   400 B, 3 KiB, 4 KiB -> 8 KiB      10 KiB -> 16 KiB     16..64 KiB -> exact
 *   96 KiB -> 104 KiB   384 KiB -> 416 KiB   768 KiB -> 1 MiB   1 MiB -> 1 MiB
 *   1.5 MiB -> 2 MiB    3 MiB -> 4 MiB
 * i.e. above 1 MiB multiples of 2 MiB, above 512 KiB one 1 MiB page, and
 * below that roughly the size plus a sixteenth, in 8 KiB steps, 8 KiB
 * minimum. */

int q38t_init(int nl, int ne, int D, int Ih, int topk, int scale_count,
              int native_fp8, int fmt, int gs){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(G.on){
        /* Process singleton: the slots are sized for ONE geometry. A second
         * model (Segment adapter) cannot share them, so it is told no and
         * stays on CPU. That is why the engine remembers the return of this
         * function instead of trusting q38t_ready(). */
        fprintf(stderr,"[q38tier] already attached to another model in this "
                       "process -> CPU path for this one\n");
        return 0;
    }
    if(fmt!=8&&fmt!=4){
        fprintf(stderr,"[q38tier] expert format %d unsupported (8=fp8, 4=int4 "
                       "grouped) -> CPU path\n",fmt);
        return 0;
    }
    if(fmt==8&&!native_fp8){
        fprintf(stderr,"[q38tier] native FP8 disabled: the RAM slots hold expanded "
                       "F32, which no fmt=8 kernel reads -> CPU path\n");
        return 0;
    }
    if(fmt==4){
        /* The group size has to divide both axes, or the three matrices do
         * not bill the same number of scales and the single G.sc stride in
         * stage() and in the uploader would be wrong. q38_int4_group_size
         * already refuses anything else at load time; this is the tier's own
         * check, because a tier that silently stages the wrong stride is the
         * defect class that costs a whole window. */
        if(gs<8||D%gs||Ih%gs||(size_t)scale_count!=(size_t)D*(size_t)Ih/(size_t)gs){
            fprintf(stderr,"[q38tier] int4 geometry refused: gs=%d, D=%d, Ih=%d, "
                           "scale_count=%d (expected %zu) -> CPU path\n",
                    gs,D,Ih,scale_count,(size_t)D*(size_t)Ih/(gs>0?(size_t)gs:1));
            return 0;
        }
        if((int64_t)D*Ih%2){
            fprintf(stderr,"[q38tier] int4 needs an even D*Ih to pack -> CPU path\n");
            return 0;
        }
    }
    if(topk>Q38T_MAX_ROWS*Q38T_MAX_DEV){
        fprintf(stderr,"[q38tier] topk=%d unsupported\n",topk); return 0;
    }
    if(nl<1||ne<1||D<1||Ih<1||scale_count<1) return 0;
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk; G.sc=(size_t)scale_count;
    G.fmt=fmt; G.gs=(fmt==4)?gs:0;
    G.mat_bytes=(fmt==4) ? (size_t)D*(size_t)Ih/2 : (size_t)D*(size_t)Ih;
    /* The charge is the exact payload: the arena (below) stores experts at
     * fixed offsets with no per-allocation rounding, so the budget counts
     * real bytes. The old footprint charge billed 3.33 MiB for a 2.64 MiB
     * int4 expert -- ~1900 experts per card lost to cudaMalloc granularity. */
    G.exp_bytes = 3*G.mat_bytes + 3*G.sc*sizeof(float);

    /* Devices: COLI_GPUS="0,1" (default: first two visible devices).
     * COLI_GPU is the singular the planner writes for a one-device plan;
     * accepting it too means a planner-written list selects a device instead
     * of silently selecting none. */
    const char *gl=getenv("COLI_GPUS");
    if(!gl || !*gl) gl=getenv("COLI_GPU");
    if(gl && *gl){
        char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
        for(char *t=strtok(buf,","); t && G.ndev<Q38T_MAX_DEV; t=strtok(NULL,","))
            G.dev[G.ndev++]=atoi(t);
    } else {
        int available=coli_cuda_available_device_count();
        int want=available<2?available:2;
        for(int i=0;i<want && i<Q38T_MAX_DEV;i++) G.dev[G.ndev++]=i;
        fprintf(stderr,"[q38tier] COLI_GPUS unset: selecting %d visible device(s)\n",G.ndev);
    }
    if(G.ndev<1){ fprintf(stderr,"[q38tier] no visible CUDA devices -> CPU path\n"); return 0; }
    q38t_affmask aff; q38t_aff_get(&aff); q38t_aff_widen(&aff);   /* CUDA's threads are born here */
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[q38tier] coli_cuda_init failed -> CPU path\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev) G.ndev=have;
    if(G.ndev<1){ fprintf(stderr,"[q38tier] no CUDA devices -> CPU path\n"); return 0; }

    /* The e4m3 LUT must be published BEFORE any fmt=8 upload: without it,
     * the backend refuses them instead of decoding against a table of zeros.
     * fmt=4 never decodes e4m3, so a failure there is not fatal -- it is
     * published anyway because the backend is process-wide and the cost is a
     * 1 KiB copy. */
    if(!coli_cuda_fp8_set_lut(E4M3_LUT) && G.fmt==8){
        fprintf(stderr,"[q38tier] coli_cuda_fp8_set_lut failed -> CPU path\n");
        return 0;
    }
    q38t_aff_restore(&aff);

    /* The expert charge is the exact payload (set with the geometry above).
     * It used to be the cudaMalloc footprint (3.33 MiB billed for a 2.64 MiB
     * int4 expert): honest while each expert was six separate cudaMallocs,
     * and the over-commit froze budgets permanently via the clamp
     * (G.budget[hd]=G.used[hd] in the uploader). The arena below removes the
     * rounding itself, so the charge follows it: real bytes, no slack. */
    /* How much VRAM to leave to the backend. One GiB was an eyeball estimate,
     * and it is measured wrong: the backend allocates its work tensors and
     * cuBLASLt workspaces AFTER the tier has already taken its weights, so
     * the "free" read here is the most optimistic value of the whole run.
     * 128-token probe with a 1 GiB reserve: declared budget 14.2/14.3 GB,
     * two "[CUDA] tensor allocation: out of memory", and the uploader's
     * adaptive clamp settled at 11.9/12.0 GB -- the backend needed ~3.3 GB,
     * not 1. Q38T_DEV_RESERVE is the default and the clamp stays as a net:
     * if the card fills up anyway the tier stops promoting instead of failing
     * the run. Q38T_DEV_RESERVE_MB lowers it stepwise (3584 -> 2048 -> 1024,
     * watching upload_fail) because in prefill the backend takes ~200 MB per
     * card, and the reserve is worth ~1900 experts per GiB. */
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t reserve=dev_reserve_for(G.dev[i]);
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        /* An explicit budget above what the card can still take made every
         * upload fail until the uploader's clamp kicked in (#1411's lesson):
         * the ceiling is min(requested, measured headroom), and the banner
         * says so. */
        size_t headroom = freeb>reserve ? freeb-reserve : 0;
        size_t b = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : headroom;
        if(b>headroom){
            fprintf(stderr,"[q38tier] dev %d: explicit budget %.1f GB above the "
                           "measured headroom %.1f GB, clamped\n",
                    G.dev[i], b/1073741824.0, headroom/1073741824.0);
            b=headroom;
        }
        G.budget[i]=b;
        fprintf(stderr,"[q38tier] dev %d: %.1f GB free, reserve %.2f GiB, "
                       "budget %.1f GB (~%zu experts)\n",
                G.dev[i], freeb/1073741824.0, reserve/1073741824.0,
                b/1073741824.0, b/G.exp_bytes);
    }
    /* Arena: exact payload slots instead of six rounded cudaMallocs per
     * expert (dev_alloc_footprint bills 3.33 MiB for a 2.64 MiB int4 expert,
     * ~1900 experts per card lost to rounding). One cudaMalloc per device,
     * sized by the measured budget above; uploads go through
     * coli_cuda_tensor_upload_into and own nothing, so the free-list is the
     * only recycling and swaps stop paying free+malloc churn. All-or-nothing:
     * if any card's arena fails, the tier declines -> CPU path, the same
     * answer as any other backend failure. */
    {
        size_t payload = 3*G.mat_bytes + 3*G.sc*sizeof(float);
        int ok = 1;
        for(int i=0;i<G.ndev && ok;i++){
            size_t slots = G.budget[i] / payload;
            G.arena[i] = slots ? coli_cuda_pipe_alloc(G.dev[i], slots*payload) : NULL;
            G.slot_free[i] = slots ? malloc(slots*sizeof(uint32_t)) : NULL;
            ok = G.arena[i] && G.slot_free[i];
            if(ok){
                G.slot_free_n[i] = (int)slots;
                for(size_t k=0;k<slots;k++) G.slot_free[i][k]=(uint32_t)k;
                G.budget[i] = slots*payload;      /* exact charge from here on */
                fprintf(stderr,"[q38tier] dev %d: arena %zu slots x %.2f MB payload"
                               " (the old six-cudaMalloc charge was 3.33 MB/expert)\n",
                        G.dev[i], slots, payload/1048576.0);
            }
        }
        if(!ok){
            for(int i=0;i<G.ndev;i++){
                if(G.arena[i]) coli_cuda_pipe_free(G.dev[i],G.arena[i]);
                free(G.slot_free[i]);
                G.arena[i]=NULL; G.slot_free[i]=NULL; G.slot_free_n[i]=0;
            }
            fprintf(stderr,"[q38tier] arena alloc failed -> CPU path\n");
            return 0;   /* nothing else allocated yet: same as the early outs */
        }
    }
    G.slot=calloc((size_t)nl*ne,sizeof(Q38TSlot));
    G.is_x_floats=(size_t)G.ndev*Q38T_MAX_ROWS*D;
    G.is_x=malloc(G.is_x_floats*sizeof(float));
    if(!G.slot||!G.is_x){
        free(G.slot); free(G.is_x);
        for(int i=0;i<G.ndev;i++){
            if(G.arena[i]) coli_cuda_pipe_free(G.dev[i],G.arena[i]);
            free(G.slot_free[i]);
            G.arena[i]=NULL; G.slot_free[i]=NULL; G.slot_free_n[i]=0;
        }
        return 0;
    }

    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u &&
               hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++)
                        G.slot[i].heat=tier_decay_value(G.heat0[i]);
                    fprintf(stderr,"[q38tier] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    pthread_mutex_init(&G.mx,NULL);
    pthread_cond_init(&G.cv,NULL);
    pthread_cond_init(&G.cv_take,NULL);
    q38t_aff_get(&aff); q38t_aff_widen(&aff);   /* the uploader inherits this mask */
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0){
        free(G.slot); free(G.is_x);
        for(int i=0;i<G.ndev;i++){
            if(G.arena[i]) coli_cuda_pipe_free(G.dev[i],G.arena[i]);
            free(G.slot_free[i]);
            G.arena[i]=NULL; G.slot_free[i]=NULL; G.slot_free_n[i]=0;
        }
        return 0;
    }
    q38t_aff_restore(&aff);
    G.on=1;
    fprintf(stderr,"[q38tier] CUDA VRAM expert tier active: %d device(s), "
                   "fmt=%d%s, %.2f MB/expert, %d experts total\n",
            G.ndev, G.fmt, G.fmt==4?" (int4 grouped)":" (fp8 e4m3)",
            G.exp_bytes/1048576.0, nl*ne);
    return 1;
}

int q38t_ready(void){ return G.on; }

int q38t_is_resident(int layer,int eid){
    if(!G.on||layer<0||layer>=G.nl||eid<0||eid>=G.ne) return 0;
    pthread_mutex_lock(&G.mx);
    int r=qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

/* --- heat ---------------------------------------------------------------- */

void q38t_note(int layer,const int *eids,int K){
    if(!G.on||!eids||layer<0||layer>=G.nl) return;
    pthread_mutex_lock(&G.mx);
    if(layer==0){
        G.tick++;
        /* Periodic decay: an old load must not own the VRAM forever. The
         * promotions, unlike qwen36_tier, are not decided here -- the bytes
         * are needed, and there are none here. */
        if(!(G.tick%1024)){
            size_t n=(size_t)G.nl*G.ne;
            for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
        }
    }
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne) continue;
        Q38TSlot *s=qs(layer,e);
        if(s->heat<0xFFFFFFFFu) s->heat++;
    }
    pthread_mutex_unlock(&G.mx);
}

/* --- offer and promotion -------------------------------------------------- */

/* Called with the lock held. Returns 1 if queued. */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,
                          const uint8_t *gate,const uint8_t *up,
                          const uint8_t *down,const float *scales){
    if(G.qn>=Q38T_QCAP){ G.q_full_skips++; return 0; }
    uint8_t *w=malloc(3*G.mat_bytes);
    float *sc=malloc(3*G.sc*sizeof(float));
    if(!w||!sc){ free(w); free(sc); G.q_full_skips++; return 0; }
    stage(w,sc,gate,up,down,scales);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid;
    G.q[G.qt_].w=w;         G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%Q38T_QCAP; G.qn++; G.inflight++;
    qs(layer,eid)->queued=1;
    pthread_cond_signal(&G.cv);
    return 1;
}

void q38t_offer(int layer,int eid,
                const uint8_t *gate,const uint8_t *up,const uint8_t *down,
                const float *scales,int planned){
    if(!G.on||!gate||!up||!down||!scales) return;
    if(layer<0||layer>=G.nl||eid<0||eid>=G.ne) return;
    long rel=-1;                              /* demoted victim awaiting relock */
    pthread_mutex_lock(&G.mx);
    G.offers++;
    Q38TSlot *s=qs(layer,eid);
    if(s->resident||s->queued){ pthread_mutex_unlock(&G.mx); goto out; }

    int di=home(eid);
    if(planned){
        /* The budget was already reserved by q38t_plan_fill: queue and done.
         * If the plan has meanwhile been cancelled (planned=0), it falls
         * back to the normal path below. */
        if(s->planned){
             /* Here it WAITS for queue space, in contrast with the hot path
              * below. Discarding a planned offer throws away the disk read
              * the engine just did to build it, and since discarding returns
              * the budget, the next q38t_plan_fill round re-plans it
              * identically: in the measured warmstart, 4535 experts out of
              * 9792 were read, copied and thrown away (46%). The warmstart
              * is by definition a "fill and wait" phase -- q38t_fill_wait()
              * exists for that -- so blocking takes nothing from anyone. In
              * the hot path instead, giving up stays right: there is a token
              * waiting there. */
            while(G.qn>=Q38T_QCAP && !G.th_stop)
                pthread_cond_wait(&G.cv_take,&G.mx);
             /* The wait released the lock: the slot may have changed. */
            if(s->resident||s->queued||!s->planned){
                if(s->planned){
                    if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes;
                    arena_give(di,s->arena_idx);
                    s->planned=0;
                }
                pthread_mutex_unlock(&G.mx); goto out;
            }
            if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
            else { if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes;
                   arena_give(di,s->arena_idx); s->planned=0; }
            pthread_mutex_unlock(&G.mx); goto out;
        }
    }

    if(G.used[di]+G.exp_bytes<=G.budget[di]){
        uint32_t idx=arena_take(di);
        if(idx!=UINT32_MAX){
            G.used[di]+=G.exp_bytes;
            s->arena_idx=idx;
            if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
            else { G.used[di]-=G.exp_bytes; arena_give(di,idx); }
        } else {
            /* Unreachable while the budget invariant holds; never silent. */
            fprintf(stderr,"[q38tier] dev %d: arena exhausted with used %.2f/%.2f GB"
                           " -- offer dropped\n",
                    G.dev[di], G.used[di]/1073741824.0, G.budget[di]/1073741824.0);
        }
        pthread_mutex_unlock(&G.mx); goto out;
    }

    /* Card full: entry is only by bumping the coldest resident of this same
     * card, and only if the shared contract (tier.h, with hysteresis)
     * authorizes it. The scan is linear over nl*ne slots: at one miss per
     * expert and a token of over a second it is noise, and keeping an ordered
     * structure would cost more complexity than it would save. */
    {
        size_t n=(size_t)G.nl*G.ne;
        long cold=-1; uint32_t ch=0;
        for(size_t i=0;i<n;i++){
            Q38TSlot *c=&G.slot[i];
            if(!c->resident||c->queued) continue;
            if(home((int)(i%G.ne))!=di) continue;
            if(cold<0||c->heat<ch){ cold=(long)i; ch=c->heat; }
        }
        if(cold>=0 && tier_should_promote(s->heat,ch)){
            Q38TSlot *v=&G.slot[cold];
            v->resident=0;                       /* from now on it is CPU fallback */
            if(enqueue_locked(layer,eid,(int)(cold/G.ne),(int)(cold%G.ne),
                              gate,up,down,scales)){
                G.swaps++;
                s->arena_idx=v->arena_idx;       /* the replacement inherits the
                                                    victim's slot; the uploader
                                                    frees the victim's tensors
                                                    before writing this slot */
                rel=cold;                        /* back to CPU fallback: relock */
            }
            else v->resident=1;                  /* queue full: put it back */
        }
    }

    pthread_mutex_unlock(&G.mx);
out:
    /* The victim just lost its VRAM slot: its host ranges must go back to
     * pinned, or a long session would slowly migrate the pinned set onto
     * whatever the hot swaps promote. Outside the mutex (I/O, see above). */
    if(rel>=0 && g_pin_lock) g_pin_lock((int)(rel/G.ne),(int)(rel%G.ne));
    return;
}
/* --- execution ------------------------------------------------------------ */

uint32_t q38t_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||!eids||!x||K<1||K>32||layer<0||layer>=G.nl) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    ColiCudaTensor *tu[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    ColiCudaTensor *td[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    static const int rows[Q38T_MAX_ROWS]={1,1,1,1,1,1,1,1};
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    G.issue_open=1;
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne){ G.miss++; continue; }
        Q38TSlot *s=qs(layer,e);
        if(!s->resident){ G.miss++; continue; }
        int di=home(e), c=G.is_cnt[di];
        /* The backend keeps a single issue in flight per device and refuses
         * more than Q38T_MAX_ROWS rows, so a card that sees more experts
         * arrive than the ceiling cannot split the issue: the extras stay on
         * the CPU via the mask. With topk=10 and two cards the eid%2 split
         * sends ~5 each way and the case does not present itself, but nothing
         * forbids it and the count tells (overflow_rows). */
        if(c>=Q38T_MAX_ROWS){ G.miss++; G.overflow_rows++; continue; }
        tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
        G.is_k[di][c]=k; G.is_cnt[di]=c+1;
        mask|=1u<<k; G.hits[di]++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*Q38T_MAX_ROWS*G.D;
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        uint64_t t0=now_ns();
        int ok=coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr);
        G.t_issue[di]+=now_ns()-t0; G.n_issue[di]++;
        if(!ok){
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    if(!mask){
        pthread_mutex_lock(&G.mx);
        G.issue_open=0;
        pthread_cond_broadcast(&G.cv_take);
        pthread_mutex_unlock(&G.mx);
    }
    return mask;
}

void q38t_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on) return;
    if(mask&&val&&out) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        uint64_t t0=now_ns();
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        G.t_take[di]+=now_ns()-t0; G.n_take[di]++;
        if(!y){
            /* These k were in the mask: the CPU skipped them and now nobody
             * computes them, i.e. the token comes out with a piece of the MoE
             * missing. It is not recoverable here (the weights in RAM have
             * already been evicted), but it must not pass silently. */
            G.take_fails++;
            if(G.take_fails==1)
                fprintf(stderr,"[q38tier] WARNING: expert_group_take failed on dev %d; "
                               "%d routed expert(s) dropped from this token\n",
                        G.dev[di], c);
            G.is_cnt[di]=0;
            continue;
        }
        uint64_t t1=now_ns();
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.t_acc+=now_ns()-t1;
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

/* --- prefill: synchronous grouped compute on the resident experts --------- */

/* Default rows per batch. See the VRAM arithmetic in q38t_expert_group(). */
#define Q38T_PREFILL_ROWS 4096

/* Grows the two staging buffers to `floats` each. They are kept for the whole
 * run: a prefill calls this 48 times per chunk and the steady size is reached
 * on the first layer. */
static int pf_reserve(size_t floats){
    if(floats<=G.pf_floats) return 1;
    float *nx=(float*)realloc(G.pf_x,floats*sizeof(float));
    if(!nx) return 0;
    G.pf_x=nx;
    float *ny=(float*)realloc(G.pf_y,floats*sizeof(float));
    if(!ny) return 0;                       /* pf_x kept: it is valid and larger */
    G.pf_y=ny; G.pf_floats=floats;
    return 1;
}

/* Runs one batch: `n` experts of the SAME device, indices bi[0..n) into the
 * caller's arrays. Gathers their rows into pf_x, calls the synchronous group,
 * scatters back into y. Returns the number of experts computed (0 = refused,
 * the caller leaves them to the CPU). */
static int pf_flush(int layer,const int *eids,const int *rows,const int *off,
                    int *bi,int n,int brows,
                    const float *x,float *y,uint8_t *done){
    if(n<1||brows<1) return 0;
    if(!pf_reserve((size_t)brows*G.D)){ G.pf_refused++; return 0; }

    ColiCudaTensor *tg[64],*tu[64],*td[64];
    int r[64];
    /* The tensors are read under the lock and issue_open is raised before
     * letting it go: from here until the flag drops the uploader cannot free
     * a victim (uploader(), "swap: the victim is freed only when no group is
     * in flight"). Same protocol as issue/take, synchronous instead of split. */
    pthread_mutex_lock(&G.mx);
    int c=0;
    for(int j=0;j<n;j++){
        Q38TSlot *s=qs(layer,eids[bi[j]]);
        if(!s->resident||!s->tg||!s->tu||!s->td){ G.pf_absent++; continue; }
        tg[c]=s->tg; tu[c]=s->tu; td[c]=s->td; r[c]=rows[bi[j]];
        /* bi[] is reused below to scatter, so compact it the same way */
        bi[c]=bi[j];
        c++;
    }
    if(c) G.issue_open=1;
    pthread_mutex_unlock(&G.mx);
    if(!c) return 0;

    int64_t gathered=0;
    for(int j=0;j<c;j++){
        memcpy(G.pf_x+gathered*G.D, x+(int64_t)off[bi[j]]*G.D,
               (size_t)r[j]*G.D*sizeof(float));
        gathered+=r[j];
    }
    uint64_t t0=now_ns();
    int ok=coli_cuda_expert_group(tg,tu,td,r,c,G.pf_y,G.pf_x);
    G.t_pf+=now_ns()-t0;

    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);

    if(!ok){ G.pf_refused++; return 0; }
    int64_t scattered=0;
    for(int j=0;j<c;j++){
        memcpy(y+(int64_t)off[bi[j]]*G.D, G.pf_y+scattered*G.D,
               (size_t)r[j]*G.D*sizeof(float));
        scattered+=r[j];
        done[bi[j]]=1;
    }
    G.pf_batches++; G.pf_experts+=(uint64_t)c; G.pf_rows+=(uint64_t)gathered;
    return c;
}

int q38t_expert_group(int layer,const int *eids,const int *rows,const int *off,
                      int count,const float *x,float *y,uint8_t *done){
    if(!G.on||!eids||!rows||!off||!x||!y||!done||count<1||layer<0||layer>=G.nl)
        return 0;
    int *bi=(int*)malloc((size_t)count*sizeof(*bi));
    if(!bi) return 0;

    /* Row budget of one batch. The ceiling that matters is VRAM: the backend
     * reserves rows*(2*D+2*I) floats of workspace per call, ~25 KB per row
     * with this geometry, so 4096 rows is ~105 MB on a card that is otherwise
     * full of experts. It is a knob because the right value depends on what
     * the budget left free, not on the model. */
    int budget=Q38T_PREFILL_ROWS;
    const char *env=getenv("Q38T_PREFILL_ROWS");
    if(env&&*env){ int v=atoi(env); if(v>0) budget=v; }

    int taken=0;
    G.pf_calls++;
    for(int di=0;di<G.ndev;di++){
        int n=0,brows=0;
        for(int c=0;c<count;c++){
            if(rows[c]<1||done[c]) continue;
            if(eids[c]<0||eids[c]>=G.ne||home(eids[c])!=di) continue;
            if(!qs(layer,eids[c])->resident) continue;   /* rechecked under lock */
            /* 64 experts is the backend's own ceiling (GroupDesc host[64],
             * backend_cuda.cu:1850); the row budget is ours. An expert whose
             * group alone exceeds the budget still goes through, on its own. */
            if(n==64||(n>0&&brows+rows[c]>budget)){
                taken+=pf_flush(layer,eids,rows,off,bi,n,brows,x,y,done);
                n=0; brows=0;
            }
            bi[n++]=c; brows+=rows[c];
        }
        if(n) taken+=pf_flush(layer,eids,rows,off,bi,n,brows,x,y,done);
    }
    free(bi);
    return taken;
}

/* --- warmstart ------------------------------------------------------------ */

static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : (ha>hb ? -1 : 0);
}

int q38t_plan_fill(int *layers,int *eids,int max){
    if(!G.on||!layers||!eids||max<1) return 0;
    if(getenv("Q38T_NO_WARMSTART")) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        /* Without HEAT_FILE there is no sensible order to invent: let the
         * traffic promote, which is already the normal way. */
        if(!G.heat0){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_order=malloc(n*sizeof(int));
        if(!G.fill_order){ pthread_mutex_unlock(&G.mx); return 0; }
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        g_sort_heat=G.heat0;
        qsort(G.fill_order,n,sizeof(int),cmp_heat_desc);
        G.fill_n=(int)n; G.fill_cur=0;
    }
    int out=0;
    while(out<max && G.fill_cur<G.fill_n){
        int i=G.fill_order[G.fill_cur++];
        int layer=i/G.ne, eid=i%G.ne, di=home(eid);
        Q38TSlot *s=&G.slot[i];
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[di]+G.exp_bytes>G.budget[di]) continue;
        uint32_t idx=arena_take(di);
        if(idx==UINT32_MAX) continue;         /* budget invariant broken: skip */
        G.used[di]+=G.exp_bytes;
        s->arena_idx=idx;
        s->planned=1;
        layers[out]=layer; eids[out]=eid; out++;
    }
    pthread_mutex_unlock(&G.mx);
    return out;
}

void q38t_cancel_plan(int layer,int eid){
    if(!G.on||layer<0||layer>=G.nl||eid<0||eid>=G.ne) return;
    pthread_mutex_lock(&G.mx);
    Q38TSlot *s=qs(layer,eid);
    if(s->planned&&!s->queued&&!s->resident){
        int di=home(eid);
        if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes;
        arena_give(di,s->arena_idx);
        s->planned=0;
        G.plan_cancels++;
    }
    pthread_mutex_unlock(&G.mx);
}

void q38t_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.inflight>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* --- shutdown and telemetry ----------------------------------------------- */

void q38t_stats(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    size_t n=(size_t)G.nl*G.ne;
    uint64_t res=0, hit=0;
    for(size_t i=0;i<n;i++) if(G.slot[i].resident) res++;
    fprintf(stderr,"[q38tier] resident %llu/%zu experts (%.1f%%)",
            (unsigned long long)res, n, n? 100.0*res/(double)n : 0.0);
    for(int i=0;i<G.ndev;i++){
        hit+=G.hits[i];
        fprintf(stderr," | dev%d hits %llu, %.1f/%.1f GB",
                G.dev[i],(unsigned long long)G.hits[i],
                G.used[i]/1073741824.0, G.budget[i]/1073741824.0);
    }
    fprintf(stderr,"\n[q38tier] gpu %llu, cpu %llu (%.1f%% on GPU) | offers %llu,"
                   " promotions %llu, swaps %llu, uploads %llu, failed %llu,"
                   " queue-full %llu, row-overflow %llu, take-fail %llu,"
                   " plan-cancels %llu\n",
            (unsigned long long)hit,(unsigned long long)G.miss,
            (hit+G.miss)? 100.0*hit/(double)(hit+G.miss) : 0.0,
            (unsigned long long)G.offers,(unsigned long long)G.promotions,
            (unsigned long long)G.swaps,(unsigned long long)G.uploads,
            (unsigned long long)G.upload_fail,
            (unsigned long long)G.q_full_skips,
            (unsigned long long)G.overflow_rows,
            (unsigned long long)G.take_fails,
            (unsigned long long)G.plan_cancels);
    /* Vedi il commento sui cronometri nella struttura G per come si leggono,
     * e in particolare perche' t_take di dev1 e' un eccesso e non un assoluto. */
    uint64_t t_is=0,t_tk=0;
    for(int i=0;i<G.ndev;i++){ t_is+=G.t_issue[i]; t_tk+=G.t_take[i]; }
    fprintf(stderr,"[q38tier] hot path: issue %.3f s, take %.3f s, accumulate %.3f s"
                   " (total %.3f s)\n",
            t_is/1e9, t_tk/1e9, G.t_acc/1e9, (t_is+t_tk+G.t_acc)/1e9);
    for(int i=0;i<G.ndev;i++)
        fprintf(stderr,"[q38tier]   dev%d: %llu issues %.3f s (%.1f us/call) |"
                       " %llu takes %.3f s (%.1f us/call)\n",
                G.dev[i],
                (unsigned long long)G.n_issue[i], G.t_issue[i]/1e9,
                G.n_issue[i]? G.t_issue[i]/1000.0/G.n_issue[i] : 0.0,
                (unsigned long long)G.n_take[i], G.t_take[i]/1e9,
                G.n_take[i]? G.t_take[i]/1000.0/G.n_take[i] : 0.0);
    if(G.pf_calls)
        fprintf(stderr,"[q38tier] prefill: %llu calls, %llu batches, %llu experts,"
                       " %llu rows, %.3f s on GPU | refused %llu, absent %llu\n",
                (unsigned long long)G.pf_calls,(unsigned long long)G.pf_batches,
                (unsigned long long)G.pf_experts,(unsigned long long)G.pf_rows,
                G.t_pf/1e9,
                (unsigned long long)G.pf_refused,(unsigned long long)G.pf_absent);
    pthread_mutex_unlock(&G.mx);
}

/* The heat learned in this session, for the next one. Same format as the
 * other tier (magic, layer, experts, then the matrix), so a file written here
 * is readable from here and that is enough: the geometries do not coincide
 * between the two engines and the header checks them before trusting. */
static void heat_save(void){
    const char *hf=getenv("HEAT_FILE");
    if(!hf) return;
    FILE *f=fopen(hf,"wb");
    if(!f){ fprintf(stderr,"[q38tier] cannot write HEAT_FILE %s\n",hf); return; }
    uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
    size_t n=(size_t)G.nl*G.ne;
    uint32_t *h=malloc(n*4);
    int ok=0;
    if(h){
        for(size_t i=0;i<n;i++) h[i]=G.slot[i].heat;
        ok = fwrite(hdr,4,3,f)==3 && fwrite(h,4,n,f)==n;
        free(h);
    }
    if(fclose(f)!=0) ok=0;
    if(!ok) fprintf(stderr,"[q38tier] HEAT_FILE %s written incompletely\n",hf);
}

void q38t_shutdown(void){
    if(!G.on) return;
    heat_save();
    pthread_mutex_lock(&G.mx);
    G.th_stop=1;
    pthread_cond_broadcast(&G.cv);
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    /* What remains in the queue was never loaded: free the staging. */
    while(G.qn>0){ free(G.q[G.qh].w); free(G.q[G.qh].s);
                   G.qh=(G.qh+1)%Q38T_QCAP; G.qn--; }
    size_t n=(size_t)G.nl*G.ne;
    for(size_t i=0;i<n;i++){
        Q38TSlot *s=&G.slot[i];
        if(s->tg) coli_cuda_tensor_free(s->tg);
        if(s->tu) coli_cuda_tensor_free(s->tu);
        if(s->td) coli_cuda_tensor_free(s->td);
        s->tg=s->tu=s->td=NULL; s->resident=0;
    }
    free(G.slot); free(G.is_x); free(G.pf_x); free(G.pf_y); G.pf_x=G.pf_y=NULL; G.pf_floats=0; free(G.fill_order); free(G.heat0);
    G.slot=NULL; G.is_x=NULL; G.fill_order=NULL; G.heat0=NULL;
    /* The tensors own nothing, so the arena dies after them, whole. */
    for(int i=0;i<G.ndev;i++){
        if(G.arena[i]) coli_cuda_pipe_free(G.dev[i],G.arena[i]);
        free(G.slot_free[i]);
        G.arena[i]=NULL; G.slot_free[i]=NULL; G.slot_free_n[i]=0;
    }
    G.on=0;
}

/* ---- dense BF16 matmul on the GPU (Q38_DENSE_GPU=1) ------------------------
 *
 * WHY THIS EXISTS, measured rather than assumed: reading the checkpoint's own
 * shapes and dtypes, one decode token moves 9.41 GiB, of which 8.17 GiB (86.9%)
 * is the DENSE resident set -- DeltaNet in_proj 34.6%, the gated residual 15.0%,
 * attention q/k/v/o 14.8%, lm_head 14.5%, DeltaNet out_proj 12.9%, the shared
 * expert 5.5% -- and only 13.1% is routed experts. At the measured 2.83 s/token
 * that is 3.57 GB/s effective, against ~50 GB/s of host RAM and ~900 GB/s of
 * VRAM sitting idle. The expert tier cannot reach any of it: every dense weight
 * is bf16 and the backend had no bf16 format, so every dense matmul stayed on
 * the CPU by construction. fmt=9 (backend_cuda.cu) removes that, and this is
 * the engine-side door to it.
 *
 * Deliberately independent of the expert tier's G state: the tier refuses to
 * attach for reasons that say nothing about the dense side (a second model in
 * the process, an expert format it does not stage), and the dense set is worth
 * VRAM regardless. The only shared resource is the backend's device context,
 * which is why this NEVER calls coli_cuda_init when one already exists --
 * coli_cuda_init resets g_nctx and would orphan every resident expert tensor. */

typedef struct {
    ColiCudaTensor *t;
    int device;
    int refused;            /* sticky: never re-attempt a failed 1 GiB upload */
    int counted;            /* its bytes are already in DG.bytes */
} Q38TDense;

static struct {
    int checked, wanted;      /* env gate, resolved once */
    int attached;             /* 0 = not tried, 1 = context adopted, -1 = refused */
    int ndev, dev[Q38T_MAX_DEV];
    int rank[Q38T_MAX_DEV];   /* dev ids, strongest card first (see dense_rank) */
    int pin;                  /* Q38_DENSE_DEV: pin every dense tensor here, -1 = auto */
    uint64_t calls, rows, refusals;
    size_t bytes;
} DG;

/* The env gate alone: cheap, touches no CUDA, and safe to call during the
 * model load. Device discovery is deliberately NOT done here. q38t_init runs
 * AFTER the weights are loaded and calls coli_cuda_init, which resets g_nctx --
 * so a context built at load time would be silently discarded under us. The
 * attach below therefore happens on first use, by which point the expert tier
 * has already built (or declined to build) the context we should be sharing. */
int q38t_dense_enabled(void){
    if(!DG.checked){
        DG.checked=1;
        const char *e=getenv("Q38_DENSE_GPU");
        DG.wanted=(e && *e=='1');
    }
    return DG.wanted;
}

/* Order the devices strongest-first for compute-heavy work.
 *
 * The cards in this machine are NOT interchangeable and the placement must say
 * so: dev1 is an RTX 5070 Ti wired PCIe x16, dev0 an RTX 5060 Ti in a slot that
 * is electrically x4 -- the card itself would do x8, but the copper is x4 and
 * the link trains there permanently. That is ~2x the SMs and ~4x the upload
 * bandwidth on one side. The previous policy picked the card with the
 * most free VRAM, which on two nearly-full cards is close to a coin toss -- and
 * a coin toss puts half the dense GEMMs on the weak, narrow card.
 *
 * SM count is the primary key because dense matmul time scales with it; the
 * PCIe width -- negotiated, not advertised, see coli_cuda_device_profile --
 * breaks ties and stands in for the cost of getting a multi-GiB resident tensor
 * onto the card in the first place. Nothing here is specific to
 * these two parts: a machine with two identical cards ranks them by device id
 * and behaves exactly as before. Q38_DENSE_DEV overrides the whole thing. */
static void dense_rank(void){
    int sm[Q38T_MAX_DEV]={0}, pw[Q38T_MAX_DEV]={0};
    for(int i=0;i<DG.ndev;i++){
        DG.rank[i]=DG.dev[i];
        coli_cuda_device_profile(DG.dev[i],&sm[i],&pw[i]);
    }
    /* Insertion sort on (sm, pcie width, -dev): ndev is 2 in practice and the
     * order must be total and stable so the log is reproducible run to run. */
    for(int i=1;i<DG.ndev;i++){
        int d=DG.rank[i], a=sm[i], b=pw[i], j=i-1;
        while(j>=0){
            int aj=0,bj=0;
            for(int k=0;k<DG.ndev;k++) if(DG.dev[k]==DG.rank[j]){ aj=sm[k]; bj=pw[k]; break; }
            if(aj>a || (aj==a && bj>b) || (aj==a && bj==b && DG.rank[j]<d)) break;
            DG.rank[j+1]=DG.rank[j]; j--;
        }
        DG.rank[j+1]=d;
    }
    for(int i=0;i<DG.ndev;i++){
        int a=0,b=0;
        for(int k=0;k<DG.ndev;k++) if(DG.dev[k]==DG.rank[i]){ a=sm[k]; b=pw[k]; break; }
        char nm[64]={0};
        coli_cuda_device_name(DG.rank[i],nm,sizeof nm);
        fprintf(stderr,"[q38dense] rank %d: dev %d (%s, %d SM, PCIe x%d)%s\n",
                i,DG.rank[i],nm[0]?nm:"unknown",a,b,i?"":"  <- preferred for dense");
    }
}

/* First use: adopt the existing device context if there is one, bootstrap our
 * own only if nobody has. Never re-init an existing context. */
static int dense_attach(void){
    if(DG.attached) return DG.attached>0;
    if(!q38t_dense_enabled()){ DG.attached=-1; return 0; }
    DG.attached=-1;

    int have=coli_cuda_device_count();
    if(have>0){
        for(int i=0;i<have && DG.ndev<Q38T_MAX_DEV;i++){
            int d=coli_cuda_device_at(i);
            if(d>=0) DG.dev[DG.ndev++]=d;
        }
    } else {
        const char *gl=getenv("COLI_GPUS");
        if(!gl || !*gl) gl=getenv("COLI_GPU");
        if(gl && *gl){
            char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
            for(char *t=strtok(buf,","); t && DG.ndev<Q38T_MAX_DEV; t=strtok(NULL,","))
                DG.dev[DG.ndev++]=atoi(t);
        } else {
            int available=coli_cuda_available_device_count();
            int want=available<2?available:2;
            for(int i=0;i<want && i<Q38T_MAX_DEV;i++) DG.dev[DG.ndev++]=i;
        }
        if(DG.ndev<1){
            fprintf(stderr,"[q38dense] no visible CUDA devices -> CPU path\n");
            return 0;
        }
        q38t_affmask aff; q38t_aff_get(&aff); q38t_aff_widen(&aff);
        if(!coli_cuda_init(DG.dev,DG.ndev)){
            fprintf(stderr,"[q38dense] coli_cuda_init failed -> CPU path\n");
            DG.ndev=0; return 0;
        }
        DG.ndev=coli_cuda_device_count();
    }
    if(DG.ndev<1) return 0;
    DG.attached=1;
    fprintf(stderr,"[q38dense] bf16 dense matmuls on GPU, %d device(s)\n",DG.ndev);
    dense_rank();
    DG.pin=-1;
    {   const char *p=getenv("Q38_DENSE_DEV");
        if(p && *p){
            int want=atoi(p);
            for(int i=0;i<DG.ndev;i++) if(DG.dev[i]==want){ DG.pin=want; break; }
            if(DG.pin<0)
                fprintf(stderr,"[q38dense] Q38_DENSE_DEV=%s is not one of the "
                               "attached devices, ignored\n",p);
            else
                fprintf(stderr,"[q38dense] pinned to dev %d by Q38_DENSE_DEV\n",DG.pin);
        }
    }
    return 1;
}

/* Strongest-card-first placement: walk the ranking and take the first card the
 * tensor actually fits on. This deliberately replaces the old least-loaded rule
 * (most free VRAM wins), which spread the dense set evenly over two cards that
 * are not evenly capable -- see dense_rank. Filling the strong card before
 * touching the weak one is the point, not a side effect: the overflow onto the
 * second card is a fallback, and the stats say how much of it happened.
 *
 * The margin leaves the backend room for its activation scratch. 256 MiB was
 * measured against the LAZY door, which throttles itself mid-forward and never
 * spends a reserve down to the last byte; EAGER staging does spend it, and the
 * first asymmetric run (reserve 1024,6144) ate the list to the margin and then
 * printed 87x "scratch allocation: out of memory" -- the weights were resident
 * but nothing could compute. So the cushion is now Q38_DENSE_MARGIN_MB,
 * default 1024, applied to eager AND lazy alike: the two paths must not
 * disagree about the budget, or a weight that eager refused legitimately
 * reappears as a mid-forward stall. The cost is honest: with an asymmetric
 * list the weak card's small reserve stops being dense overflow at all and
 * stays arena/expert ground, which is where it was always meant to live. */
static size_t dense_margin(void){
    static size_t cached=0;
    if(!cached){
        const char *e=getenv("Q38_DENSE_MARGIN_MB");
        int mb=e&&*e?atoi(e):1024;
        cached=(size_t)(mb<0?0:mb)*1048576;
    }
    return cached;
}

static int dense_pick_device(size_t need){
    size_t margin=dense_margin();
    if(DG.pin>=0){
        size_t fr=0,tot=0;
        if(coli_cuda_mem_info(DG.pin,&fr,&tot) && fr>=need+margin) return DG.pin;
        return -1;   /* pinned means pinned: no silent spill to the other card */
    }
    for(int i=0;i<DG.ndev;i++){
        size_t fr=0,tot=0;
        if(!coli_cuda_mem_info(DG.rank[i],&fr,&tot)) continue;
        if(fr>=need+margin) return DG.rank[i];
    }
    return -1;
}

int q38t_dense_matmul(void **slot, float *y, const float *x, const uint16_t *w,
                      int S, int I, int O){
    if(!slot || !y || !x || !w || S<1 || I<1 || O<1) return 0;
    if(!dense_attach()) return 0;

    Q38TDense *d=(Q38TDense*)*slot;
    if(!d){
        d=(Q38TDense*)calloc(1,sizeof *d);
        if(!d) return 0;
        d->device=-1;
        *slot=d;
    }
    if(d->refused) return 0;

    if(!d->t){
        size_t need=(size_t)I*(size_t)O*2;
        int dev=dense_pick_device(need);
        if(dev<0){
            fprintf(stderr,"[q38dense] [%d,%d] bf16 (%.2f GiB) does not fit -> CPU "
                           "for this weight\n",O,I,need/1073741824.0);
            d->refused=1; DG.refusals++; return 0;
        }
        d->device=dev;
    }
    /* fmt=9, scales NULL, gs 0: the first call uploads and the device copy is
     * reused from then on. A failure here is also sticky -- if the upload could
     * not be made once, retrying it per token only buys a slower CPU path. */
    if(!coli_cuda_matmul(&d->t, y, x, w, NULL, 9, S, I, O, d->device, 0)){
        if(!d->t){
            fprintf(stderr,"[q38dense] upload of [%d,%d] bf16 failed -> CPU for "
                           "this weight\n",O,I);
            d->refused=1; DG.refusals++;
        }
        return 0;
    }
    if(!d->counted && d->t){       /* charge the residency once, at first success */
        DG.bytes += coli_cuda_tensor_bytes(d->t);
        d->counted=1;
        fprintf(stderr,"[q38dense] resident [%d,%d] bf16 on dev%d, %.2f GiB total\n",
                O,I,d->device,DG.bytes/1073741824.0);
    }
    DG.calls++; DG.rows+=(uint64_t)S;
    return 1;
}

/* Eager staging: upload a gated weight to VRAM WITHOUT executing anything, so
 * capacity failures surface at startup (where the log reads them) instead of
 * as stalls inside the first forwards. Shares the slot, the sticky refusal
 * and the accounting with the lazy path: a staged weight makes
 * q38t_dense_matmul's upload branch a no-op, and a refused one falls to CPU
 * exactly as it always did. */
int q38t_dense_stage(void **slot, const uint16_t *w, int I, int O){
    if(!slot || !w || I<1 || O<1) return 0;
    if(!dense_attach()) return 0;

    Q38TDense *d=(Q38TDense*)*slot;
    if(!d){
        d=(Q38TDense*)calloc(1,sizeof *d);
        if(!d) return 0;
        d->device=-1;
        *slot=d;
    }
    if(d->refused) return 0;
    if(d->t) return 1;

    size_t need=(size_t)I*(size_t)O*2;
    int dev=dense_pick_device(need);
    if(dev<0){
        fprintf(stderr,"[q38dense] eager: [%d,%d] bf16 (%.2f GiB) does not fit -> "
                       "CPU for this weight\n",O,I,need/1073741824.0);
        d->refused=1; DG.refusals++; return 0;
    }
    d->device=dev;
    if(!coli_cuda_tensor_upload_g(&d->t,w,NULL,9,I,O,dev,0)){
        fprintf(stderr,"[q38dense] eager: upload of [%d,%d] bf16 failed -> CPU for "
                       "this weight\n",O,I);
        if(!d->t) d->device=-1;
        d->refused=1; DG.refusals++; return 0;
    }
    if(!d->counted && d->t){
        DG.bytes += coli_cuda_tensor_bytes(d->t);
        d->counted=1;
    }
    return 1;
}

void q38t_dense_release(void **slot){
    if(!slot||!*slot) return;
    Q38TDense *d=(Q38TDense*)*slot;
    if(d->t) coli_cuda_tensor_free(d->t);
    free(d);
    *slot=NULL;
}

uint64_t q38t_dense_bytes(void){ return (uint64_t)DG.bytes; }

void q38t_dense_stats(void){
    if(DG.attached<=0) return;
    fprintf(stderr,"[q38dense] calls=%llu rows=%llu resident=%.2f GiB refused=%llu\n",
            (unsigned long long)DG.calls,(unsigned long long)DG.rows,
            DG.bytes/1073741824.0,(unsigned long long)DG.refusals);
    /* The q38 engine never printed the TC_W4A16 routing counter, so the A/B
     * with the flag on was blind: "no delta" could not be told apart from
     * "never fired". This closes that gap for good. */
    uint64_t calls=0,experts=0,erows=0,tcrows=0; double h2d=0,k=0,d2h=0;
    coli_cuda_group_stats(&calls,&experts,&erows,&h2d,&k,&d2h);
    coli_cuda_tc_w4a16_rows(&tcrows);
    fprintf(stderr,"[q38dense] expert-groups: %llu calls %llu rows, TC_W4A16 %llu/%llu (%.1f%%)\n",
            (unsigned long long)calls,(unsigned long long)erows,
            (unsigned long long)tcrows,(unsigned long long)erows,
            erows?100.0*(double)tcrows/(double)erows:0.0);
    /* The oracle's answer did not carry the ~480 us/call the engine pays at
     * S=1, and none of its candidate explanations survived measurement:
     * decode clocks run hot, host CPU saturation adds 1.5 us/call, and the
     * dense stream is legacy-default while the tier's is nonblocking, so no
     * queued work can hide inside the call either. This line is where that
     * residue gets a shape: submission (launch return), execution (kernel
     * event span), or host download. */
    uint64_t tn=0; double wall[4]={0,0,0,0},mev[3]={0,0,0},mx[3]={0,0,0};
    coli_cuda_mm_trace(&tn,wall,mev,mx);
    if(tn)fprintf(stderr,"[q38dense] mm trace S=1 %llu calls: up %.1f h2d %.1f launch %.1f down %.1f us/call"
                         " | dev h2d %.1f kern %.1f d2h %.1f | max launch %.0f down %.0f kern %.0f us\n",
            (unsigned long long)tn,wall[0],wall[1],wall[2],wall[3],
            mev[0],mev[1],mev[2],mx[0],mx[1],mx[2]);
}

#endif /* COLI_CUDA */
