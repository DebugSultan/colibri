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

/* Reserve in bytes: Q38T_DEV_RESERVE_MB (MiB), default Q38T_DEV_RESERVE.
 * Read on every q38t_init -- once per process in production -- so tests can
 * flip the env between inits without a reset hook. */
static size_t dev_reserve(void){
    const char *m=getenv("Q38T_DEV_RESERVE_MB");
    double mb=(m && *m) ? atof(m) : 0.0;
    if(mb>0) return (size_t)(mb*1024.0*1024.0);
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
            if(a)coli_cuda_tensor_free(a);
            if(b)coli_cuda_tensor_free(b);
            if(c)coli_cuda_tensor_free(c);
        } else pthread_mutex_unlock(&G.mx);

        int dv=G.dev[home(eid)];
        /* gate/up are [inter,hidden], down is [hidden,inter]; the signature
         * wants (I=input, O=output), not (rows, columns). The _g variant is
         * used for both formats: with gs=0 it is the plain upload, and fmt=4
         * needs gs to derive ng=(I+gs-1)/gs and the scale count. */
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok = coli_cuda_tensor_upload_g(&tg, w,               sc,        G.fmt, G.D,  G.Ih, dv, G.gs)
              && coli_cuda_tensor_upload_g(&tu, w+G.mat_bytes,   sc+G.sc,   G.fmt, G.D,  G.Ih, dv, G.gs)
              && coli_cuda_tensor_upload_g(&td, w+2*G.mat_bytes, sc+2*G.sc, G.fmt, G.Ih, G.D,  dv, G.gs);
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        Q38TSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid);
                G.upload_fail++;
                if(G.used[hd]>=G.exp_bytes) G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* card really full: stop */
                if(tg)coli_cuda_tensor_free(tg);
                if(tu)coli_cuda_tensor_free(tu);
                if(td)coli_cuda_tensor_free(td); }
        s->queued=0; s->planned=0;
        G.inflight--;
        pthread_cond_broadcast(&G.cv_take);          /* this upload is complete */
        pthread_mutex_unlock(&G.mx);
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

/* VRAM an allocation of `bytes` really occupies (cudaMalloc granularity,
 * see the exp_bytes computation in q38t_init). */
static size_t dev_alloc_footprint(size_t bytes){
    /* measured with cudaMemGetInfo over 256 allocations each (driver 5xx):
     *   400 B, 3 KiB, 4 KiB -> 8 KiB      10 KiB -> 16 KiB     16..64 KiB -> exact
     *   96 KiB -> 104 KiB   384 KiB -> 416 KiB   768 KiB -> 1 MiB   1 MiB -> 1 MiB
     *   1.5 MiB -> 2 MiB    3 MiB -> 4 MiB
     * i.e. above 1 MiB multiples of 2 MiB, above 512 KiB one 1 MiB page, and
     * below that roughly the size plus a sixteenth, in 8 KiB steps, 8 KiB
     * minimum. The small-size rule is a fit, slightly conservative. */
    const size_t KiB = 1024u, MiB = 1048576u;
    if(bytes > MiB) return (bytes + 2*MiB - 1) / (2*MiB) * (2*MiB);
    if(bytes > 512*KiB) return MiB;
    size_t b = bytes + bytes/16;
    if(b < 8*KiB) b = 8*KiB;
    return (b + 8*KiB - 1) / (8*KiB) * (8*KiB);
}

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

    /* Charge what the device allocator takes, not what the bytes measure:
     * cudaMalloc rounds an allocation above 1 MiB up to a multiple of 2 MiB,
     * one above 512 KiB up to 1 MiB, and small ones to 8 KiB steps
     * (dev_alloc_footprint has the measured table).
     * An expert is three weight allocations plus three scale allocations,
     * whichever the format: fmt=4 halves the weight side and multiplies the
     * scale side by 128*128/gs, which for gs=64 is a net ~2x more experts
     * resident per card.
     * Charged by payload, the fp8 qwen38 expert (3 x 1.56 MiB) looked like
     * 4.69 MiB and took 6.03 MiB: the budget over-committed by ~28 % and the
     * first "tensor allocation: out of memory" froze it permanently via the
     * clamp (G.budget[hd]=G.used[hd] in the uploader). Now the planned count
     * is the resident count. The 22-28 % the granularity costs is real; only
     * pooling experts into one arena per device would win it back (open). */
    G.exp_bytes = 3*dev_alloc_footprint(G.mat_bytes) + 3*dev_alloc_footprint(G.sc*sizeof(float));
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
    size_t reserve=dev_reserve();
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
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
    G.slot=calloc((size_t)nl*ne,sizeof(Q38TSlot));
    G.is_x_floats=(size_t)G.ndev*Q38T_MAX_ROWS*D;
    G.is_x=malloc(G.is_x_floats*sizeof(float));
    if(!G.slot||!G.is_x){ free(G.slot); free(G.is_x); return 0; }

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
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0){ free(G.slot); free(G.is_x); return 0; }
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
                    s->planned=0;
                }
                pthread_mutex_unlock(&G.mx); goto out;
            }
            if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
            else { if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes; s->planned=0; }
            pthread_mutex_unlock(&G.mx); goto out;
        }
    }

    if(G.used[di]+G.exp_bytes<=G.budget[di]){
        G.used[di]+=G.exp_bytes;
        if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
        else G.used[di]-=G.exp_bytes;
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
                              gate,up,down,scales)) G.swaps++;
            else v->resident=1;                  /* queue full: put it back */
        }
    }
    pthread_mutex_unlock(&G.mx);
out:
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
        G.used[di]+=G.exp_bytes;
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
    G.on=0;
}

#endif /* COLI_CUDA */
