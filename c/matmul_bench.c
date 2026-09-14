/* matmul_bench -- standalone microbenchmark for the CPU weight-matmul kernels.
 *
 * Builds and runs on its own; it links nothing from the engine and changes
 * nothing in it. It exists to separate two defects that the engine's timer
 * bank cannot tell apart, because both land in the same phase counter:
 *
 *   (1) the kernels are scalar. matmul_fp8 and q38_matmul_bf16 compile to
 *       vmulss/vaddss with no packed instruction in the loop, and the build
 *       carries no -ffast-math, so the float reduction cannot auto-vectorize
 *       and the accumulator is one serial dependency chain. Remedy: SIMD.
 *   (2) the s loop sits INSIDE the o loop, so every weight row is loaded and
 *       decoded once per row of activations. Remedy: structural -- decode the
 *       block once and reuse it across the rows.
 *
 * At S=1 (decode) only (1) exists. At S=12 (prefill with the widened chunk)
 * both do, and they have to be weighed apart or the SIMD variant gets credited
 * with a gain that is really reuse. Hence V1, which vectorizes but keeps the
 * original loop order, sitting between V0 and V2.
 *
 * The weights are drawn from a pool larger than L3 (16 MiB per CCX on Zen 2).
 * A microbenchmark over a single matrix would measure the cache rather than
 * DRAM and would flatter every variant at once.
 *
 * Shapes come from the model config: hidden 2560, routed expert 2560->640
 * (gate, up) and 640->2560 (down), 10 of 512 experts per token in all 48
 * layers. Gated DeltaNet has 16 QK heads and 48 V heads of width 128, hence
 * the resident projections 2560->6144 and 6144->2560.
 *
 * S values come from measured routing counters, not from the row count: one
 * prefill chunk of 274 rows spreads 2740 row-to-expert assignments over 233
 * distinct experts per layer, so an expert sees about 12 rows, not 274. Only
 * the resident matmuls see the whole prompt at once.
 *
 * Build:  make bench-matmul        (x86-64; the vector paths need AVX2+FMA)
 * Run:    OMP_NUM_THREADS=20 ./matmul_bench [seconds-per-measurement]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_VEC 1
#else
#define HAVE_VEC 0
#endif

#define FP8_BLOCK 128
static inline int64_t fp8_nblk(int n){ return ((int64_t)n + FP8_BLOCK - 1) / FP8_BLOCK; }

/* Same table as quant.h, rebuilt at startup from the OCP E4M3-FN definition so
 * this file stays self-contained. The vector decoder below is checked against
 * it over all 256 bytes before anything is measured. */
static float E4M3_LUT[256];
static void lut_init(void){
    for(int b=0;b<256;b++){
        int sgn=(b>>7)&1, e=(b>>3)&0xF, m=b&0x7;
        float v;
        if(e==0xF && m==0x7) v=NAN;           /* the only NaN encoding in E4M3-FN */
        else if(e==0) v=ldexpf((float)m,-9);  /* subnormal: m*2^-9 */
        else v=ldexpf(1.0f+(float)m/8.0f,e-7);
        E4M3_LUT[b]= sgn? -v : v;
    }
}
static inline float e4m3_decode(uint8_t b){ return E4M3_LUT[b]; }

/* ---- V0: scalar reference, a verbatim copy of quant.h matmul_fp8 --------- */
static void matmul_fp8_v0(float *y,const float *x,const uint8_t *q8,const float *bscale,
                          int S,int I,int O){
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+(int64_t)(o/FP8_BLOCK)*nblkI;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float sc=scl[bi]; float acc=0;
                for(int i=base;i<base+blen;i++) acc+=e4m3_decode(w[i])*xs[i];
                a+=(double)acc*sc;
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}

/* Exact double-precision reference. Comparing only against V0 would say "V2
 * differs", never "V2 is wrong"; against this one it can say which variant is
 * further from the mathematics. */
static void matmul_fp8_exact(double *y,const float *x,const uint8_t *q8,const float *bscale,
                             int S,int I,int O){
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+(int64_t)(o/FP8_BLOCK)*nblkI;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                double acc=0;
                for(int i=base;i<base+blen;i++) acc+=(double)e4m3_decode(w[i])*(double)xs[i];
                a+=acc*(double)scl[bi];
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

static inline float bf16_to_f32(uint16_t h){ union{uint32_t u;float f;}c; c.u=(uint32_t)h<<16; return c.f; }

/* ---- V0 bf16: verbatim copy of qwen38_core.h q38_matmul_bf16 ------------ */
static void matmul_bf16_v0(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0.f;
            for(int i=0;i<I;i++) a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}

#if HAVE_VEC
/* ---- vector E4M3 decode --------------------------------------------------
 * An e4m3 byte orders sign/exp/mant the same way f32 does: placing exp at bits
 * 23-26 and mant at bits 20-22 yields 2^(e-127)*(1+m/8), which is the wanted
 * value scaled by 2^-120. The 2^120 correction is a power of two, so the
 * multiply is exact, and e4m3 subnormals (exp==0) land correctly with no
 * branch of their own. The one case the bits do not reproduce is NaN: E4M3-FN
 * reserves it for mant==7 at exp==15, where the bit pattern alone would give a
 * finite number -- hence the blend. The NaN must survive: propagating it is
 * declared, tested policy (see quant.h and tests/test_logit_nan.c), not an
 * implementation detail. */
static inline __m256 e4m3_dec8_bits(const uint8_t *p){
    __m256i b   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));
    __m256i mag = _mm256_and_si256(b,_mm256_set1_epi32(0x7F));
    __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(b,_mm256_set1_epi32(0x80)),24);
    __m256i bits= _mm256_or_si256(sgn,_mm256_slli_epi32(mag,20));
    __m256  f   = _mm256_mul_ps(_mm256_castsi256_ps(bits),_mm256_set1_ps(0x1p120f));
    __m256i nan = _mm256_cmpeq_epi32(mag,_mm256_set1_epi32(0x7F));
    return _mm256_blendv_ps(f,_mm256_set1_ps(NAN),_mm256_castsi256_ps(nan));
}
/* The obvious alternative: gather straight from the scalar LUT. Zen 2
 * microcodes vpgatherdd, but here the cost is amortized over the rows, so it
 * is measured rather than assumed. */
static inline __m256 e4m3_dec8_gather(const uint8_t *p){
    __m256i b=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));
    return _mm256_i32gather_ps(E4M3_LUT,b,4);
}
static inline __m256 bf16_dec8(const uint16_t *p){
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)),16));
}
static inline float hsum8(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi);
    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
    lo=_mm_add_ss(lo,_mm_shuffle_ps(lo,lo,0x55));
    return _mm_cvtss_f32(lo);
}

/* ---- V1: SIMD only, loop order unchanged (no reuse across rows) ---------
 * Isolates plain vectorization: the weight row is still decoded once per row
 * of activations, exactly as in V0. The V1->V2 gap is therefore the price of
 * the loop structure, not of the instruction set. Four independent
 * accumulators break the serial dependency chain, which is also why the
 * vector variants come out closer to the exact result than V0 does. */
static void matmul_fp8_v1(float *y,const float *x,const uint8_t *q8,const float *bscale,
                          int S,int I,int O){
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+(int64_t)(o/FP8_BLOCK)*nblkI;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
                int k=0;
                for(;k+32<=blen;k+=32){
                    a0=_mm256_fmadd_ps(e4m3_dec8_bits(w+base+k   ),_mm256_loadu_ps(xs+base+k   ),a0);
                    a1=_mm256_fmadd_ps(e4m3_dec8_bits(w+base+k+ 8),_mm256_loadu_ps(xs+base+k+ 8),a1);
                    a2=_mm256_fmadd_ps(e4m3_dec8_bits(w+base+k+16),_mm256_loadu_ps(xs+base+k+16),a2);
                    a3=_mm256_fmadd_ps(e4m3_dec8_bits(w+base+k+24),_mm256_loadu_ps(xs+base+k+24),a3);
                }
                float acc=hsum8(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
                for(;k<blen;k++) acc+=e4m3_decode(w[base+k])*xs[base+k];
                a+=(double)acc*scl[bi];
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}

/* ---- V2/V3: SIMD plus reuse across rows ---------------------------------
 * A 128-weight block is decoded once and serves every row in the group; the
 * per-row accumulators stay live across blocks. Summation order matches V1,
 * so the numerical result is identical, but the decode cost is divided by the
 * number of rows.
 * Rows are processed in groups of SBLK so that the live activations stay in
 * L1 and the stack frame does not depend on S: 64*128*4 = 32 KiB, the L1D of
 * Zen 2. decoder=0 selects the bit trick, decoder=1 the gather. */
#define SBLK 64
static void matmul_fp8_reuse(float *y,const float *x,const uint8_t *q8,const float *bscale,
                             int S,int I,int O,int decoder){
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+(int64_t)(o/FP8_BLOCK)*nblkI;
        float wd[FP8_BLOCK] __attribute__((aligned(32)));
        double a[SBLK];
        for(int s0=0;s0<S;s0+=SBLK){
            int ns=S-s0; if(ns>SBLK) ns=SBLK;
            for(int s=0;s<ns;s++) a[s]=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                int k=0;
                if(decoder==0) for(;k+8<=blen;k+=8) _mm256_store_ps(wd+k,e4m3_dec8_bits(w+base+k));
                else           for(;k+8<=blen;k+=8) _mm256_store_ps(wd+k,e4m3_dec8_gather(w+base+k));
                for(;k<blen;k++) wd[k]=e4m3_decode(w[base+k]);
                float sc=scl[bi];
                for(int s=0;s<ns;s++){
                    const float *xs=x+(int64_t)(s0+s)*I+base;
                    __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
                    int j=0;
                    for(;j+32<=blen;j+=32){
                        a0=_mm256_fmadd_ps(_mm256_load_ps(wd+j   ),_mm256_loadu_ps(xs+j   ),a0);
                        a1=_mm256_fmadd_ps(_mm256_load_ps(wd+j+ 8),_mm256_loadu_ps(xs+j+ 8),a1);
                        a2=_mm256_fmadd_ps(_mm256_load_ps(wd+j+16),_mm256_loadu_ps(xs+j+16),a2);
                        a3=_mm256_fmadd_ps(_mm256_load_ps(wd+j+24),_mm256_loadu_ps(xs+j+24),a3);
                    }
                    float acc=hsum8(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
                    for(;j<blen;j++) acc+=wd[j]*xs[j];
                    a[s]+=(double)acc*sc;
                }
            }
            for(int s=0;s<ns;s++) y[(int64_t)(s0+s)*O+o]=(float)a[s];
        }
    }
}
static void matmul_fp8_v2(float *y,const float *x,const uint8_t *q,const float *b,int S,int I,int O){ matmul_fp8_reuse(y,x,q,b,S,I,O,0); }
static void matmul_fp8_v3(float *y,const float *x,const uint8_t *q,const float *b,int S,int I,int O){ matmul_fp8_reuse(y,x,q,b,S,I,O,1); }

static void matmul_bf16_v1(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0; int i=0;
            for(;i+32<=I;i+=32){
                a0=_mm256_fmadd_ps(bf16_dec8(w+i   ),_mm256_loadu_ps(xs+i   ),a0);
                a1=_mm256_fmadd_ps(bf16_dec8(w+i+ 8),_mm256_loadu_ps(xs+i+ 8),a1);
                a2=_mm256_fmadd_ps(bf16_dec8(w+i+16),_mm256_loadu_ps(xs+i+16),a2);
                a3=_mm256_fmadd_ps(bf16_dec8(w+i+24),_mm256_loadu_ps(xs+i+24),a3);
            }
            float acc=hsum8(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
            for(;i<I;i++) acc+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=acc;
        }
    }
}
/* Blocked along I by 128 as in the fp8 kernel, so the decode can be reused
 * across rows without a buffer as large as I; accumulation stays float over
 * the whole of I as in the original, so only the summation order changes. */
static void matmul_bf16_v2(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        float wd[128] __attribute__((aligned(32)));
        float a[SBLK];
        for(int s0=0;s0<S;s0+=SBLK){
            int ns=S-s0; if(ns>SBLK) ns=SBLK;
            for(int s=0;s<ns;s++) a[s]=0.f;
            for(int base=0;base<I;base+=128){
                int blen=128; if(base+blen>I) blen=I-base;
                int k=0;
                for(;k+8<=blen;k+=8) _mm256_store_ps(wd+k,bf16_dec8(w+base+k));
                for(;k<blen;k++) wd[k]=bf16_to_f32(w[base+k]);
                for(int s=0;s<ns;s++){
                    const float *xs=x+(int64_t)(s0+s)*I+base;
                    __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0; int j=0;
                    for(;j+32<=blen;j+=32){
                        a0=_mm256_fmadd_ps(_mm256_load_ps(wd+j   ),_mm256_loadu_ps(xs+j   ),a0);
                        a1=_mm256_fmadd_ps(_mm256_load_ps(wd+j+ 8),_mm256_loadu_ps(xs+j+ 8),a1);
                        a2=_mm256_fmadd_ps(_mm256_load_ps(wd+j+16),_mm256_loadu_ps(xs+j+16),a2);
                        a3=_mm256_fmadd_ps(_mm256_load_ps(wd+j+24),_mm256_loadu_ps(xs+j+24),a3);
                    }
                    float acc=hsum8(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
                    for(;j<blen;j++) acc+=wd[j]*xs[j];
                    a[s]+=acc;
                }
            }
            for(int s=0;s<ns;s++) y[(int64_t)(s0+s)*O+o]=a[s];
        }
    }
}
#endif /* HAVE_VEC */

/* ---- harness ------------------------------------------------------------ */
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static uint64_t rs=88172645463325252ull;
static inline uint64_t xs64(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static inline float frand(void){ return (float)((int64_t)(xs64()>>40)-8192)/8192.0f; }

/* One prefill chunk touches ~233 distinct experts per layer, all cold. A pool
 * that fits in L3 would measure the cache instead of DRAM. */
#define POOL_MIB 384
static double g_secs=0.4;

typedef void (*fp8_fn)(float*,const float*,const uint8_t*,const float*,int,int,int);
typedef void (*bf16_fn)(float*,const float*,const uint16_t*,int,int,int);

static void bench_fp8(const char *tag,int I,int O,const int *Sv,int nS){
    int64_t wb=(int64_t)I*O;
    int npool=(int)(((int64_t)POOL_MIB<<20)/wb); if(npool<2)npool=2; if(npool>512)npool=512;
    uint8_t **pool=malloc(sizeof(*pool)*npool);
    int64_t nblk=fp8_nblk(O)*fp8_nblk(I);
    float *bs=malloc(sizeof(float)*nblk);
    for(int64_t i=0;i<nblk;i++) bs[i]=0.005f+0.05f*(float)((xs64()>>40)&255)/255.0f;
    for(int p=0;p<npool;p++){
        pool[p]=aligned_alloc(64,(size_t)wb);
        /* No NaN in the measurement data: NaN handling is asserted separately
         * over all 256 bytes, and here it would poison the error norms. */
        for(int64_t i=0;i<wb;i++){ uint8_t b=(uint8_t)(xs64()>>33); if((b&0x7F)==0x7F) b&=0xFE; pool[p][i]=b; }
    }
    int Smax=Sv[nS-1];
    float *x=aligned_alloc(64,(size_t)Smax*I*sizeof(float));
    for(int64_t i=0;i<(int64_t)Smax*I;i++) x[i]=frand();
    float *y0=aligned_alloc(64,(size_t)Smax*O*sizeof(float));
    float *y1=aligned_alloc(64,(size_t)Smax*O*sizeof(float));
    double *ye=malloc((size_t)Smax*O*sizeof(double));

    printf("\n=== FP8 %s  I=%d O=%d  (pool of %d matrices, %.0f MiB)\n",
           tag,I,O,npool,(double)(npool*wb)/1048576.0);
    printf("%4s  %-16s %10s %9s %11s %11s\n","S","kernel","GFLOP/s","vs V0","err vs V0","err vs exact");

#if HAVE_VEC
    const char *nm[4]={"V0 scalar","V1 simd","V2 simd+reuse","V3 simd+gather"};
    fp8_fn fn[4]={matmul_fp8_v0,matmul_fp8_v1,matmul_fp8_v2,matmul_fp8_v3};
    int nv=4;
#else
    const char *nm[1]={"V0 scalar"};
    fp8_fn fn[1]={matmul_fp8_v0};
    int nv=1;
#endif
    for(int si=0;si<nS;si++){
        int S=Sv[si]; double base=0;
        matmul_fp8_exact(ye,x,pool[0],bs,S,I,O);
        matmul_fp8_v0(y0,x,pool[0],bs,S,I,O);
        for(int v=0;v<nv;v++){
            fn[v](y1,x,pool[0],bs,S,I,O);
            double d0=0,de=0,nrm=0;
            for(int64_t i=0;i<(int64_t)S*O;i++){
                double r=ye[i],u=y1[i],z=y0[i];
                nrm+=r*r; d0+=(u-z)*(u-z); de+=(u-r)*(u-r);
            }
            nrm=sqrt(nrm)+1e-30;
            double t0=now(),t1; int64_t reps=0;
            do{ for(int p=0;p<npool;p++){ fn[v](y1,x,pool[p],bs,S,I,O); reps++; } t1=now(); }while(t1-t0<g_secs);
            double gf=2.0*(double)S*I*O*(double)reps/(t1-t0)/1e9;
            if(v==0) base=gf;
            printf("%4d  %-16s %10.2f %8.2fx %11.2e %11.2e\n",
                   S,nm[v],gf,gf/base,sqrt(d0)/nrm,sqrt(de)/nrm);
        }
    }
    for(int p=0;p<npool;p++) free(pool[p]);
    free(pool);free(bs);free(x);free(y0);free(y1);free(ye);
}

static void bench_bf16(const char *tag,int I,int O,const int *Sv,int nS){
    int64_t wel=(int64_t)I*O;
    int npool=(int)(((int64_t)POOL_MIB<<20)/(wel*2)); if(npool<2)npool=2; if(npool>64)npool=64;
    uint16_t **pool=malloc(sizeof(*pool)*npool);
    for(int p=0;p<npool;p++){
        pool[p]=aligned_alloc(64,(size_t)wel*2);
        for(int64_t i=0;i<wel;i++){ union{float f;uint32_t u;}c; c.f=0.05f*frand(); pool[p][i]=(uint16_t)(c.u>>16); }
    }
    int Smax=Sv[nS-1];
    float *x=aligned_alloc(64,(size_t)Smax*I*sizeof(float));
    for(int64_t i=0;i<(int64_t)Smax*I;i++) x[i]=frand();
    float *y0=aligned_alloc(64,(size_t)Smax*O*sizeof(float));
    float *y1=aligned_alloc(64,(size_t)Smax*O*sizeof(float));
    printf("\n=== BF16 %s  I=%d O=%d  (pool of %d matrices, %.0f MiB)\n",
           tag,I,O,npool,(double)(npool*wel*2)/1048576.0);
    printf("%4s  %-16s %10s %9s %11s\n","S","kernel","GFLOP/s","vs V0","err vs V0");
#if HAVE_VEC
    const char *nm[3]={"V0 scalar","V1 simd","V2 simd+reuse"};
    bf16_fn fn[3]={matmul_bf16_v0,matmul_bf16_v1,matmul_bf16_v2};
    int nv=3;
#else
    const char *nm[1]={"V0 scalar"};
    bf16_fn fn[1]={matmul_bf16_v0};
    int nv=1;
#endif
    for(int si=0;si<nS;si++){
        int S=Sv[si]; double base=0;
        matmul_bf16_v0(y0,x,pool[0],S,I,O);
        for(int v=0;v<nv;v++){
            fn[v](y1,x,pool[0],S,I,O);
            double d0=0,nrm=0;
            for(int64_t i=0;i<(int64_t)S*O;i++){
                nrm+=(double)y0[i]*y0[i];
                d0+=((double)y1[i]-y0[i])*((double)y1[i]-y0[i]);
            }
            nrm=sqrt(nrm)+1e-30;
            double t0=now(),t1; int64_t reps=0;
            do{ for(int p=0;p<npool;p++){ fn[v](y1,x,pool[p],S,I,O); reps++; } t1=now(); }while(t1-t0<g_secs);
            double gf=2.0*(double)S*I*O*(double)reps/(t1-t0)/1e9;
            if(v==0) base=gf;
            printf("%4d  %-16s %10.2f %8.2fx %11.2e\n",S,nm[v],gf,gf/base,sqrt(d0)/nrm);
        }
    }
    for(int p=0;p<npool;p++) free(pool[p]);
    free(pool);free(x);free(y0);free(y1);
}

int main(int argc,char **argv){
    if(argc>1){ double s=atof(argv[1]); if(s>0) g_secs=s; }
    lut_init();
#if HAVE_VEC
    /* The vector decode must match the table on all 256 bytes, NaN included.
     * If it misses even one, nothing measured afterwards means anything. */
    {
        uint8_t all[256]; for(int i=0;i<256;i++) all[i]=(uint8_t)i;
        float ob[256],og[256];
        for(int i=0;i<256;i+=8){
            _mm256_storeu_ps(ob+i,e4m3_dec8_bits(all+i));
            _mm256_storeu_ps(og+i,e4m3_dec8_gather(all+i));
        }
        int bad=0,nans=0;
        for(int i=0;i<256;i++){
            float r=E4M3_LUT[i];
            if(isnan(r)){ nans++; if(!isnan(ob[i])||!isnan(og[i])) bad++; }
            else { union{float f;uint32_t u;}a,b,c; a.f=r;b.f=ob[i];c.f=og[i];
                   if(a.u!=b.u||a.u!=c.u) bad++; }
        }
        printf("E4M3 decode: 256 bytes, %d NaN expected, %d bit-exact mismatches\n",nans,bad);
        if(bad){ printf("ABORT: the vector decode is not exact\n"); return 1; }
    }
#else
    printf("built without AVX2+FMA: scalar reference only\n");
#endif
#ifdef _OPENMP
    printf("OMP threads: %d\n",omp_get_max_threads());
#endif
    printf("seconds per measurement: %.2f\n",g_secs);

    /* Real S values. 1 = decode. 4 = narrow chunk (32 rows, ~3.3 rows per
     * expert). 12 = widened chunk (274 rows over 233 distinct experts per
     * layer, ~11.8). 32 = the heavy tail; routing is far from uniform. */
    static const int Se[4]={1,4,12,32};
    bench_fp8("gate/up",2560,640,Se,4);
    bench_fp8("down",   640,2560,Se,4);

    /* Resident weights see the whole prompt in one go: S=1 in decode, S=274
     * in this prefill. Shapes: v-proj 2560->6144, o-proj 6144->2560. */
    static const int Sr[2]={1,274};
    bench_bf16("v-proj",2560,6144,Sr,2);
    bench_bf16("o-proj",6144,2560,Sr,2);
    return 0;
}
