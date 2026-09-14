/* bench_qwen38_int4_gemv.c -- is int4-gs64 penalised on an AVX2-only CPU?
 *
 * This bank runs BEFORE the int4 expert slots are ported into the loader,
 * because the answer decides the port: packed slots (decode inside the GEMV)
 * or unpack-at-load (int8 slots, 5.2 MiB each, which costs the residency the
 * whole phase is about).
 *
 * The reference arm is the engine's own fp8 path, qwen38_matmul.h's
 * q38_matmul_fp8_avx2 -- that header carries no Model dependency, which is
 * exactly what lets a toy driver call it. The int4 arms are:
 *
 *   i4-scalar   nibble decode with no SIMD at all, the floor;
 *   i4-avx2     quant.h's matmul_i4_grouped() exactly as the engine has it;
 *   i4-avx2-r   the same decode, but blocked like the fp8 kernel: a group is
 *               decoded once into a tile and then serves every row of the
 *               batch. D1 measured that this reuse is worth MORE than the
 *               vectorisation itself (3.1x against 2.0x), and
 *               matmul_i4_grouped does not have it -- its row loop sits INSIDE
 *               the output loop, so at S>1 it re-decodes the same nibbles once
 *               per position. Without this arm a red verdict would be about
 *               the kernel we happen to have, not about the format.
 *
 * Why a pool instead of one matrix: one expert projection is 1.6 MB in fp8 and
 * 0.8 MB in int4, so a repeat loop over a single matrix measures the L3 of a
 * 3900X (64 MB), not DRAM -- and the decode regime is DRAM-bound, which is the
 * whole question. The pool is sized to ~384 MiB of fp8 weights (the figure D1
 * used, so the fp8 numbers here stay comparable with the ones already in the
 * register) and every pass walks all of it.
 *
 * The pools hold the SAME number of experts in both formats, not the same
 * number of bytes: that asymmetry is the thing being measured.
 *
 *   make tests/bench_qwen38_int4_gemv ARCH=native
 *   OMP_NUM_THREADS=20 ./tests/bench_qwen38_int4_gemv [pool_MiB]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "../qwen38_matmul.h"   /* pulls quant.h: matmul_i4_grouped, e4m3 */

#define GS 64

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec;
}

/* ---- deterministic weights ------------------------------------------------ */
/* Sum of four uniforms: roughly gaussian, which is what keeps the int4 error at
 * the theoretical value rather than an artefact of a flat distribution. */
static uint64_t lcg(uint64_t *s){ *s = *s*6364136223846793005ULL + 1442695040888963407ULL; return *s; }
static float randn(uint64_t *s){
    double a=0; for(int k=0;k<4;k++) a += (double)((lcg(s)>>11)&0xFFFFF)/(double)0x100000 - 0.5;
    return (float)(a*0.5);
}

/* ---- int4 gs64 quantisation, byte-identical to the converter's container --- */
/* Offset binary (u = q + 8), scales [O][I/gs] row-major, floor 1e-8: the three
 * choices tools/convert_qwen38_int4.py writes and matmul_i4_grouped decodes. */
static void quant_i4(const float *w,int O,int I,uint8_t *q4,float *sc){
    int rb=I/2, ng=I/GS;
    for(int o=0;o<O;o++){
        for(int g=0;g<ng;g++){
            const float *src=w+(int64_t)o*I+(int64_t)g*GS;
            float amax=0; for(int i=0;i<GS;i++){ float a=fabsf(src[i]); if(a>amax)amax=a; }
            float s=amax/7.0f; if(s<1e-8f)s=1e-8f;
            sc[(int64_t)o*ng+g]=s;
            for(int i=0;i<GS;i+=2){
                int lo=(int)lrintf(src[i]/s),   hi=(int)lrintf(src[i+1]/s);
                if(lo>7)lo=7; if(lo<-8)lo=-8; if(hi>7)hi=7; if(hi<-8)hi=-8;
                q4[(int64_t)o*rb+(g*GS+i)/2]=(uint8_t)((lo+8)|((hi+8)<<4));
            }
        }
    }
}

/* ---- fp8 e4m3 quantisation with 128x128 block scales (the checkpoint's own) - */
static float POS[128]; static uint8_t POSB[128]; static int NPOS=0;
static int cmpf(const void *a,const void *b){
    float x=*(const float*)a,y=*(const float*)b; return x<y?-1:x>y?1:0;
}
static void build_pos(void){
    float tmp[256]; int n=0;
    for(int b=0;b<128;b++){ float v=E4M3_LUT[b]; if(v==v && v>0) tmp[n++]=v; }
    qsort(tmp,(size_t)n,sizeof(float),cmpf);
    for(int i=0;i<n;i++){
        POS[i]=tmp[i];
        for(int b=0;b<128;b++) if(E4M3_LUT[b]==tmp[i]){ POSB[i]=(uint8_t)b; break; }
    }
    NPOS=n;
}
static uint8_t enc_e4m3(float v){
    int neg = v<0; float a = neg ? -v : v;
    if(!(a>0)) return (uint8_t)(neg?0x80:0x00);
    int lo=0, hi=NPOS-1;                       /* nearest on the sorted ladder */
    while(lo<hi){ int mid=(lo+hi)/2; if(POS[mid]<a) lo=mid+1; else hi=mid; }
    int best=lo;
    if(lo>0 && (a-POS[lo-1]) < (POS[lo]-a)) best=lo-1;
    return (uint8_t)(POSB[best] | (neg?0x80:0x00));
}
static void quant_fp8(const float *w,int O,int I,uint8_t *q8,float *bs){
    int64_t nbI=fp8_nblk(I), nbO=fp8_nblk(O);
    for(int64_t bo=0;bo<nbO;bo++)for(int64_t bi=0;bi<nbI;bi++){
        int o0=(int)(bo*FP8_BLOCK), i0=(int)(bi*FP8_BLOCK);
        int o1=o0+FP8_BLOCK>O?O:o0+FP8_BLOCK, i1=i0+FP8_BLOCK>I?I:i0+FP8_BLOCK;
        float amax=0;
        for(int o=o0;o<o1;o++)for(int i=i0;i<i1;i++){
            float a=fabsf(w[(int64_t)o*I+i]); if(a>amax)amax=a; }
        float s=amax/448.0f; if(s<1e-8f)s=1e-8f;
        bs[bo*nbI+bi]=s;
        for(int o=o0;o<o1;o++)for(int i=i0;i<i1;i++)
            q8[(int64_t)o*I+i]=enc_e4m3(w[(int64_t)o*I+i]/s);
    }
}

/* ---- arm 1: the scalar floor ---------------------------------------------- */
static void i4_scalar(float *y,const float *x,const uint8_t *q4,const float *scale,
                      int S,int I,int O){
    int rb=I/2, ng=I/GS;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){
                float sc=scl[g], acc=0;
                for(int i=g*GS;i<(g+1)*GS;i+=2){
                    uint8_t byte=w[i>>1];
                    acc += xs[i]*(float)((int)(byte&0xF)-8) + xs[i+1]*(float)((int)(byte>>4)-8);
                }
                a += acc*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- arm 3: AVX2 decode once per group, then serve the whole batch --------- */
static void i4_tile(float *y,const float *x,const uint8_t *q4,const float *scale,
                    int S,int I,int O){
    int rb=I/2, ng=I/GS;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int r0=0;r0<S;r0+=Q38_MV_ROWS){
            int nr=S-r0<Q38_MV_ROWS?S-r0:Q38_MV_ROWS;
            double a[Q38_MV_ROWS]; float tile[GS];
            for(int r=0;r<nr;r++)a[r]=0;
            for(int g=0;g<ng;g++){
                int base=g*GS;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                for(int k=0;k<GS;k+=16){
                    __m128i by=_mm_loadl_epi64((const __m128i*)(w+((base+k)>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    _mm256_storeu_ps(tile+k,
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8)));
                    _mm256_storeu_ps(tile+k+8,
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(
                            _mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8)));
                }
#else
                for(int k=0;k<GS;k+=2){ uint8_t byte=w[(base+k)>>1];
                    tile[k]=(float)((int)(byte&0xF)-8); tile[k+1]=(float)((int)(byte>>4)-8); }
#endif
                float sc=scl[g];
                for(int r=0;r<nr;r++)
                    a[r]+=(double)q38_dot8(tile,x+(int64_t)(r0+r)*I+base,GS)*sc;
            }
            for(int r=0;r<nr;r++)y[(int64_t)(r0+r)*O+o]=(float)a[r];
        }
    }
}

/* ---- the bank ------------------------------------------------------------- */
enum { ARM_I4S, ARM_I4V, ARM_I4T, ARM_FP8, N_ARM };
static const char *ARM_NAME[N_ARM]={"i4-scalar","i4-avx2","i4-avx2-r","fp8-D2"};

static double dmin(double a,double b){ return a<b?a:b; }

static void run_shape(const char *label,int I,int O,int pool_mib,const int *SS,int nS){
    int64_t elems=(int64_t)O*I;
    int N=(int)(((int64_t)pool_mib<<20)/elems);           /* fp8 bytes == elems */
    if(N<1)N=1;
    int rb=I/2, ng=I/GS;
    int64_t nbI=fp8_nblk(I), nbO=fp8_nblk(O);

    uint8_t *q4=malloc((size_t)N*O*rb);
    float   *qs=malloc((size_t)N*O*ng*sizeof(float));
    uint8_t *q8=malloc((size_t)N*elems);
    float   *bs=malloc((size_t)N*nbO*nbI*sizeof(float));
    if(!q4||!qs||!q8||!bs){ fprintf(stderr,"oom\n"); exit(2); }

    double t0=now_s();
    #pragma omp parallel for schedule(dynamic)
    for(int n=0;n<N;n++){
        uint64_t seed=0x9E3779B97F4A7C15ULL ^ ((uint64_t)n*1000003u) ^ ((uint64_t)I<<32);
        float *w=malloc((size_t)elems*sizeof(float));
        for(int64_t i=0;i<elems;i++) w[i]=randn(&seed);
        quant_i4(w,O,I,q4+(int64_t)n*O*rb, qs+(int64_t)n*O*ng);
        quant_fp8(w,O,I,q8+(int64_t)n*elems, bs+(int64_t)n*nbO*nbI);
        free(w);
    }
    double t_setup=now_s()-t0;

    int Smax=0; for(int k=0;k<nS;k++) if(SS[k]>Smax)Smax=SS[k];
    float *x=malloc((size_t)Smax*I*sizeof(float));
    float *y=malloc((size_t)Smax*O*sizeof(float));
    uint64_t xseed=12345; for(int64_t i=0;i<(int64_t)Smax*I;i++) x[i]=randn(&xseed);

    printf("\n== %s  I=%d O=%d  pool %d experts (%.0f MiB fp8 / %.0f MiB int4)  setup %.1fs\n",
           label,I,O,N,
           (double)((int64_t)N*elems)/1048576.0,
           (double)((int64_t)N*((int64_t)O*rb+(int64_t)O*ng*4))/1048576.0,
           t_setup);
    printf("   %-6s %-10s %10s %10s %10s %12s\n","S","arm","ms/pass","GFLOP/s","GB/s","vs fp8-D2");

    /* Correctness gate before any timing: the three int4 arms must agree, or
     * the numbers below would be timing three different computations. */
    {
        float *ya=malloc((size_t)Smax*O*sizeof(float));
        float *yb=malloc((size_t)Smax*O*sizeof(float));
        float *yc=malloc((size_t)Smax*O*sizeof(float));
        int S=SS[nS-1];
        i4_scalar(ya,x,q4,qs,S,I,O);
        matmul_i4_grouped(yb,x,q4,qs,S,I,O,GS);
        i4_tile(yc,x,q4,qs,S,I,O);
        double wb=0,wc=0,nrm=0;
        for(int64_t i=0;i<(int64_t)S*O;i++){
            wb=fmax(wb,fabs((double)ya[i]-(double)yb[i]));
            wc=fmax(wc,fabs((double)ya[i]-(double)yc[i]));
            nrm+=(double)ya[i]*(double)ya[i];
        }
        nrm=sqrt(nrm/(double)((int64_t)S*O));
        printf("   int4 arms agree: rms=%.4f  max|scalar-avx2|=%.2e  max|scalar-tile|=%.2e\n",
               nrm,wb,wc);
        if(wb>1e-3*(nrm+1.0)||wc>1e-3*(nrm+1.0)){ fprintf(stderr,"ARMS DISAGREE\n"); exit(1); }
        free(ya);free(yb);free(yc);
    }

    for(int k=0;k<nS;k++){
        int S=SS[k];
        double best[N_ARM];
        double flops=2.0*(double)S*(double)I*(double)O*(double)N;
        for(int a=0;a<N_ARM;a++){
            best[a]=1e30;
            double acc=0; int passes=0;
            while(passes<3 || acc<0.25){
                double t=now_s(); volatile float sink=0;
                for(int n=0;n<N;n++){
                    switch(a){
                    case ARM_I4S:
                        i4_scalar(y,x,q4+(int64_t)n*O*rb,qs+(int64_t)n*O*ng,S,I,O); break;
                    case ARM_I4V:
                        matmul_i4_grouped(y,x,q4+(int64_t)n*O*rb,qs+(int64_t)n*O*ng,S,I,O,GS); break;
                    case ARM_I4T:
                        i4_tile(y,x,q4+(int64_t)n*O*rb,qs+(int64_t)n*O*ng,S,I,O); break;
                    default:
                        q38_matmul_fp8_avx2(y,x,q8+(int64_t)n*elems,bs+(int64_t)n*nbO*nbI,S,I,O); break;
                    }
                    sink+=y[0];
                }
                double dt=now_s()-t; (void)sink;
                best[a]=dmin(best[a],dt); acc+=dt; passes++;
                if(a==ARM_I4S && passes>=3) break;   /* the floor is slow; 3 passes is enough */
            }
        }
        for(int a=0;a<N_ARM;a++){
            double bytes = (a==ARM_FP8)
                ? (double)((int64_t)N*(elems+nbO*nbI*4))
                : (double)((int64_t)N*((int64_t)O*rb+(int64_t)O*ng*4));
            printf("   %-6d %-10s %10.1f %10.1f %10.1f %11.2fx\n",
                   S,ARM_NAME[a],best[a]*1e3,flops/best[a]*1e-9,bytes/best[a]*1e-9,
                   best[ARM_FP8]/best[a]);
        }
    }
    free(q4);free(qs);free(q8);free(bs);free(x);free(y);
}

int main(int argc,char **argv){
    int pool_mib = argc>1 ? atoi(argv[1]) : 384;
    build_pos();
#ifdef _OPENMP
    printf("threads: %d   pool target: %d MiB (fp8)\n",omp_get_max_threads(),pool_mib);
#else
    printf("threads: 1 (no OpenMP)   pool target: %d MiB (fp8)\n",pool_mib);
#endif
#ifdef __AVX2__
    printf("AVX2: yes");
#else
    printf("AVX2: NO");
#endif
#if defined(__AVX512F__)
    printf("   AVX512: yes (not expected on Zen 2)\n");
#else
    printf("   AVX512: no\n");
#endif
    static const int SS[]={1,4,12,32};
    run_shape("expert gate/up",2560,640,pool_mib,SS,4);
    run_shape("expert down",   640,2560,pool_mib,SS,4);
    return 0;
}
