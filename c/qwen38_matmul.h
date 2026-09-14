/* qwen38_matmul.h — dense weight matmuls for the Qwen3.8-Flash-Next text core.
 *
 * Header-only and free of any Model or tokenizer dependency, which is the
 * point: qwen38_core.h cannot be compiled on its own, so kernels living there
 * could only ever be tested through a full engine run.  Here they can be
 * driven directly by a toy bank.  qwen38_core.h includes this header and
 * chooses between the reference and the vector form in q38_weight_matmul.
 */
#ifndef COLI_QWEN38_MATMUL_H
#define COLI_QWEN38_MATMUL_H

#include <stdint.h>

#include "quant.h"

/* The same shift st.h's loader applies when it materialises bf16 as f32,
 * spelled here so a compute header needs nothing from the safetensors
 * reader -- which defines _GNU_SOURCE and must be included before libc. */
static inline float q38_bf16(uint16_t h) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)h << 16;
    return c.f;
}

/* W is row-major [O,I], y=x@W^T. */
static void q38_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * w[i];
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* Native BF16 storage with FP32 activations and accumulation.  This deliberately
 * does not round activations to BF16 or use BF16 dot-product instructions: it is
 * the storage-equivalent form of the existing st_read_f32 reference. */
static void q38_matmul_bf16(float *y,const float *x,const uint16_t *W,
                            int S,int I,int O) {
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;float a=0.f;
            for(int i=0;i<I;i++)a+=xs[i]*q38_bf16(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- AVX2 forms ----------------------------------------------------------
 *
 * The two kernels above decode one weight at a time and walk a weight row
 * once per sequence position.  Measured on this node (Zen 2, AVX2, no
 * AVX-512, 20 threads) the cost splits two ways: vectorising the decode is
 * worth 2.0x at S=12, and decoding a tile once and serving the whole batch
 * from it is worth another 3.1x on top of that -- 6.3x together, 7.5x at
 * S=32.  Both are prefill levers only: at S=1 either form sits on the DRAM
 * roof (~50 GFLOP/s against a ~45 GB/s bus) and neither changes anything.
 *
 * fp8 is therefore blocked: each 128-weight quantisation block is decoded
 * once into a stack tile that the rows of the batch share.  bf16 is
 * deliberately not blocked -- its weights are twice as wide, its decode is a
 * single shift, and blocking measured slower than the plain vector form.
 * deepseek_v41.c reached both conclusions independently in mv8_rows and mvb.
 *
 * The arithmetic follows the reference exactly: for fp8 a float accumulator
 * inside a quantisation block and a double across blocks, for bf16 a float
 * throughout.  The vector form is not bit-identical to the scalar one -- the
 * summation order differs -- but it is measurably closer to a float64
 * reference on these shapes, so it is an accuracy improvement as well as a
 * speed one.  What it IS bit-identical to is itself under any split of the
 * batch, which is the property test_qwen38_simd_matmul pins: a row computed
 * alone and the same row computed inside a batch must agree to the bit.
 */
#ifdef __AVX2__
/* Rows of the batch served from one decoded fp8 tile.  The tile itself is
 * 512 B; what bounds this is the per-thread accumulator array.  Any value
 * produces the same numbers -- only the reuse factor changes. */
#define Q38_MV_ROWS 64

/* Four independent accumulators to cover the FMA latency, then one horizontal
 * sum: the reduction order is fixed, so the result depends only on n. */
static inline float q38_dot8(const float *w,const float *x,int n) {
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
    __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
    int i=0;
    for(;i+32<=n;i+=32){
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),   _mm256_loadu_ps(x+i),   a0);
        a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+8), _mm256_loadu_ps(x+i+8), a1);
        a2=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+16),_mm256_loadu_ps(x+i+16),a2);
        a3=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+24),_mm256_loadu_ps(x+i+24),a3);
    }
    for(;i+8<=n;i+=8)
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),a0);
    float sum=hsum256(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
    for(;i<n;i++)sum+=w[i]*x[i];
    return sum;
}

static void q38_matmul_fp8_avx2(float *y,const float *x,const uint8_t *q8,
                                const float *bscale,int S,int I,int O) {
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+(int64_t)(o/FP8_BLOCK)*nblkI;
        for(int r0=0;r0<S;r0+=Q38_MV_ROWS){
            int nr=S-r0<Q38_MV_ROWS?S-r0:Q38_MV_ROWS;
            double a[Q38_MV_ROWS];
            float tile[FP8_BLOCK];
            for(int r=0;r<nr;r++)a[r]=0;
            for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
                int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK;
                if(base+blen>I)blen=I-base;
                int k=0;
                for(;k+8<=blen;k+=8)
                    _mm256_storeu_ps(tile+k,e4m3_decode8(w+base+k));
                for(;k<blen;k++)tile[k]=e4m3_decode(w[base+k]);
                float sc=scl[bi];
                for(int r=0;r<nr;r++)
                    a[r]+=(double)q38_dot8(tile,x+(int64_t)(r0+r)*I+base,blen)*sc;
            }
            for(int r=0;r<nr;r++)y[(int64_t)(r0+r)*O+o]=(float)a[r];
        }
    }
}

/* Grouped int4 (gs64 expert container), same blocking as fp8: one group of
 * weights is decoded into a stack tile once and the whole batch chunk is
 * served from it.  The bank (bench_qwen38_int4_gemv) measured the unblocked
 * form at 0.44-0.66x of fp8-D2 for S>=4 precisely because it re-decoded each
 * group per batch row; this form is the fourth arm (1.05-1.20x fp8 at S=4/12,
 * parity at S=32).  At S=1 it degenerates to the plain vector decode with one
 * tile buffer: 133.7 vs 134.0 GFLOP/s in the bank, i.e. free.  Accumulation
 * is float inside a group and double across groups, mirroring the fp8 kernel;
 * the result is bit-identical to itself under any batch split (row
 * accumulators reset per chunk) and closer to a float64 reference than
 * matmul_i4_grouped's float-throughout accumulation. */
static void q38_matmul_i4_avx2(float *y,const float *x,const uint8_t *q4,
                               const float *scales,int gs,int S,int I,int O) {
    if(gs>64){fprintf(stderr,"q38_matmul_i4_avx2: gs=%d exceeds tile\n",gs);exit(1);}
    int ng=(I+gs-1)/gs;
    const __m128i m4=_mm_set1_epi8(0x0F);
    const __m256i b8=_mm256_set1_epi32(8);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*((I+1)/2);
        const float *scl=scales+(int64_t)o*ng;
        float tile[64];              /* the container pins gs=64 */
        for(int r0=0;r0<S;r0+=Q38_MV_ROWS){
            int nr=S-r0<Q38_MV_ROWS?S-r0:Q38_MV_ROWS;
            double a[Q38_MV_ROWS];
            for(int r=0;r<nr;r++)a[r]=0;
            for(int g=0;g*gs<I;g++){
                int base=g*gs,blen=gs;
                if(base+blen>I)blen=I-base;
                int k=0;
                for(;k+16<=blen;k+=16){
                    __m128i by=_mm_loadl_epi64((const __m128i*)(w+((base+k)>>1)));
                    __m128i lo=_mm_and_si128(by,m4);
                    __m128i hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    _mm256_storeu_ps(tile+k,
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(
                            _mm256_cvtepu8_epi32(nib),b8)));
                    _mm256_storeu_ps(tile+k+8,
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(
                            _mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8)));
                }
                for(;k<blen;k++){
                    uint8_t byte=w[(base+k)>>1];
                    tile[k]=(float)((int)((base+k)&1?byte>>4:byte&0xF)-8);
                }
                float sc=scl[g];
                for(int r=0;r<nr;r++)
                    a[r]+=(double)q38_dot8(tile,x+(int64_t)(r0+r)*I+base,blen)*sc;
            }
            for(int r=0;r<nr;r++)y[(int64_t)(r0+r)*O+o]=(float)a[r];
        }
    }
}

static void q38_matmul_bf16_avx2(float *y,const float *x,const uint16_t *W,
                                 int S,int I,int O) {
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
            __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
            int i=0;
            for(;i+32<=I;i+=32){
                a0=_mm256_fmadd_ps(bf16_decode8(w+i),   _mm256_loadu_ps(xs+i),   a0);
                a1=_mm256_fmadd_ps(bf16_decode8(w+i+8), _mm256_loadu_ps(xs+i+8), a1);
                a2=_mm256_fmadd_ps(bf16_decode8(w+i+16),_mm256_loadu_ps(xs+i+16),a2);
                a3=_mm256_fmadd_ps(bf16_decode8(w+i+24),_mm256_loadu_ps(xs+i+24),a3);
            }
            for(;i+8<=I;i+=8)
                a0=_mm256_fmadd_ps(bf16_decode8(w+i),_mm256_loadu_ps(xs+i),a0);
            float acc=hsum256(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
            for(;i<I;i++)acc+=xs[i]*q38_bf16(w[i]);
            y[(int64_t)s*O+o]=acc;
        }
    }
}
#endif /* __AVX2__ */

#endif /* COLI_QWEN38_MATMUL_H */
