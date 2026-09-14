/* The vector weight matmuls must agree with themselves and beat the reference.
 *
 * qwen38_matmul.h decodes an fp8 quantisation block once and serves a block of
 * batch rows from the decoded tile.  That reuse is where the 6-7x comes from
 * and it is also the only thing in the kernel that can silently go wrong: get
 * the row blocking off by one, or let an accumulator leak across blocks, and
 * the output is still plausible -- slightly wrong logits, no crash, nothing a
 * smoke test would catch.  So the first property here is exact: a row computed
 * on its own and the same row computed inside a batch of 100 must agree to the
 * bit.  Q38_MV_ROWS is 64, so a batch of 100 really does split.
 *
 * The second property is that the vector form is not merely close to the
 * scalar reference but closer than it to the truth.  Both accumulate a
 * 128-weight block in float; the reference does it as one serial chain, the
 * vector form as four, which is shorter and therefore rounds less.  Measured
 * against a long double reference on the engine's own shapes the vector form
 * was ~2.7x more accurate, so this asserts the direction of that difference
 * rather than just a tolerance -- a tolerance alone would pass a kernel that
 * had quietly become worse.
 *
 * NaN is the third: e4m3's 0x7F and 0xFF are NaN and a corrupt weight must
 * poison its output rather than turn into a plausible +-480.
 *
 * Geometries deliberately include one with ugly tails (I not a multiple of
 * 128 and its last block not a multiple of 8) to exercise the scalar
 * remainder paths on both the decode and the dot.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../qwen38_matmul.h"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float rnd_unit(void) { return (float)((double)rnd() / 4294967295.0 * 2.0 - 1.0); }

/* An e4m3 byte with a realistic exponent: bias is 7, so a field of 6..8 spans
 * 0.5 to 3.75.  Uniform random bytes would span 2^-9 to 448 and let a handful
 * of terms dominate every dot product, which would say nothing about how the
 * kernels round. */
static uint8_t rnd_e4m3(void) {
    uint32_t r = rnd();
    return (uint8_t)((r & 0x80u) | (((6u + ((r >> 8) % 3u)) & 0xFu) << 3) | ((r >> 4) & 0x7u));
}

/* Exact value of one output, block scaling included, plus the magnitude that
 * went into it.  The magnitude is the denominator that makes the error
 * meaningful: with random activations a dot product of 2560 terms cancels down
 * to a small fraction of what was summed, so dividing by the result would
 * measure the cancellation, not the kernel.  Dividing by sum|w*x| is the
 * standard backward error -- how much of what was actually added got lost. */
typedef struct { long double value, magnitude; } Ref;

static Ref ref_fp8(const float *x, const uint8_t *w, const float *scl,
                   int I, int s) {
    const float *xs = x + (int64_t)s * I;
    Ref r = {0, 0};
    for (int64_t bi = 0; bi * FP8_BLOCK < I; bi++) {
        int base = (int)(bi * FP8_BLOCK), blen = FP8_BLOCK;
        if (base + blen > I) blen = I - base;
        long double acc = 0, mag = 0;
        for (int i = base; i < base + blen; i++) {
            long double t = (long double)e4m3_decode(w[i]) * (long double)xs[i];
            acc += t;
            mag += fabsl(t);
        }
        r.value += acc * (long double)scl[bi];
        r.magnitude += mag * (long double)fabsf(scl[bi]);
    }
    return r;
}

static Ref ref_bf16(const float *x, const uint16_t *w, int I, int s) {
    const float *xs = x + (int64_t)s * I;
    Ref r = {0, 0};
    for (int i = 0; i < I; i++) {
        long double t = (long double)q38_bf16(w[i]) * (long double)xs[i];
        r.value += t;
        r.magnitude += fabsl(t);
    }
    return r;
}

static double rel(double got, Ref r) {
    long double m = r.magnitude;
    if (m < 1e-30L) m = 1e-30L;
    return (double)(fabsl((long double)got - r.value) / m);
}

#ifdef __AVX2__
/* max relative error of each kernel against the long double truth */
static void fp8_case(int I, int O, int S, const char *name) {
    uint8_t *w = malloc((size_t)I * O);
    float *x = malloc(sizeof(float) * (size_t)I * S);
    int64_t nb = fp8_nblk(I) * fp8_nblk(O);
    float *scl = malloc(sizeof(float) * (size_t)nb);
    float *ys = malloc(sizeof(float) * (size_t)O * S);   /* scalar reference */
    float *yv = malloc(sizeof(float) * (size_t)O * S);   /* vector, whole batch */
    float *y1 = malloc(sizeof(float) * (size_t)O * S);   /* vector, row at a time */
    if (!w || !x || !scl || !ys || !yv || !y1) { printf("  FAIL: OOM\n"); fails++; return; }

    for (int64_t i = 0; i < (int64_t)I * O; i++) w[i] = rnd_e4m3();
    for (int64_t i = 0; i < (int64_t)I * S; i++) x[i] = rnd_unit();
    for (int64_t i = 0; i < nb; i++) scl[i] = 0.5f + (float)((double)rnd() / 4294967295.0);

    matmul_fp8(ys, x, w, scl, S, I, O);
    q38_matmul_fp8_avx2(yv, x, w, scl, S, I, O);
    for (int s = 0; s < S; s++)
        q38_matmul_fp8_avx2(y1 + (int64_t)s * O, x + (int64_t)s * I, w, scl, 1, I, O);

    int split_bad = 0;
    double es = 0, ev = 0;
    for (int s = 0; s < S; s++) {
        for (int o = 0; o < O; o++) {
            int64_t k = (int64_t)s * O + o;
            if (memcmp(&yv[k], &y1[k], sizeof(float))) split_bad++;
            Ref r = ref_fp8(x, w + (int64_t)o * I,
                            scl + (int64_t)(o / FP8_BLOCK) * fp8_nblk(I), I, s);
            double a = rel(ys[k], r), b = rel(yv[k], r);
            if (a > es) es = a;
            if (b > ev) ev = b;
        }
    }
    printf("  fp8 %-14s S=%-4d scalar %.3g  vector %.3g  (%.2fx better)\n",
           name, S, es, ev, ev > 0 ? es / ev : 0.0);
    check(split_bad == 0, "a batched fp8 row differs from the same row computed alone");
    check(ev < 1e-6, "the vector fp8 kernel is not within tolerance of the truth");
    check(ev <= es, "the vector fp8 kernel is less accurate than the scalar reference");

    free(w); free(x); free(scl); free(ys); free(yv); free(y1);
}

static void bf16_case(int I, int O, int S, const char *name) {
    uint16_t *w = malloc(sizeof(uint16_t) * (size_t)I * O);
    float *x = malloc(sizeof(float) * (size_t)I * S);
    float *ys = malloc(sizeof(float) * (size_t)O * S);
    float *yv = malloc(sizeof(float) * (size_t)O * S);
    float *y1 = malloc(sizeof(float) * (size_t)O * S);
    if (!w || !x || !ys || !yv || !y1) { printf("  FAIL: OOM\n"); fails++; return; }

    /* bf16 of a random unit float: the top 16 bits, which is what the loader
     * stores and what the decoders read back. */
    for (int64_t i = 0; i < (int64_t)I * O; i++) {
        float f = rnd_unit();
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        w[i] = (uint16_t)(bits >> 16);
    }
    for (int64_t i = 0; i < (int64_t)I * S; i++) x[i] = rnd_unit();

    q38_matmul_bf16(ys, x, w, S, I, O);
    q38_matmul_bf16_avx2(yv, x, w, S, I, O);
    for (int s = 0; s < S; s++)
        q38_matmul_bf16_avx2(y1 + (int64_t)s * O, x + (int64_t)s * I, w, 1, I, O);

    int split_bad = 0;
    double es = 0, ev = 0;
    for (int s = 0; s < S; s++) {
        for (int o = 0; o < O; o++) {
            int64_t k = (int64_t)s * O + o;
            if (memcmp(&yv[k], &y1[k], sizeof(float))) split_bad++;
            Ref r = ref_bf16(x, w + (int64_t)o * I, I, s);
            double a = rel(ys[k], r), b = rel(yv[k], r);
            if (a > es) es = a;
            if (b > ev) ev = b;
        }
    }
    printf("  bf16 %-13s S=%-4d scalar %.3g  vector %.3g  (%.2fx better)\n",
           name, S, es, ev, ev > 0 ? es / ev : 0.0);
    check(split_bad == 0, "a batched bf16 row differs from the same row computed alone");
    check(ev < 1e-6, "the vector bf16 kernel is not within tolerance of the truth");
    check(ev <= es, "the vector bf16 kernel is less accurate than the scalar reference");

    free(w); free(x); free(ys); free(yv); free(y1);
}

/* A corrupt weight must reach the output as NaN, not as a plausible number. */
static void nan_case(void) {
    enum { I = 256, O = 8, S = 3 };
    uint8_t w[I * O];
    float x[I * S], scl[8], y[O * S];
    for (int i = 0; i < I * O; i++) w[i] = rnd_e4m3();
    for (int i = 0; i < I * S; i++) x[i] = rnd_unit();
    for (int i = 0; i < 8; i++) scl[i] = 1.0f;
    w[(int64_t)3 * I + 200] = 0x7F;   /* e4m3 NaN, second block of row 3 */

    q38_matmul_fp8_avx2(y, x, w, scl, S, I, O);
    int poisoned = 1, spilled = 0;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            int bad = isnan(y[(int64_t)s * O + o]);
            if (o == 3 && !bad) poisoned = 0;
            if (o != 3 && bad) spilled = 1;
        }
    check(poisoned, "a NaN weight did not reach the output it belongs to");
    check(!spilled, "a NaN weight poisoned an output row it does not belong to");
}
#endif

int main(void) {
#ifndef __AVX2__
    printf("SKIP: built without AVX2, the reference kernels are the only ones\n");
    return 0;
#else
    /* the engine's own shapes: expert gate/up, expert down, a dense proj */
    fp8_case(2560, 640, 1, "expert gate/up");
    fp8_case(2560, 640, 100, "expert gate/up");
    fp8_case(640, 2560, 100, "expert down");
    fp8_case(323, 131, 70, "ragged tails");

    bf16_case(2560, 200, 1, "v-proj");
    bf16_case(2560, 200, 100, "v-proj");
    bf16_case(37, 11, 5, "ragged tails");

    nan_case();

    if (fails) { printf("test_qwen38_simd_matmul: %d failures\n", fails); return 1; }
    printf("test_qwen38_simd_matmul: ok\n");
    return 0;
#endif
}
