/* test_qwen38_int4_container.c — the qwen38 int4 container decodes under the
 * engine's own kernel.
 *
 * tools/convert_qwen38_int4.py has a --selftest, but that only proves the tool
 * agrees with itself: it quantizes and dequantizes through two functions in the
 * same file, so a consistent misreading of the container spec passes it. The
 * property that actually matters is cross-language: the bytes the CONVERTER
 * writes must be the bytes the ENGINE reads, and the engine's int4 group-scaled
 * decoder is quant.h's matmul_i4_grouped().
 *
 * The contract is therefore pinned as data. The converter emits
 * tests/fixtures/qwen38_int4_gs64.bin (regenerate with
 *   python3 tools/convert_qwen38_int4.py --emit-fixture tests/fixtures/qwen38_int4_gs64.bin
 * ) holding the packed nibbles, the group scales, an activation, and the y it
 * expects. This replays them through the real kernel. If either side ever
 * changes its nibble convention (offset binary, u = q + 8), its scale layout
 * ([O][I/gs] row-major) or its group stride, the numbers stop matching here.
 *
 * The two failure modes this is built to catch, both silent otherwise:
 *   1. two's complement instead of offset binary — qwen36's expert container
 *      uses it, the two differ by XOR 0x8, and neither side would crash;
 *   2. scales indexed [I/gs][O] instead of [O][I/gs] — right size, wrong map. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../quant.h"

#define FIXTURE_MAGIC 0x51333849   /* "Q38I" */

static int fail = 0;
static void check(int cond, const char *what){
    printf("%s %s\n", cond ? "ok  " : "FAIL", what);
    if(!cond) fail = 1;
}

static void *slurp(const char *path, size_t *len){
    FILE *f = fopen(path, "rb");
    if(!f){ fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n);
    if(fread(p, 1, (size_t)n, f) != (size_t)n){ fprintf(stderr, "short read\n"); exit(2); }
    fclose(f); *len = (size_t)n; return p;
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1] : "tests/fixtures/qwen38_int4_gs64.bin";
    size_t len; uint8_t *blob = slurp(path, &len);

    int32_t hdr[5]; memcpy(hdr, blob, sizeof hdr);
    check(hdr[0] == FIXTURE_MAGIC, "fixture magic");
    int O = hdr[1], I = hdr[2], S = hdr[3], gs = hdr[4];
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;

    size_t off = sizeof hdr;
    size_t n_q4 = (size_t)O * rb, n_sc = (size_t)O * ng;
    size_t n_x = (size_t)S * I, n_y = (size_t)S * O;
    size_t want = off + n_q4 + (n_sc + n_x + n_y) * sizeof(float);
    check(len == want, "fixture size matches its header");
    if(fail) return 1;

    const uint8_t *q4 = blob + off;                          off += n_q4;
    const float *scale = (const float *)(blob + off);        off += n_sc * sizeof(float);
    const float *x = (const float *)(blob + off);            off += n_x * sizeof(float);
    const float *yref = (const float *)(blob + off);

    float *y = malloc(n_y * sizeof(float));
    matmul_i4_grouped(y, x, q4, scale, S, I, O, gs);

    /* Tolerance is about summation order, not about quantization: both sides
     * multiply the SAME dequantized weights by the same activations, so only
     * f32 accumulation order differs. A wrong nibble or scale mapping misses by
     * order 1, four decades above this. */
    double worst = 0, norm = 0;
    for(size_t i = 0; i < n_y; i++){
        double d = fabs((double)y[i] - (double)yref[i]);
        if(d > worst) worst = d;
        norm += (double)yref[i] * (double)yref[i];
    }
    norm = sqrt(norm / (double)n_y);
    printf("     O=%d I=%d S=%d gs=%d  rms(yref)=%.4f  max|dy|=%.3e\n",
           O, I, S, gs, norm, worst);
    check(worst < 1e-3 * (norm + 1.0), "matmul_i4_grouped reproduces the converter's y");

    /* Counterproof 1: read the nibbles as two's complement, i.e. qwen36's
     * convention, by flipping bit 3 of every nibble. If THIS still passed, the
     * test above would be measuring nothing. */
    uint8_t *flip = malloc(n_q4);
    for(size_t i = 0; i < n_q4; i++) flip[i] = q4[i] ^ 0x88;
    matmul_i4_grouped(y, x, flip, scale, S, I, O, gs);
    double worst_flip = 0;
    for(size_t i = 0; i < n_y; i++){
        double d = fabs((double)y[i] - (double)yref[i]);
        if(d > worst_flip) worst_flip = d;
    }
    printf("     counterproof two's complement: max|dy|=%.3e\n", worst_flip);
    check(worst_flip > 1e-2 * norm, "the wrong nibble convention is actually caught");

    /* Counterproof 2: transpose the scale map to [I/gs][O]. Same byte count,
     * same values, wrong assignment — the mistake a reshape typo produces. */
    if(ng > 1 && O > 1){
        float *sw = malloc(n_sc * sizeof(float));
        for(int o = 0; o < O; o++)
            for(int g = 0; g < ng; g++) sw[(size_t)g * O + o] = scale[(size_t)o * ng + g];
        matmul_i4_grouped(y, x, q4, sw, S, I, O, gs);
        double worst_sw = 0;
        for(size_t i = 0; i < n_y; i++){
            double d = fabs((double)y[i] - (double)yref[i]);
            if(d > worst_sw) worst_sw = d;
        }
        printf("     counterproof transposed scales: max|dy|=%.3e\n", worst_sw);
        check(worst_sw > 1e-2 * norm, "the wrong scale layout is actually caught");
        free(sw);
    }

    free(flip); free(y); free(blob);
    printf(fail ? "FAILED\n" : "PASS\n");
    return fail;
}
