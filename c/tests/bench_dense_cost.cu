/* fmt=9 dense per-call cost oracle -- splits the 89 ms/token that is NOT bandwidth.
 *
 * The decode profile says resident-mm costs ~95-100 ms/token across N=193 dense
 * calls (derived from two agreeing counters in a profiled run: solving
 * N*(prompt+64)=1509760 and N*(chunks+64)=43259 gives N=193, prompt=7758,
 * chunks=160). That is 0.50 ms per call against ~11 ms/token of theoretical
 * PCIe/VRAM traffic, so ~89 ms/token -- 30% of the token -- is something else.
 *
 * Three candidate terms, and they call for three different fixes:
 *   a) per-call fixed cost  -> two blocking cudaMemcpy + launch + sync
 *                              (coli_cuda_matmul, backend_cuda.cu:1823).
 *                              This is the ONLY term CUDA graphs can remove.
 *   b) bandwidth            -> unavoidable, it is the weight read itself.
 *   c) per-output overhead  -> quant_matmul (backend_cuda.cu:475) sends fmt=9
 *                              into the generic branch: one 256-thread block per
 *                              (o,s) with an 8-barrier tree reduction. At S=1
 *                              with I=2560 each thread does ~10 FMAs, so the
 *                              reduction costs as much as the work; and the 100
 *                              [320,10240] weights launch 320 blocks on 70 SMs.
 *                              This is what a warp-level kernel (the fmt=8
 *                              quant_matmul_f8w rework, backend_cuda.cu:447)
 *                              would remove.
 *
 * The model is therefore  t = a + b*(I*O) + c*O,  fitted by least squares over
 * the REAL shapes (from the checkpoint safetensors headers) plus extreme probes
 * chosen to make the fit identifiable:
 *   [64,10240] and [10240,64] carry the SAME I*O (655360) but O differs 160x,
 *   so they separate b from c on their own; [1,64] anchors a.
 *
 * Deliberately NOT done: an S=1 vs S=8 comparison. Both hypotheses predict the
 * same result there -- the weights are read once regardless of S -- so it
 * falsifies nothing.
 *
 * The oracle then reconstructs the per-token dense cost from the fitted terms
 * over the real shape census and prints the split in ms/token. If the
 * reconstruction lands far below the measured ~95-100 ms, the residual is a
 * FOURTH term the model does not carry (stream contention with the expert tier,
 * which ran 59.7 s on GPU in the same prefill) -- and that is a finding, not a
 * failure of the fit.
 *
 * Shapes are [O,I] (torch out_features, in_features), matching the safetensors
 * header order. Counts are the real multiplicities of the 424 two-dimensional
 * BF16 non-routed weights >= 4 MiB, 9.449 GiB total.
 *
 * Needs a free card: allocates up to ~1.2 GiB for the lm_head shape (skipped,
 * with a note, if the upload is refused). Build is CPU-only and always allowed:
 *   nvcc -O2 -std=c++17 -arch=native tests/bench_dense_cost.cu -o dense_cost_bench
 * Env: COLI_DENSE_BENCH_DEV (default 0 = the 5070 Ti, CUDA enumerates
 *      fastest-first), COLI_DENSE_BENCH_REPS (default 21, odd -> true median).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    int O, I;
    int count;          /* multiplicity in the checkpoint; 0 = synthetic probe */
    const char *note;
} Shape;

/* The 14 real shapes + 3 probes. Real counts sum to 424. */
static Shape shapes[] = {
    {   320, 10240, 100, "deltanet in_proj (small O, wide I)" },
    { 10240,   320, 100, "deltanet out_proj (wide O, small I)" },
    {  2560,  6144,  49, "" },
    { 10240,  2560,  37, "" },
    {  6144,  2560,  36, "" },
    {  3456,  1152,  27, "" },
    {  4304,  1152,  27, "" },
    {  1152,  4304,  27, "" },
    { 12288,  2560,  13, "" },
    {  2560,  2560,   3, "" },
    {248320,  2560,   2, "lm_head (1212.5 MiB each)" },
    {  4608,  4608,   1, "" },
    {  2560,  4608,   1, "" },
    {  2304,  1152,   1, "" },
    {    64, 10240,   0, "PROBE: same I*O as the next, O 160x smaller" },
    { 10240,    64,   0, "PROBE: same I*O as the previous, O 160x larger" },
    {    64,    64,   0, "PROBE: anchors the fixed per-call cost a" },
};
static const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

/* Solve the 3x3 normal equations by Gaussian elimination with partial pivoting.
 * Columns are pre-normalized by the caller, so the system stays conditioned
 * even though I*O spans 4096 .. 635M. */
static int solve3(double A[3][3], double b[3], double out[3]) {
    for (int col = 0; col < 3; col++) {
        int piv = col;
        for (int r = col + 1; r < 3; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-18) return 0;
        if (piv != col) {
            for (int k = 0; k < 3; k++) { double t = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < 3; r++) {
            double f = A[r][col] / A[col][col];
            for (int k = col; k < 3; k++) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int r = 2; r >= 0; r--) {
        double s = b[r];
        for (int k = r + 1; k < 3; k++) s -= A[r][k] * out[k];
        out[r] = s / A[r][r];
    }
    return 1;
}

int main(void) {
    const char *e;
    int dev = (e = getenv("COLI_DENSE_BENCH_DEV")) ? atoi(e) : 0;
    int reps = (e = getenv("COLI_DENSE_BENCH_REPS")) ? atoi(e) : 21;
    if (reps < 3) reps = 3;

    if (!coli_cuda_init((int[]){dev}, 1)) { printf("FAIL cuda init (dev %d)\n", dev); return 1; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("device %d: %s, SM=%d\n", dev, prop.name, prop.multiProcessorCount);
    printf("reps=%d (median reported), S=1 (decode)\n\n", reps);

    /* rand() is left unseeded on purpose: the weight patterns are identical from
     * run to run, so two runs of this bench differ only in machine state. */
    double *ts = (double *)malloc(sizeof(double) * reps);
    double med[64], measured[64];
    int ok[64];
    memset(ok, 0, sizeof(ok));

    printf("%7s %7s %5s %9s %9s %9s  %s\n",
           "O", "I", "n", "MiB", "med_ms", "GB/s", "note");
    for (int s = 0; s < nshape; s++) {
        int O = shapes[s].O, I = shapes[s].I;
        size_t wb = (size_t)O * I * 2;
        uint16_t *w = (uint16_t *)malloc(wb);
        float *x = (float *)malloc((size_t)I * 4);
        float *y = (float *)malloc((size_t)O * 4);
        if (!w || !x || !y) { printf("  [skip %dx%d: host alloc]\n", O, I); free(w); free(x); free(y); continue; }
        /* Real-magnitude bf16 patterns: random 16-bit noise would be mostly
         * inf/NaN, and denormal/NaN traffic is not what the model ships. */
        for (size_t i = 0; i < (size_t)O * I; i++) {
            float v = (rand() / (float)RAND_MAX - .5f) * 2.f;
            uint32_t u; memcpy(&u, &v, 4); w[i] = (uint16_t)(u >> 16);
        }
        for (int i = 0; i < I; i++) x[i] = (rand() / (float)RAND_MAX - .5f) * 2.f;

        ColiCudaTensor *t = NULL;
        /* First call uploads; it is staging cost, excluded from the timing. */
        if (!coli_cuda_matmul(&t, y, x, w, NULL, 9, 1, I, O, dev, 0)) {
            printf("%7d %7d %5d %9.1f %9s %9s  UPLOAD REFUSED (VRAM?)\n",
                   O, I, shapes[s].count, wb / 1048576.0, "-", "-");
            free(w); free(x); free(y); continue;
        }
        for (int r = 0; r < 3; r++) coli_cuda_matmul(&t, y, x, w, NULL, 9, 1, I, O, dev, 0);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            coli_cuda_matmul(&t, y, x, w, NULL, 9, 1, I, O, dev, 0);
            ts[r] = now_ms() - t0;
        }
        qsort(ts, reps, sizeof(double), cmp_double);
        med[s] = ts[reps / 2];
        measured[s] = med[s];
        ok[s] = 1;
        printf("%7d %7d %5d %9.1f %9.4f %9.1f  %s\n",
               O, I, shapes[s].count, wb / 1048576.0, med[s],
               wb / (med[s] * 1e-3) / 1e9, shapes[s].note);

        coli_cuda_tensor_free(t);
        free(w); free(x); free(y);
    }
    free(ts);

    /* ---- the fit: t = a + b*(I*O) + c*O ---- */
    double sc_io = 0, sc_o = 0;
    for (int s = 0; s < nshape; s++) if (ok[s]) {
        double io = (double)shapes[s].I * shapes[s].O;
        if (io > sc_io) sc_io = io;
        if (shapes[s].O > sc_o) sc_o = shapes[s].O;
    }
    if (sc_io <= 0) { printf("\nFAIL: no shape measured\n"); return 1; }

    double A[3][3] = {{0}}, rhs[3] = {0}, sol[3] = {0};
    int n = 0;
    for (int s = 0; s < nshape; s++) if (ok[s]) {
        double f[3] = { 1.0, (double)shapes[s].I * shapes[s].O / sc_io, shapes[s].O / sc_o };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) A[i][j] += f[i] * f[j];
            rhs[i] += f[i] * measured[s];
        }
        n++;
    }
    if (n < 3 || !solve3(A, rhs, sol)) { printf("\nFAIL: fit is singular (%d shapes)\n", n); return 1; }

    double a = sol[0], b = sol[1] / sc_io, c = sol[2] / sc_o;   /* ms, ms/element, ms/row */

    printf("\n---- fit over %d shapes:  t = a + b*(I*O) + c*O ----\n", n);
    printf("  a = %.4f ms        per-call fixed cost (two blocking memcpy + launch + sync)\n", a);
    printf("  b = %.3f ns/elem   -> %.1f GB/s effective on the 2-byte weight read\n",
           b * 1e6, 2.0 / (b * 1e-3) / 1e9);
    printf("  c = %.3f us/row    per-output-row reduction overhead\n", c * 1e3);

    double worst = 0;
    printf("  residuals (measured - model), ms:\n");
    for (int s = 0; s < nshape; s++) if (ok[s]) {
        double pred = a + b * (double)shapes[s].I * shapes[s].O + c * shapes[s].O;
        double res = measured[s] - pred;
        if (fabs(res) > fabs(worst)) worst = res;
        printf("    [%6d,%6d] meas %8.4f  model %8.4f  res %+8.4f\n",
               shapes[s].O, shapes[s].I, measured[s], pred, res);
    }
    printf("  worst residual: %+.4f ms\n", worst);

    /* ---- reconstruction: what the fit says a decode token costs ---- */
    double ta = 0, tb = 0, tc = 0;
    int calls = 0;
    for (int s = 0; s < nshape; s++) {
        if (!shapes[s].count) continue;
        if (!ok[s]) { printf("\n  NOTE: [%d,%d] x%d was not measured -- reconstruction is a LOWER bound\n",
                             shapes[s].O, shapes[s].I, shapes[s].count); continue; }
        ta += shapes[s].count * a;
        tb += shapes[s].count * b * (double)shapes[s].I * shapes[s].O;
        tc += shapes[s].count * c * shapes[s].O;
        calls += shapes[s].count;
    }
    printf("\n---- reconstruction over the full census (%d weights) ----\n", calls);
    printf("  fixed per-call (a) : %7.2f ms   <- the ONLY part CUDA graphs can remove\n", ta);
    printf("  bandwidth      (b) : %7.2f ms   <- irreducible, it is the weight read\n", tb);
    printf("  per-row        (c) : %7.2f ms   <- what a warp-level fmt=9 kernel removes\n", tc);
    printf("  model total        : %7.2f ms\n", ta + tb + tc);
    printf("\n  READ THIS BEFORE COMPARING TO 95-100 ms.\n");
    printf("  The census is the %d two-dimensional BF16 non-routed weights >= 4 MiB in the\n", calls);
    printf("  safetensors headers. The ENGINE stages 193 of them and refuses 136 (the rest\n");
    printf("  stay on CPU), and lm_head is counted twice here but is not run twice per token.\n");
    printf("  So the total above is an UPPER bound on the GPU dense cost, not the per-token\n");
    printf("  figure. The number to compare against ~95-100 ms/token over N=193 calls is the\n");
    printf("  same sum restricted to what a run actually stages -- take the staged list from\n");
    printf("  the [q38dense] eager line of the run and re-weight.\n");
    printf("  What IS directly usable regardless: the RATIO a : b : c, which is the split\n");
    printf("  the levers act on, and it does not depend on the multiplicities being exact.\n");
    printf("\n  ratio a : b : c = %.1f%% : %.1f%% : %.1f%%\n",
           100.0 * ta / (ta + tb + tc), 100.0 * tb / (ta + tb + tc), 100.0 * tc / (ta + tb + tc));
    printf("  If a+b+c lands far BELOW the measured decode cost even after re-weighting,\n"
           "  the residual is a fourth term the model does not carry -- stream contention\n"
           "  with the expert tier is the standing candidate (59.7 s on GPU in the same run).\n");

    printf("\nok\n");
    return 0;
}
