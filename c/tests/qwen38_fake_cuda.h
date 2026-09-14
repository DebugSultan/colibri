/* qwen38_fake_cuda.h -- fake CUDA backend shared by the qwen38 tier tests.
 *
 * Same shape as qwen36_fake_cuda.h: defines every coli_cuda_* symbol
 * qwen38_tier.c links against (signatures from backend_cuda.h, which the
 * tier includes on its own) and RECORDS what it receives, so a test can
 * assert on real upload traffic without a GPU or the CUDA toolkit. A test
 * that only checked "q38t_init returns 1" would pass even with the tier
 * fully broken.
 *
 * Settable hooks:
 *   fake_ndev        - device count returned by coli_cuda_available_device_count
 *                      and coli_cuda_device_count (default 1).
 *   fake_free_bytes  - what coli_cuda_mem_info reports as free (default 2 GiB;
 *                      the tier's Q38T_DEV_RESERVE is 3.5 GiB, so tests that
 *                      want a nonzero budget must raise this).
 *   fake_upload_hook - called at the start of every tensor upload, on the
 *                      uploader thread, with the tensor's fmt. A test that
 *                      needs an upload to take TIME (a real cudaMemcpy does)
 *                      sleeps here; NULL (the default) uploads instantly. */
#ifndef QWEN38_FAKE_CUDA_H
#define QWEN38_FAKE_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../backend_cuda.h"

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; };

static int fake_uploads;
static int last_fmt = -1;
static size_t last_bytes;
static unsigned char captured[4096];
static size_t captured_len;

static int fake_ndev = 1;
static size_t fake_free_bytes = 2ull << 30;    /* what coli_cuda_mem_info reports as free */
static void (*fake_upload_hook)(int fmt) = NULL;

static int upload_common(ColiCudaTensor **t, const void *w, int fmt,
                         int I, int O, int device, int gs) {
    if (fake_upload_hook) fake_upload_hook(fmt);
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w;
    *t = n;
    fake_uploads++;
    last_fmt = fmt;
    last_bytes = (size_t)I * O;
    if (fake_uploads == 1) {
        captured_len = last_bytes < sizeof captured ? last_bytes : sizeof captured;
        memcpy(captured, w, captured_len);
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    (void)s; return upload_common(t, w, fmt, I, O, device, 0);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) { free(t); }
int coli_cuda_available_device_count(void) { return fake_ndev; }
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
static int fake_lut_published;
int coli_cuda_fp8_set_lut(const float *lut) { fake_lut_published = lut != NULL; return lut != NULL; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = fake_free_bytes; *total = 8ull << 30;
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    (void)u; (void)d; (void)rows; (void)count; (void)x;
    return 0;
}
const float *coli_cuda_expert_group_take(int device) { (void)device; return NULL; }
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}

#endif /* QWEN38_FAKE_CUDA_H */
