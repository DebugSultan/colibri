/* qwen38 tier invariants on the fake backend, no GPU: the accounting charges
 * at cudaMalloc granularity, the budget is min(requested, measured headroom),
 * the planner's singular COLI_GPU selects a device, planned reservations are
 * released by q38t_cancel_plan, and the byte bookkeeping stays inside the
 * budget on one and on two devices. Ported from the qwen36 tier test family
 * (93c981b) to the offer-driven tier. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen38_fake_cuda.h"

#include "../qwen38_tier.c"

#include "../compat.h"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 1, NE = 8, D = 64, IH = 32, TOPK = 2, SC = 2 };
/* The int4 arm: gs has to divide both axes, and scale_count is derived from
 * the geometry (D*IH/gs), never declared -- q38t_init refuses a mismatch. */
enum { GS4 = 8, SC4 = D * IH / GS4 };

static unsigned char g[NE][D * IH], u[NE][D * IH], d[NE][D * IH];
static float sc[NE][3 * SC];
static float sc4[NE][3 * SC4];

static void seed(void) {
    for (int eid = 0; eid < NE; eid++) {
        memset(g[eid], (unsigned char)(eid + 1), sizeof g[eid]);
        memset(u[eid], (unsigned char)(eid + 2), sizeof u[eid]);
        memset(d[eid], (unsigned char)(eid + 3), sizeof d[eid]);
        for (int i = 0; i < 3 * SC; i++) sc[eid][i] = 1.0f;
        for (int i = 0; i < 3 * SC4; i++) sc4[eid][i] = 1.0f;
    }
}

static int resident_count(int layer) {
    int r = 0;
    for (int eid = 0; eid < NE; eid++) r += qs(layer, eid)->resident;
    return r;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    seed();

    /* --- 1. accounting: the charge is the cudaMalloc footprint ----------- */
    setenv("COLI_GPUS", "0", 1);
    unsetenv("CUDA_EXPERT_GB");
    fake_ndev = 1;
    fake_free_bytes = 8ull << 30;
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: the tier should start on the fake backend\n");
        return 1;
    }
    size_t payload = 3u * (D * IH) + 3u * SC * sizeof(float);
    size_t want_exp = 3u * dev_alloc_footprint((size_t)D * IH)
                    + 3u * dev_alloc_footprint(SC * sizeof(float));
    check(G.exp_bytes == want_exp, "exp_bytes is not the footprint sum");
    check(G.exp_bytes > payload, "the footprint charge does not exceed the payload");
    check(G.exp_bytes != payload + 4096, "exp_bytes still uses the old flat-slack formula");
    check(G.budget[0] == fake_free_bytes - Q38T_DEV_RESERVE,
          "an auto budget is not the measured headroom");
    q38t_shutdown();

    /* --- 2. an explicit budget is clamped to the headroom ---------------- */
    setenv("CUDA_EXPERT_GB", "100", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an over-large explicit budget\n");
        return 1;
    }
    check(G.budget[0] == fake_free_bytes - Q38T_DEV_RESERVE,
          "an explicit budget above the headroom was not clamped to it");
    q38t_shutdown();
    setenv("CUDA_EXPERT_GB", "1", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an explicit budget below the headroom\n");
        return 1;
    }
    check(G.budget[0] == (size_t)1 << 30, "an explicit budget under the headroom was not honoured");
    q38t_shutdown();
    unsetenv("CUDA_EXPERT_GB");

    /* --- 3. COLI_GPU, the planner's singular, selects a device ----------- */
    unsetenv("COLI_GPUS");
    setenv("COLI_GPU", "0", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with the singular COLI_GPU\n");
        return 1;
    }
    check(G.ndev == 1 && G.dev[0] == 0, "COLI_GPU=0 did not select device 0");
    q38t_shutdown();
    setenv("COLI_GPUS", "0", 1);
    unsetenv("COLI_GPU");

    /* --- 4. workload: bytes stay inside the budget, plans can be cancelled */
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init before the workload section\n");
        return 1;
    }
    int u0 = fake_uploads;
    for (int eid = 0; eid < NE; eid++)
        q38t_offer(0, eid, g[eid], u[eid], d[eid], sc[eid], 0);
    q38t_fill_wait();
    check(fake_uploads - u0 == 3 * NE, "the uploads did not all complete");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == (size_t)NE * G.exp_bytes, "used[0] does not match eight resident experts");
    check(G.used[0] <= G.budget[0], "used[0] exceeds the budget");
    pthread_mutex_unlock(&G.mx);
    check(resident_count(0) == NE, "not every offered expert is resident");
    q38t_shutdown();

    /* --- 4b. plans: reserve, cancel releases, the rest fills --------------
     * plan_fill needs a HEAT_FILE: without one it deliberately invents no
     * order and lets the traffic promote. */
    const char *tmp_heat = "q38t_test_heat.tmp";
    FILE *f = fopen(tmp_heat, "wb");
    if (!f) { printf("  FAIL: cannot write the temporary heat file\n"); return 1; }
    uint32_t hdr[3] = {0x51544831u, NL, NE};
    uint32_t heat[NE];
    for (int i = 0; i < NE; i++) heat[i] = (uint32_t)(NE - i);
    fwrite(hdr, 4, 3, f);
    fwrite(heat, 4, NE, f);
    fclose(f);
    setenv("HEAT_FILE", tmp_heat, 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init before the plan section\n");
        return 1;
    }
    int L[NE], E[NE];
    int n = q38t_plan_fill(L, E, NE);
    check(n == NE, "the plan should cover all experts (the budget dwarfs them)");
    q38t_cancel_plan(L[0], E[0]);   /* the engine changed its mind about one */
    for (int i = 1; i < n; i++)
        q38t_offer(L[i], E[i], g[E[i]], u[E[i]], d[E[i]], sc[E[i]], 1);
    q38t_fill_wait();
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == (size_t)(n - 1) * G.exp_bytes,
          "the cancelled reservation was not released");
    check(G.used[0] <= G.budget[0], "planned offers pushed used[] past the budget");
    pthread_mutex_unlock(&G.mx);
    check(resident_count(0) == n - 1, "the planned residents do not match the plan minus the cancellation");
    q38t_shutdown();
    unsetenv("HEAT_FILE");
    remove(tmp_heat);

    /* --- 5. two devices: the home split halves the bookkeeping ----------- */
    fake_ndev = 2;
    setenv("COLI_GPUS", "0,1", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init on two fake devices\n");
        return 1;
    }
    for (int eid = 0; eid < NE; eid++)
        q38t_offer(0, eid, g[eid], u[eid], d[eid], sc[eid], 0);
    q38t_fill_wait();
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == (size_t)(NE / 2) * G.exp_bytes && G.used[1] == (size_t)(NE / 2) * G.exp_bytes,
          "the eid%%ndev home split did not halve the per-device bytes");
    pthread_mutex_unlock(&G.mx);
    q38t_shutdown();
    fake_ndev = 1;
    setenv("COLI_GPUS", "0", 1);

    /* --- 6. the int4 arm: half the weight bytes, and gs reaches the backend
     * ---------------------------------------------------------------------
     * A tier staging int4 while telling the backend fmt=8 would read the
     * nibbles as e4m3 bytes and produce plausible-looking garbage, so the
     * assertions below are on what the uploader actually handed over: the
     * format, the group size, and the byte count (nibble-packed, so half).
     * The fp8 exp_bytes measured in section 1 is the number to beat -- the
     * whole point of the arm is more resident experts per GB. */
    {
        /* native_fp8 = 0: an int4 container does not need the FP8 path at all,
         * and the tier used to refuse to start without it. */
        if (!q38t_init(NL, NE, D, IH, TOPK, SC4, 0, 4, GS4)) {
            printf("  FAIL: the int4 tier should start without native fp8\n");
            return 1;
        }
        check(G.fmt == 4 && G.gs == GS4, "the tier did not record fmt=4/gs");
        check(G.mat_bytes == (size_t)D * IH / 2, "int4 matrix bytes are not nibble-packed");
        check(G.exp_bytes == 3u * dev_alloc_footprint((size_t)D * IH / 2)
                           + 3u * dev_alloc_footprint(SC4 * sizeof(float)),
              "int4 exp_bytes is not the nibble+scale footprint sum");

        int u4 = fake_uploads;
        last_fmt = -1; last_gs = -1; last_bytes = 0;
        for (int eid = 0; eid < NE; eid++)
            q38t_offer(0, eid, g[eid], u[eid], d[eid], sc4[eid], 0);
        q38t_fill_wait();
        check(fake_uploads - u4 == 3 * NE, "the int4 uploads did not all complete");
        check(last_fmt == 4, "the uploader announced the wrong format to the backend");
        check(last_gs == GS4, "the group size never reached the backend (per-row int4 instead)");
        check(last_bytes == (size_t)D * IH / 2, "the backend was handed fp8-sized weights");
        check(resident_count(0) == NE, "not every int4 expert is resident");
        q38t_shutdown();

        /* The VRAM saving is asserted at the production geometry, not at the
         * toy one above: 64*32 bytes rounds to the same cudaMalloc granule
         * whether it is halved or not, so the toy shapes cannot show it. */
        {
            size_t fp8_mat = (size_t)2560 * 640, i4_mat = fp8_mat / 2;
            check(dev_alloc_footprint(i4_mat) * 2 == dev_alloc_footprint(fp8_mat),
                  "halving the weight bytes did not halve the VRAM footprint at 2560x640");
        }

        /* Refusals: the geometry is checked, not trusted. */
        check(!q38t_init(NL, NE, D, IH, TOPK, SC4, 1, 5, GS4),
              "an unknown format was accepted");
        check(!q38t_init(NL, NE, D, IH, TOPK, SC4, 1, 4, 7),
              "a group size that divides neither axis was accepted");
        check(!q38t_init(NL, NE, D, IH, TOPK, SC4 + 1, 1, 4, GS4),
              "a scale count inconsistent with D*IH/gs was accepted");
    }

    if (fails) { printf("test_qwen38_tier: %d failures\n", fails); return 1; }
    printf("test_qwen38_tier: ok\n");
    return 0;
}
