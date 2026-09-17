/* qwen38 tier invariants on the fake backend, no GPU: the accounting charges
 * the exact payload (the arena stores experts with no rounding), the budget is
 * min(requested, measured headroom) slot-aligned to the payload,
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
    for (int eid = 0; eid < NE; eid++) r += vt_qs(layer, eid)->resident;
    return r;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    seed();

    /* --- 1. accounting: the charge is the exact payload ------------------ */
    setenv("COLI_GPUS", "0", 1);
    unsetenv("CUDA_EXPERT_GB");
    fake_ndev = 1;
    fake_free_bytes = 8ull << 30;
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: the tier should start on the fake backend\n");
        return 1;
    }
    size_t payload = 3u * (D * IH) + 3u * SC * sizeof(float);
    check(VTG.exp_bytes == payload, "the charge is not the exact payload");
    check(VTG.budget[0] == (fake_free_bytes - Q38T_DEV_RESERVE) / payload * payload,
          "the arena budget is not slot-aligned to the headroom");
    q38t_shutdown();

    /* --- 2. an explicit budget is clamped to the headroom ---------------- */
    setenv("CUDA_EXPERT_GB", "100", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an over-large explicit budget\n");
        return 1;
    }
    check(VTG.budget[0] == (fake_free_bytes - Q38T_DEV_RESERVE) / payload * payload,
          "an explicit budget above the headroom was not clamped to it");
    q38t_shutdown();
    setenv("CUDA_EXPERT_GB", "1", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an explicit budget below the headroom\n");
        return 1;
    }
    check(VTG.budget[0] == ((size_t)1 << 30) / payload * payload,
          "an explicit budget under the headroom was not honoured");
    q38t_shutdown();
    unsetenv("CUDA_EXPERT_GB");

    /* --- 2b. the reserve is parametric (Q38T_DEV_RESERVE_MB) ---------------
     * The 3.5 GiB default was measured in decode; in prefill the backend
     * takes ~200 MB per card, so the breaking point is found by lowering
     * this stepwise. The knob must move the headroom, survive the explicit
     * budget clamp, and fall back to the default on junk. */
    setenv("Q38T_DEV_RESERVE_MB", "16", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with a parametric reserve\n");
        return 1;
    }
    check(VTG.budget[0] == (fake_free_bytes - ((size_t)16 << 20)) / payload * payload,
          "Q38T_DEV_RESERVE_MB did not move the auto budget");
    q38t_shutdown();

    setenv("CUDA_EXPERT_GB", "100", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an explicit budget over a parametric reserve\n");
        return 1;
    }
    check(VTG.budget[0] == (fake_free_bytes - ((size_t)16 << 20)) / payload * payload,
          "the explicit budget was not clamped to the parametric headroom");
    q38t_shutdown();
    unsetenv("CUDA_EXPERT_GB");

    setenv("Q38T_DEV_RESERVE_MB", "junk", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with a junk reserve override\n");
        return 1;
    }
    check(VTG.budget[0] == (fake_free_bytes - Q38T_DEV_RESERVE) / payload * payload,
          "a junk Q38T_DEV_RESERVE_MB did not fall back to the default");
    q38t_shutdown();
    unsetenv("Q38T_DEV_RESERVE_MB");

    /* --- 3. COLI_GPU, the planner's singular, selects a device ----------- */
    unsetenv("COLI_GPUS");
    setenv("COLI_GPU", "0", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with the singular COLI_GPU\n");
        return 1;
    }
    check(VTG.ndev == 1 && VTG.dev[0] == 0, "COLI_GPU=0 did not select device 0");
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
    pthread_mutex_lock(&VTG.mx);
    check(VTG.used[0] == (size_t)NE * VTG.exp_bytes, "used[0] does not match eight resident experts");
    check(VTG.used[0] <= VTG.budget[0], "used[0] exceeds the budget");
    pthread_mutex_unlock(&VTG.mx);
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
    pthread_mutex_lock(&VTG.mx);
    check(VTG.used[0] == (size_t)(n - 1) * VTG.exp_bytes,
          "the cancelled reservation was not released");
    check(VTG.used[0] <= VTG.budget[0], "planned offers pushed used[] past the budget");
    pthread_mutex_unlock(&VTG.mx);
    check(resident_count(0) == n - 1, "the planned residents do not match the plan minus the cancellation");
    q38t_shutdown();

    /* --- 4c. arena slots are recycled -------------------------------------
     * CUDA_EXPERT_GB tuned so the arena holds exactly NE payload slots:
     * a full plan, a full cancel, and a second full plan only succeeds if
     * every cancelled reservation gave its slot back. One leaked slot and
     * the second plan comes up short. */
    setenv("CUDA_EXPERT_GB", "0.00005", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init with an arena sized to exactly NE slots\n");
        return 1;
    }
    check(VTG.budget[0] >= (size_t)NE * VTG.exp_bytes && VTG.budget[0] < (size_t)(NE + 1) * VTG.exp_bytes,
          "the tuned budget did not land on exactly NE arena slots");
    int L1[NE], E1[NE];
    int n1 = q38t_plan_fill(L1, E1, NE);
    check(n1 == NE, "the first plan did not fill the arena");
    for (int i = 0; i < n1; i++) q38t_cancel_plan(L1[i], E1[i]);
    /* Direct on the free-list: plan_fill's cursor is single-shot, so a
     * second plan cannot re-cover the ground -- the free count is the truth. */
    check(VTG.slot_free_n[0] == (int)(VTG.budget[0] / VTG.exp_bytes),
          "the cancelled slots were not given back to the arena");
    check(VTG.used[0] == 0, "the cancelled charges were not released");
    q38t_shutdown();
    unsetenv("CUDA_EXPERT_GB");

    unsetenv("HEAT_FILE");
    remove(tmp_heat);

    /* --- 5. two devices: the vt_home split halves the bookkeeping ----------- */
    fake_ndev = 2;
    setenv("COLI_GPUS", "0,1", 1);
    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1, 8, 0)) {
        printf("  FAIL: init on two fake devices\n");
        return 1;
    }
    for (int eid = 0; eid < NE; eid++)
        q38t_offer(0, eid, g[eid], u[eid], d[eid], sc[eid], 0);
    q38t_fill_wait();
    pthread_mutex_lock(&VTG.mx);
    check(VTG.used[0] == (size_t)(NE / 2) * VTG.exp_bytes && VTG.used[1] == (size_t)(NE / 2) * VTG.exp_bytes,
          "the eid%%ndev vt_home split did not halve the per-device bytes");
    pthread_mutex_unlock(&VTG.mx);
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
        check(VTG.fmt == 4 && VTG.gs == GS4, "the tier did not record fmt=4/gs");
        check(VTG.mat_bytes == (size_t)D * IH / 2, "int4 matrix bytes are not nibble-packed");
        check(VTG.exp_bytes == 3u * ((size_t)D * IH / 2) + 3u * SC4 * sizeof(float),
              "int4 exp_bytes is not the nibble+scale payload");

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
