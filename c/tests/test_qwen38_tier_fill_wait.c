/* q38t_fill_wait() returns before the last upload has landed.
 *
 * The defect (upstream #1360's class, fixed in qwen36_tier by df44b87):
 * q38t_fill_wait waited for the upload QUEUE to drain (G.qn == 0). The
 * uploader frees the ring slot when it DEQUEUES an entry, before it calls
 * the backend, so the queue is empty while the last expert is still being
 * copied to the device. Whoever called q38t_fill_wait then sees that expert
 * as queued=1, resident=0 -- the warmstart counts it a miss on the first
 * token: a redundant disk read and a wasted offer.
 *
 * This test makes the race deterministic instead of probabilistic: the fake
 * backend's upload hook sleeps a few milliseconds per tensor, so the uploader
 * is always mid-copy when the queue empties. With the old wait the last
 * expert is caught queued=1 every run; with the fix (an in-flight count that
 * the uploader decrements only after the slot is resident) q38t_fill_wait
 * blocks until every enqueued expert has actually landed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qwen38_fake_cuda.h"

#include "../qwen38_tier.c"

#include "../compat.h"   /* setenv: MinGW has none, and qwen38_tier.c does not
                          * pull it in the way an engine .c does */

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* 3 ms per tensor, 9 ms per expert: long against the microseconds the
 * dequeue-to-upload gap takes, short against the test budget. */
static void slow_upload(int fmt) {
    (void)fmt;
    struct timespec ts = {0, 3000000};
    nanosleep(&ts, NULL);
}

int main(void) {
    enum { NL = 1, NE = 8, D = 64, IH = 32, TOPK = 2, SC = 2 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    fake_free_bytes = 8ull << 30;   /* above Q38T_DEV_RESERVE, or the budget is 0 */
    fake_upload_hook = slow_upload;

    if (!q38t_init(NL, NE, D, IH, TOPK, SC, 1 /* native_fp8 */, 8 /* fmt */, 0 /* gs */)) {
        printf("  FAIL: the tier should start on the fake backend\n");
        return 1;
    }

    static unsigned char g[NE][D * IH], u[NE][D * IH], d[NE][D * IH];
    static float sc[NE][3 * SC];
    for (int eid = 0; eid < NE; eid++) {
        memset(g[eid], (unsigned char)(eid + 1), sizeof g[eid]);
        memset(u[eid], (unsigned char)(eid + 2), sizeof u[eid]);
        memset(d[eid], (unsigned char)(eid + 3), sizeof d[eid]);
        for (int i = 0; i < 3 * SC; i++) sc[eid][i] = 1.0f;
        q38t_offer(0, eid, g[eid], u[eid], d[eid], sc[eid], 0);
    }

    q38t_fill_wait();

    /* The property, read under the tier's own lock the instant the wait
     * returns: nothing is still queued, and everything we enqueued is
     * resident. No polling, no sleeping -- the wait IS the guarantee. */
    int queued = 0, resident = 0;
    pthread_mutex_lock(&G.mx);
    for (int eid = 0; eid < NE; eid++) { queued += qs(0, eid)->queued; resident += qs(0, eid)->resident; }
    pthread_mutex_unlock(&G.mx);
    check(queued == 0, "an expert is still queued when q38t_fill_wait returns");
    check(resident == NE, "not every enqueued expert is resident when q38t_fill_wait returns");
    check(fake_uploads == 3 * NE, "each expert should have uploaded exactly three tensors by the time the wait returns");

    q38t_shutdown();

    if (fails) { printf("test_qwen38_tier_fill_wait: %d failures\n", fails); return 1; }
    printf("test_qwen38_tier_fill_wait: ok\n");
    return 0;
}
