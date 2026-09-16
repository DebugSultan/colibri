/* dev_reserve_for: the per-device reserve parser, exercised without a GPU.
 *
 * Including the translation unit is deliberate -- the function is static and the
 * point of the test is the real parser, not a copy of it that can drift. No CUDA
 * entry point is called, so this runs on a machine whose cards are busy. */
#include "../qwen38_tier.c"

static int fails = 0;

static void expect(const char *env, int device, double want_mib, const char *what){
    if(env) setenv("Q38T_DEV_RESERVE_MB", env, 1);
    else    unsetenv("Q38T_DEV_RESERVE_MB");
    size_t got = dev_reserve_for(device);
    double got_mib = (double)got / (1024.0*1024.0);
    int ok = got_mib > want_mib - 0.001 && got_mib < want_mib + 0.001;
    if(!ok) fails++;
    printf("%-5s %-14s dev%d -> %8.1f MiB (want %8.1f)  %s\n",
           ok ? "ok" : "FAIL", env ? env : "(unset)", device, got_mib, want_mib, what);
}

int main(void){
    const double dflt = (double)Q38T_DEV_RESERVE / (1024.0*1024.0);

    /* 1. Unset and malformed input must both land on the compiled-in default:
     *    a reserve that silently becomes zero would let the arena eat the card. */
    expect(NULL,  0, dflt, "unset -> default");
    expect("",    0, dflt, "empty -> default");
    expect("0",   0, dflt, "zero is not a reserve -> default");
    expect("abc", 0, dflt, "garbage -> default");

    /* 2. A single value keeps today's behaviour exactly: same number everywhere. */
    expect("2048", 0, 2048.0, "single value, dev0");
    expect("2048", 1, 2048.0, "single value, dev1");
    expect("2048", 7, 2048.0, "single value, device past the pair");

    /* 3. A list is indexed by device id -- this is the whole feature: a big dense
     *    budget on the strong card, a packed arena on the weak one. */
    expect("1024,6144", 0, 1024.0, "list, dev0");
    expect("1024,6144", 1, 6144.0, "list, dev1");
    expect("1024,6144", 2, dflt,   "list, device past the end -> default");

    /* 4. Shape robustness: spaces, fractional MiB, a hole in the middle. */
    expect(" 512 , 8192 ", 1, 8192.0, "spaces around values");
    expect("1536.5,2048",  0, 1536.5, "fractional MiB");
    expect("1024,0,2048",  1, dflt,   "zero entry falls back, neighbours unaffected");
    expect("1024,0,2048",  2, 2048.0, "third entry still reachable");

    printf("\n%s: %d failure(s)\n", fails ? "FAILED" : "PASS", fails);
    return fails ? 1 : 0;
}
