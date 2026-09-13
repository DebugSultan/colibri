/* omp_tune.h — sizing the OpenMP team on PHYSICAL CORES.
 *
 * WHY ONLY THE SIZING, AND NOT THE SPIN-WAIT.
 * The tuning block in colibri.c does two things with OPPOSITE risk profiles,
 * and they must be kept separate:
 *
 *   sizing           OMP_NUM_THREADS = physical cores (no SMT)
 *                    #718: +2.3x on Zen3 (5950X, 16C/32T) just by changing
 *                    the thread count. The gain is so large that it
 *                    "drowns out most of the deltas quoted here".
 *
 *   spin-wait        OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME
 *                    #707: -2.2x on the decode of a low-residency host
 *                          (M1 Max 32 GB, ~10% of experts resident)
 *                    #116: -39% on Metal      #159: ~3x on x86+CUDA
 *                    #341: 3000% of CPU on FreeBSD with the team idle
 *                    Mechanism: where a token is made of bytes from disk, a
 *                    team spinning in place steals cores from the I/O pool
 *                    that is doing the real work.
 *
 * Kimi K3 and OLMoE had NEITHER. Here they take only the first: Kimi is the
 * most disk-bound engine in the project (measured: 6.7% hit rate, 891 GB
 * read for 32 tokens), i.e. exactly the regime where the second half does
 * harm. Adding it would have been a measurable regression. GLM now calls
 * this same helper after its optional hot-team re-exec, so it receives the
 * physical-core sizing without changing the independent spin-wait policy
 * that its existing block controls.
 *
 * WHY NO RE-EXEC IS NEEDED HERE.
 * colibri.c re-executes itself because the CONSTRUCTOR of libgomp reads
 * OMP_WAIT_POLICY & co. before main(): a setenv() inside main() arrives too
 * late. The thread count is different: omp_set_num_threads() is a runtime
 * API and takes effect immediately. The safe half is also the simple half.
 *
 * RULE ON FAILURES: if the physical core count cannot be determined, do NOT
 * guess — leave the OpenMP default. A wrong count is worse than no count
 * (cf. #325, where a silent fallback to 1 pinned the decode to a single
 * core).
 */
#ifndef COLI_OMP_TUNE_H
#define COLI_OMP_TUNE_H

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <dirent.h>
#endif

/* Physical core count, or 0 if not determinable. Never an invented value. */
#if defined(_WIN32)
static int coli_count_windows_physical_cores(const void *buf, DWORD bytes)
{
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    const size_t header_size = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                                        Processor);
    int cores = 0;

    while ((size_t)(end - p) >= header_size) {
        LOGICAL_PROCESSOR_RELATIONSHIP relationship;
        DWORD record_size;
        memcpy(&relationship,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
               sizeof(relationship));
        memcpy(&record_size,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size),
               sizeof(record_size));
        if (record_size < header_size || (size_t)(end - p) < record_size)
            break; /* Reject a zero, truncated, or otherwise malformed record. */
        if (relationship == RelationProcessorCore) cores++;
        p += record_size;
    }
    return cores;
}
#endif

static int coli_physical_cores(void)
{
#if defined(_WIN32)
    DWORD need = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &need);
    if (!need) return 0;
    void *buf = malloc(need);
    if (!buf) return 0;
    int cores = GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &need)
                    ? coli_count_windows_physical_cores(buf, need)
                    : 0;
    free(buf);
    return cores;

#elif defined(__APPLE__)
    /* hw.perflevel0.logicalcpu = the PERFORMANCE cores. On Apple Silicon
     * hw.physicalcpu counts them all, E-cores included (10 on an M1 Max),
     * and with a matmul barrier the slowest thread sets the pace: the
     * E-cores slow the team down instead of helping (#707, -4.2% decode).
     * On Intel Macs perflevel* does not exist: there hw.physicalcpu is
     * correct. */
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;

#else
    /* Linux: a physical core = a distinct thread_siblings_list. Counting the
     * unique lists deduplicates SMT without interpreting the topology. */
    DIR *d = opendir("/sys/devices/system/cpu");
    if (!d) return 0;
    char seen[1024][64];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < 1024) {
        if (strncmp(e->d_name, "cpu", 3) != 0 || e->d_name[3] < '0' || e->d_name[3] > '9')
            continue;
        /* d_name can be up to 255 bytes; leave enough room for the fixed
         * sysfs prefix/suffix so -Wformat-truncation stays honest when this
         * shared helper is compiled into the GLM engine too. */
        char path[512], line[64];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/%s/topology/thread_siblings_list", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\n")] = 0;
            int dup = 0;
            for (int i = 0; i < n; i++) if (strcmp(seen[i], line) == 0) { dup = 1; break; }
            if (!dup) { snprintf(seen[n], sizeof(seen[0]), "%s", line); n++; }
        }
        fclose(f);
    }
    closedir(d);
    return n;
#endif
}

/* Size the OpenMP team on physical cores. Respects OMP_NUM_THREADS if the
 * user set it, and does nothing if the count is not reliable. `engine` ends
 * up only in the log line. */
static void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* same kill-switch as the other engines */
    if (getenv("OMP_NUM_THREADS")) return; /* the user commands */

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* unknown -> OpenMP default */
    int logical = omp_get_max_threads();
    if (phys >= logical) return;           /* no SMT to avoid: stay silent */

    omp_set_num_threads(phys);
    fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                    "SMT can halve decode throughput on some CPUs (#718); "
                    "set OMP_NUM_THREADS=<n> to override\n",
            engine, phys, logical);
#else
    (void)engine;
#endif
}

/* Size the team KEEPING SMT, minus a reserve of whole physical cores.
 * The OPPOSITE policy of coli_omp_tune_threads() above, and opposite for a
 * measured reason, not a taste: the two serve two different regimes.
 *
 *   physical cores   the regime of #718: int4 GEMV on already-resident
 *                    weights. The regions are tiny and back-to-back, two SMT
 *                    siblings contend for the same vector unit and the team
 *                    collapses (2.3x on Zen3).
 *
 *   SMT included     the regime of Qwen3.8-Flash-Next: FP8 expert matmuls
 *                    on pages JUST FAULTED from disk. The bottleneck is the
 *                    bandwidth to memory, not the vector unit, and there the
 *                    SMT sibling does not contend: it covers the other's
 *                    latency.
 *
 * The measurement that authorizes this (qwen38 campaign, Phase 0.4 of
 * 13/09/2026; minutes in qwen38cuda/PIANO_NEXT.md §9). Ryzen 9 3900X
 * 12C/24T, 64 GB, cap 128, prompt 274 tokens + 16 new = 290 forwards, every
 * step repeated:
 *
 *   OMP_NUM_THREADS   16 tokens (copy)   resident-mm   deltanet
 *     12 (physical)     128.5 s          3648 ms/fwd   2440 ms/fwd
 *     16                109.9 s          2792          1974
 *     20                 95.7 s  -25.5%  2296  -37.1%  1661  -31.9%
 *
 * The scaling efficiency stays ~97% up to 20: no saturation, which is
 * precisely the signature of the bandwidth-bound regime. `expert-read`
 * (disk) stays flat, 1108 -> 1087 ms/fwd: the whole gain is in the dense
 * CPU.
 *
 * WHY A RESERVE, AND WHY IN WHOLE CORES.
 * The step at 24 was not measured: excluded by the owner because a production
 * service runs next to it on this node, and saturating all the logical CPUs
 * would starve it. The reserve is counted in WHOLE physical cores (both
 * their SMT siblings) because leaving half a core leaves nothing: the
 * remaining sibling keeps contending for the same unit. With reserve_cores=2
 * on a 3900X: 24 - 2*2 = 20, which is the number measured above.
 *
 * FLOOR. Never go below the physical cores, i.e. below the current default:
 * on a host without SMT, or with few cores, this function must not be able
 * to make things worse than what is already there. The RULE ON FAILURES at
 * the top of the file also applies: count not determinable -> leave the
 * OpenMP default.
 */
static void coli_omp_tune_threads_smt(const char *engine, int reserve_cores)
{
#ifdef _OPENMP
    if (getenv("COLI_NO_OMP_TUNE")) return; /* same kill-switch as the others */
    if (getenv("OMP_NUM_THREADS")) return;  /* the user commands */
    if (reserve_cores < 0) return;

    int phys = coli_physical_cores();
    if (phys <= 0) return;                  /* unknown -> OpenMP default */
    int logical = omp_get_max_threads();
    /* logical <= phys: no SMT to exploit (or a restricted affinity that has
     * already decided for us). In both cases touch nothing. */
    if (logical <= phys) return;

    int per_core = logical / phys;          /* SMT siblings per physical core */
    int team = logical - reserve_cores * per_core;
    if (team < phys) team = phys;           /* never below the current default */
    if (team >= logical) return;            /* null reserve: nothing to say */

    omp_set_num_threads(team);
    fprintf(stderr, "[OMP] %s: %d threads (SMT included) on %d logical CPUs, "
                    "%d physical cores reserved; the FP8 expert matmuls are "
                    "bandwidth-bound and SMT pays (Phase 0.4: -25.5%%); "
                    "set OMP_NUM_THREADS=<n> to override\n",
            engine, team, logical, reserve_cores);
#else
    (void)engine; (void)reserve_cores;
#endif
}

#endif /* COLI_OMP_TUNE_H */
