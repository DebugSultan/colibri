/* omp_tune.h — dimensionamento della squadra OpenMP sui CORE FISICI.
 *
 * PERCHE' SOLO IL DIMENSIONAMENTO, E NON LO SPIN-WAIT.
 * Il blocco di tuning in colibri.c fa due cose che hanno profili di rischio
 * OPPOSTI, e vanno tenute separate:
 *
 *   dimensionamento  OMP_NUM_THREADS = core fisici (niente SMT)
 *                    #718: +2.3x su Zen3 (5950X, 16C/32T) solo cambiando
 *                    il numero di thread. Il guadagno e' cosi' grande che
 *                    "sommerge la maggior parte dei delta che si citano qui".
 *
 *   spin-wait        OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME
 *                    #707: -2.2x sul decode di un host a bassa residenza
 *                          (M1 Max 32 GB, ~10% di expert residenti)
 *                    #116: -39% su Metal      #159: ~3x su x86+CUDA
 *                    #341: 3000% di CPU su FreeBSD con la squadra ferma
 *                    Meccanismo: dove il token e' fatto di byte dal disco,
 *                    una squadra che gira a vuoto ruba i core al pool di I/O
 *                    che sta facendo il lavoro vero.
 *
 * Kimi K3 e OLMoE non avevano NESSUNA delle due. Qui prendono solo la prima:
 * Kimi e' il motore piu' disk-bound del progetto (misurato: 6.7% di hit,
 * 891 GB letti per 32 token), cioe' esattamente il regime in cui la seconda
 * meta' fa danno. Aggiungergliela sarebbe stato un peggioramento misurabile.
 * GLM now calls this same helper after its optional hot-team re-exec, so it
 * receives the physical-core sizing without changing the independent spin-wait
 * policy that its existing block controls.
 *
 * PERCHE' QUI NON SERVE IL RE-EXEC.
 * colibri.c si ri-esegue perche' OMP_WAIT_POLICY & co. le legge il COSTRUTTORE
 * di libgomp, prima di main(): un setenv() dentro main() arriva tardi. Il numero
 * di thread no: omp_set_num_threads() e' una API di runtime e ha effetto subito.
 * La meta' sicura e' anche la meta' semplice.
 *
 * REGOLA SUI FALLIMENTI: se il conteggio dei core fisici non e' determinabile,
 * NON si indovina — si lascia il default di OpenMP. Un conteggio sbagliato e'
 * peggio di nessun conteggio (cfr. #325, dove un fallback silenzioso a 1
 * inchiodava il decode su un core solo).
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

/* Numero di core FISICI, o 0 se non determinabile. Mai un valore inventato. */
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
    /* hw.perflevel0.logicalcpu = i core PERFORMANCE. Su Apple Silicon
     * hw.physicalcpu li conta tutti, E-core compresi (10 su un M1 Max), e con
     * una barriera per matmul e' il thread piu' lento a dettare il passo: gli
     * E-core rallentano la squadra invece di aiutarla (#707, -4.2% decode).
     * Su Intel Mac perflevel* non esiste: li' hw.physicalcpu e' corretto. */
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;

#else
    /* Linux: un core fisico = una lista di thread_siblings distinta. Contare le
     * liste uniche deduplica l'SMT senza dover interpretare la topologia. */
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

/* Dimensiona la squadra OpenMP sui core fisici. Rispetta OMP_NUM_THREADS se
 * l'utente l'ha impostata, e non fa nulla se il conteggio non e' affidabile.
 * `engine` finisce solo nella riga di log. */
static void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* stesso kill-switch degli altri motori */
    if (getenv("OMP_NUM_THREADS")) return; /* l'utente comanda */

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* sconosciuto -> default di OpenMP */
    int logical = omp_get_max_threads();
    if (phys >= logical) return;           /* niente SMT da evitare: silenzio */

    omp_set_num_threads(phys);
    fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                    "SMT can halve decode throughput on some CPUs (#718); "
                    "set OMP_NUM_THREADS=<n> to override\n",
            engine, phys, logical);
#else
    (void)engine;
#endif
}

/* Dimensiona la squadra TENENDO l'SMT, meno una riserva di core fisici interi.
 * Politica OPPOSTA a coli_omp_tune_threads() qui sopra, e opposta per una
 * ragione misurata, non per gusto: le due servono due regimi diversi.
 *
 *   core fisici   il regime di #718: GEMV int4 su pesi gia' residenti. Le
 *                 regioni sono minuscole e back-to-back, due fratelli SMT si
 *                 contendono la stessa unita' vettoriale e la squadra collassa
 *                 (2.3x su Zen3).
 *
 *   SMT inclusa   il regime di Qwen3.8-Flash-Next: matmul FP8 degli esperti su
 *                 pagine APPENA FAULTATE dal disco. Il collo e' la banda verso
 *                 la memoria, non l'unita' vettoriale, e li' il fratello SMT
 *                 non contende: copre la latenza dell'altro.
 *
 * La misura che autorizza questo (campagna qwen38, Fase 0.4 del 13/09/2026;
 * verbale in qwen38cuda/PIANO_NEXT.md §9). Ryzen 9 3900X 12C/24T, 64 GB, cap
 * 128, prompt 274 token + 16 nuovi = 290 forward, ogni gradino ripetuto:
 *
 *   OMP_NUM_THREADS   16 token (copia)   resident-mm   deltanet
 *     12 (fisici)        128,5 s          3648 ms/fwd   2440 ms/fwd
 *     16                 109,9 s          2792          1974
 *     20                  95,7 s  -25,5%  2296  -37,1%  1661  -31,9%
 *
 * L'efficienza di scaling resta ~97% fino a 20: nessuna saturazione, che e'
 * per l'appunto la firma del regime bandwidth-bound. `expert-read` (disco)
 * resta piatto, 1108 -> 1087 ms/fwd: il guadagno e' tutto nel denso CPU.
 *
 * PERCHE' UNA RISERVA, E PERCHE' IN CORE INTERI.
 * Il gradino a 24 non e' stato misurato: escluso dal titolare perche' su questo
 * nodo gira un servizio di produzione accanto, e saturare tutte le CPU logiche
 * lo affamerebbe. La riserva si conta in core fisici INTERI (entrambi i loro
 * fratelli SMT) perche' lasciare mezzo core non lascia niente: il fratello
 * rimasto continua a contendere la stessa unita'. Con reserve_cores=2 su un
 * 3900X: 24 - 2*2 = 20, che e' il numero misurato sopra.
 *
 * PAVIMENTO. Non si scende mai sotto i core fisici, cioe' sotto il default
 * odierno: su un host senza SMT, o con pochi core, questa funzione non deve
 * poter peggiorare cio' che c'e' gia'. Vale anche la REGOLA SUI FALLIMENTI in
 * testa al file: conteggio non determinabile -> si lascia il default di OpenMP.
 */
static void coli_omp_tune_threads_smt(const char *engine, int reserve_cores)
{
#ifdef _OPENMP
    if (getenv("COLI_NO_OMP_TUNE")) return; /* stesso kill-switch degli altri */
    if (getenv("OMP_NUM_THREADS")) return;  /* l'utente comanda */
    if (reserve_cores < 0) return;

    int phys = coli_physical_cores();
    if (phys <= 0) return;                  /* sconosciuto -> default di OpenMP */
    int logical = omp_get_max_threads();
    /* logical <= phys: niente SMT da sfruttare (oppure una affinity ristretta
     * che ha gia' deciso per noi). In entrambi i casi non si tocca nulla. */
    if (logical <= phys) return;

    int per_core = logical / phys;          /* fratelli SMT per core fisico */
    int team = logical - reserve_cores * per_core;
    if (team < phys) team = phys;           /* mai sotto il default odierno */
    if (team >= logical) return;            /* riserva nulla: niente da dire */

    omp_set_num_threads(team);
    fprintf(stderr, "[OMP] %s: %d threads (SMT inclusa) su %d CPU logiche, "
                    "%d core fisici riservati; il matmul FP8 degli esperti e' "
                    "bandwidth-bound e l'SMT paga (Fase 0.4: -25,5%%); "
                    "set OMP_NUM_THREADS=<n> to override\n",
            engine, team, logical, reserve_cores);
#else
    (void)engine; (void)reserve_cores;
#endif
}

#endif /* COLI_OMP_TUNE_H */
