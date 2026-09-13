/* qwen38_tier.h -- optional CUDA VRAM expert tier for the qwen38 engine.
 *
 * Same concept as qwen36_tier.h ("route -> place -> overlap -> learn"): the
 * hot experts are promoted into DEVICE_LOCAL VRAM across one or more GPUs and
 * computed there through the existing expert-group API of backend_cuda.cu.
 * One home device per expert (eid % n_gpus), LFRU heat with hysteresis from
 * tier.h, uploads on a background thread through staging copies.
 *
 * IL VINCOLO E' INVERTITO RISPETTO A qwen36_tier.h.
 *
 * Quel tier pretende cap_experts_per_layer == n_experts e tiene puntatori
 * grezzi dentro gli slot RAM, che percio' non devono mai essere sfrattati: con
 * la residenza totale garantita puo' andarsi a prendere i pesi da solo, e
 * infatti warmstart, lookahead e swap LFRU girano senza mai richiamare il
 * motore. Qui la residenza totale non esiste e non puo' esistere: gli esperti
 * di qwen38 stanno su disco (185 GB su una macchina da 64 GB) e gli slot SONO
 * una cache LRU con sfratto -- peggio, con COLI_MAP_EXPERTS=1 (il default piu'
 * veloce, Fase 0.2: 1,18 vs 1,64 s/token) lo slot non possiede nemmeno i byte,
 * li PUNTA dentro una mappatura del file che il kernel puo' smontare sotto i
 * piedi. Un puntatore conservato qui sarebbe penzolante entro pochi token.
 *
 * Da cui le due differenze d'interfaccia:
 *
 *  1. Il tier POSSIEDE la propria copia in VRAM e non ricorda mai un indirizzo
 *     di RAM. Fra la copia di staging e l'upload non c'e' nessuna dipendenza
 *     dallo slot che l'ha originata: quello puo' essere sfrattato subito dopo.
 *
 *  2. Il tier non sa leggere il disco, quindi non puo' promuovere da solo.
 *     Chiede (q38t_plan_fill) e attende un'OFFERTA: il motore, che il pread lo
 *     sa fare, carica l'esperto e lo passa con q38t_offer(). Lo stesso vale per
 *     le promozioni a caldo -- si offre cio' che si e' appena caricato per un
 *     miss, che e' esattamente il momento in cui i byte sono in mano.
 *
 * In cambio il formato e' un regalo: qwen38 e' FP8 nativo e con native_fp8
 * attivo lo slot contiene gia' e4m3 grezzo con scale per blocco 128x128
 * (q38_load_native_fp8_ranges), cioe' **esattamente** il fmt=8 che
 * backend_cuda.cu carica (:1412) e che i kernel grouped_hidden_f8w_dual /
 * grouped_down_f8w (:933,:968) macinano. Lo staging e' una memcpy, non una
 * conversione: niente XOR nibble come in qwen36_tier.c:63, niente espansione.
 * Il ramo di espansione BF16/F32 di q38_load_fp8_expert_weight riguarda solo
 * native_fp8 spento, e li' il tier resta fermo (vedi q38t_init).
 *
 * Il guadagno principale non e' il matmul: e' che un hit VRAM NON tocca il
 * disco. Il motore nota il routing per tutti i K, lancia i residenti e carica
 * in RAM soltanto i mancanti. Con ~4,69 MiB per esperto (3*2560*640 byte piu'
 * 300 scale) e ~29 GB di budget su due schede da 16 GB, circa 6.300 dei 24.576
 * esperti stanno in VRAM: quella frazione di miss sparisce dal percorso
 * expert-read, che la Fase 0.3 ha mostrato essere il costo dominante.
 *
 * Ordine d'uso nel decode (S=1):
 *
 *      q38t_note(layer, idx, K);                     // calore, tutti i K
 *      uint32_t m = q38t_issue(layer, idx, K, xs);   // lancia i residenti
 *      for (k non in m) { slot = q38_expert_get(...); ...CPU...;
 *                         q38t_offer(layer, idx[k], slot...); }
 *      q38t_take(m, route_gates, K, ys);             // accumula i residenti
 *
 * Il prefill di qwen38 e' gia' in forma expert-group (righe raggruppate per
 * esperto, tre matmul per gruppo) e potrebbe usare coli_cuda_expert_group()
 * sincrono, che non ha il tetto di righe dell'issue asincrono; non e' ancora
 * esposto qui, si fa dopo aver misurato il decode.
 *
 * Abilitazione: COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 * [HEAT_FILE=<path>] [Q38T_NO_WARMSTART=1]. Compilato solo quando il build
 * definisce -DCOLI_CUDA (CUDA=1); altrimenti gli stub inline qui sotto
 * tengono il motore su CPU a costo zero, come fa qwen36_tier.h. */
#ifndef QWEN38_TIER_H
#define QWEN38_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init dopo il caricamento del modello, prima del primo token. Torna 1 se il
 * tier e' attivo.  NON chiede la capienza della cache RAM: gli slot possono
 * essere sfrattati quanto vogliono, ed e' il motivo per cui questo tier esiste
 * in una forma diversa dall'altro.  Rifiuta (tornando 0, il motore resta su
 * CPU) se native_fp8 e' spento: senza FP8 nativo lo slot contiene F32 espanso,
 * quattro volte piu' grande e in un formato che i kernel fmt=8 non leggono.
 * scale_count e' fp8_nblk(hidden)*fp8_nblk(inter), cioe' quanti float valgono
 * le scale di UNA delle tre matrici -- il motore lo ha gia' calcolato per la
 * sua scale bank per layer. */
int  q38t_init(int n_layers, int n_experts, int hidden, int inter, int topk,
               int scale_count, int native_fp8);
int  q38t_ready(void);
int  q38t_is_resident(int layer, int eid);
void q38t_shutdown(void);

/* Calore dei K esperti instradati per questo token. Va chiamata PRIMA di
 * q38t_issue e prima di decidere quali caricare: un esperto residente in VRAM
 * non passa mai da q38_expert_get, quindi questo e' l'unico punto in cui il
 * tier lo vede passare. Non accoda nulla e non tocca il disco. */
void q38t_note(int layer, const int *eids, int K);

/* Offerta: il motore ha i byte di questo esperto in mano adesso (ha appena
 * servito un miss, o sta eseguendo un piano di warmstart). Il tier decide se
 * merita VRAM, e in caso copia SUBITO in staging -- al ritorno il chiamante
 * puo' sfrattare lo slot, mappato o no.
 *
 *   gate,up,down: e4m3 grezzo, [inter,hidden], [inter,hidden], [hidden,inter];
 *                 nella cache nativa sono i tre puntatori di Slot.gate/up/down
 *                 (contigui nello slab quando la copia e' attiva, sparsi in
 *                 tre mappature quando COLI_MAP_EXPERTS=1: non si assume che
 *                 lo siano).
 *   scales:       3*scale_count float [gate|up|down], cioe' la fetta di questo
 *                 esperto dentro la scale bank del layer.
 *
 * planned = 1 quando l'esperto viene da q38t_plan_fill (budget gia' riservato
 * in quel momento); 0 per un'offerta spontanea a caldo, che deve ancora
 * guadagnarsi il posto. */
void q38t_offer(int layer, int eid,
                const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                const float *scales, int planned);

/* Lancia i gruppi GPU per il sottoinsieme residente dei K esperti selezionati
 * (asincrono, tutti i device in parallelo). Torna la maschera dei k presi in
 * carico dalla GPU: quelli si calcolano da soli mentre la CPU fa i mancanti.
 * Poi q38t_take().
 *
 * Nota sul tetto di righe: coli_cuda_expert_group_issue rifiuta piu' di 8
 * righe totali per device (backend_cuda.cu:2091, "decode-scale only"). qwen38
 * ha topk=10, e la spartizione per home device (eid % n_gpus) di norma lascia
 * ~5 righe a scheda; ma niente vieta che dieci esperti cadano tutti sulla
 * stessa. Spezzare il lancio non e' un'opzione -- il backend ammette un solo
 * issue in volo per device, e un secondo richiederebbe la take del primo, cioe'
 * proprio la sincronizzazione che l'asincronia doveva evitare. I sopranumerari
 * restano quindi alla CPU via maschera, come i non residenti: mai un risultato
 * sbagliato, al piu' un token piu' lento. Il contatore row-overflow di
 * q38t_stats dice se il caso si presenta davvero. */
uint32_t q38t_issue(int layer, const int *eids, int K, const float *x);

/* Raccoglie i risultati GPU e accumula val[k]*y_k dentro out[hidden]. Va
 * chiamata dopo ogni issue che ha tornato una maschera non nulla, anche se nel
 * frattempo la CPU ha fatto tutto il resto. */
void q38t_take(uint32_t mask, const float *val, int K, float *out);

/* Warmstart: il tier pianifica l'insieme da riempire (ordine di calore, budget
 * riservato subito) e lo consegna; il motore carica ogni esperto e lo restituisce
 * con q38t_offer(..., planned=1), da quanti thread vuole. q38t_fill_wait()
 * blocca finche' la coda di upload non e' vuota. */
int  q38t_plan_fill(int *layers, int *eids, int max);
/* Restituisce il budget riservato per un esperto pianificato che il motore ha
 * poi deciso di non caricare (fuori dal suo intervallo di layer, formato
 * inatteso). Senza, quella VRAM resterebbe prenotata per sempre da nessuno. */
void q38t_cancel_plan(int layer, int eid);
void q38t_fill_wait(void);

/* Un blocco di telemetria su stderr: residenza, hit/miss, upload per device. */
void q38t_stats(void);

#else /* !COLI_CUDA: stub inline, il motore resta CPU-only */

static inline int  q38t_init(int a,int b,int c,int d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  q38t_ready(void){return 0;}
static inline int  q38t_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void q38t_shutdown(void){}
static inline void q38t_note(int a,const int*b,int c){(void)a;(void)b;(void)c;}
static inline void q38t_offer(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;}
static inline uint32_t q38t_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void q38t_take(uint32_t a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline int  q38t_plan_fill(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void q38t_cancel_plan(int a,int b){(void)a;(void)b;}
static inline void q38t_fill_wait(void){}
static inline void q38t_stats(void){}

#endif /* COLI_CUDA */
#endif /* QWEN38_TIER_H */
