# Handoff — debito ⑥ chiuso: il tier VRAM del trunk e il container int4 si escludono a vicenda

Destinatario: l'LLM che sta lavorando a questa campagna insieme a me.
Scritto il 2026-09-19 a fine finestra GPU. Tutto ciò che segue è misurato sul rig,
non dedotto: dove c'è un numero c'è un run che lo ha stampato.

## 1. Il risultato in una riga

Il tier VRAM degli esperti di upstream (`c/qwen36_tier.c`, prefisso `[qtier]`) non
è mai stato acceso in nessuno dei quattro bracci della ribatteria, perché
`Q38_INT4_SNAP` — che tutti e quattro impostano — instrada ogni esperto su un
percorso di caricamento che non produce mai uno slot FP8 nativo, e la guardia
d'ingresso del tier accetta **solo** slot FP8 nativi. Non è una politica di
promozione troppo timida: è una porta chiusa a monte.

## 2. La catena, con i numeri di riga

1. `c/qwen38_core.h:1763` — `q38_load_expert()` apre con:
   ```c
   if(q38_try_load_int4_expert(m,layer,eid,s))return;
   ```
   Con `Q38_INT4_SNAP` impostato questo ramo vince **sempre**: l'esperto entra da
   `q38_load_int4_ranges()` → `q38_bind_int4_slot()`, che popola `i4_slab`.
   `fp8_slab` resta `NULL` e il percorso FP8 nativo non viene mai visitato.

2. `c/qwen38_core.h:2380` — la guardia:
   ```c
   static void q38_tier_note(int layer,int eid,const Slot *ex) {
       if(!qt_ready()||!ex->fp8_slab||ex->gate.data!=ex->fp8_slab)return;
   ```
   `!ex->fp8_slab` è vero per ogni esperto ⇒ `return` per ogni esperto.

3. `c/qwen36_tier.c:886` — `qt_note()` non viene quindi chiamata nemmeno una
   volta. È l'unico ingresso alla promozione: in stream mode va a
   `stream_promote_locked()` (`:868`), che con budget libero chiama
   direttamente `enqueue_locked()`. Il budget era liberissimo (`used` 1,94/2,04 GB
   contro `budget` 12,30 GB per device), quindi **una sola** chiamata sarebbe
   bastata a produrre un upload.

4. `c/qwen36_tier.c:840` — `q_full_skips` resta 0. Questo è il dettaglio che
   chiude ogni spiegazione alternativa: la coda non è mai stata piena, era
   **vuota**. Non è congestione, non è budget, non è LFRU.

## 3. Le misure

Tutti i run: braccio `trunk` (`Q38_TIER_TRUNK=1`), `COLI_CUDA=1`, `COLI_TIMERS=1`,
`N_NEW=8`, prompt 2k, `Q38T_DEV_RESERVE_MB=1024`, heat file della campagna, disco
fermo e prod spenta.

| run | `Q38_INT4_SNAP` | `COLI_MAP_EXPERTS` | cap | resident | uploads | miss(CPU) | hit rate | parallel-batches |
|---|---|---|---|---|---|---|---|---|
| replica batteria | sì | 1 | 8192 | 0/24576 | 0 | 3360 | 0,0 % | 2001 |
| controprova mmap | sì | **0** | 32 | 0/24576 | 0 | **3360** | 0,0 % | 18619 |
| **prova positiva** | **no** | 1 | 8192 | **115/24576** | **115** | 3195 | **4,9 %** | 2034 |

`parallel-batches > 0` in tutti e tre: nessun run è caduto nel cliff #1594
(`cap < topk`), quindi sono tutti confronti leciti.

Il `4,9 %` è basso in assoluto solo perché il run è di 8 token: 115 esperti
promossi su un budget che ne ammette ~2091 per device. Il punto non è il valore,
è che **non è zero**.

## 4. Un'ipotesi che ho pubblicato e poi ritirato — leggila, ti risparmia il giro

Prima di questa misura avevo identificato una causa diversa e l'avevo trovata
convincente: nel ramo mapping di `q38_load_native_fp8_ranges`
(`c/qwen38_core.h:1281-1295`) c'è un `free(slot->fp8_slab); slot->fp8_slab=NULL`
quando i tre range sono mappati, e questo affama letteralmente la guardia del
punto 2. Avevo quindi previsto che `COLI_MAP_EXPERTS=0` riaccendesse il tier, e
ne avevo tratto una conclusione strutturale (il nostro host-tier *richiede*
`COLI_MAP_EXPERTS=1`, `c/qwen38_core.h:1566`, quindi le due torri sarebbero state
mutuamente esclusive per costruzione).

**La controprova l'ha falsificata.** Con `COLI_MAP_EXPERTS=0` non cambia nulla, e
`miss(CPU) 3360` è identico *bit per bit* nei due bracci. La catena era reale come
codice ma non è mai stata quella percorsa: con int4 quella funzione non viene
proprio chiamata. Se hai in coda ragionamenti costruiti sopra quella
mutua-esclusione, scartali.

La lezione di metodo, che vale per entrambi: una catena di codice coerente non è
una misura, e il braccio di controllo va acceso **sul flag che si crede causale**,
non sul primo flag che la catena nomina.

## 5. Cosa questo fa al «trunk −12,39 %» di notte-18

Va riletichettato, non ritirato. Quel numero ha misurato una configurazione reale
e resta valido per ciò che ha misurato:

- il trunk **denso** int8 di upstream **era attivo** in ogni braccio
  (`553 of 553 offered matrices resident as int8 (3,96 GiB)` compare in tutti i
  log, compresi quelli a hit rate 0);
- l'**expert tier** di upstream **era spento** in ogni braccio.

Quindi: sugli esperti il confronto fra le due torri **non è mai avvenuto**. Se
quel −12,39 % viene usato in una discussione upstream o in una conclusione sulla
qualità del loro tier, è scorretto. Va detto esplicitamente, con questa tabella.

## 6. Fatto collaterale che ti riguarda se rifai misure di memoria

Il primo tentativo di controprova, a `cap=8192` con `COLI_MAP_EXPERTS=0`, è stato
ucciso dall'OOM killer (`anon-rss 31,7 GB`, ore 20:06:50). Senza mapping la LRU
per layer diventa slab **anonimi** non evictable, invece che mapping su page
cache: le due configurazioni **non sono confrontabili a parità di `cap`**. La
controprova è stata rifatta a `cap=32` (RSS 11,8 GB dopo il load). Se progetti un
A/B su `COLI_MAP_EXPERTS`, il `cap` va riscalato o la misura muore.

## 7. Decisione presa — ⑥ è chiuso (19/09, titolare)

Il debito si chiude qui: **il ponte int4 → `qt_note` non si costruisce.** Le
domande che restavano aperte sono state decise, e il motivo non è di gusto, è
nei contatori.

Il ponte avrebbe avuto solo due forme, entrambe perdenti:

1. **Soddisfare la guardia caricando anche l'FP8 nativo** — raddoppia i byte per
   esperto (2,64 → 6,02 MB) e cancella esattamente il guadagno per cui il
   container int4 esiste: il +38 % end-to-end della A/B a container completo era
   guadagno byte-contabile.
2. **Insegnare a upstream il `fmt=4` gs64** — lavoro dentro il loro albero su un
   formato che è nostro, e che dovrebbero mantenere loro.

E soprattutto **il premio non c'è**, il che rende la scelta indipendente dal
costo del ponte. Le due misure che lo dicono:

- il controllo positivo (stesso braccio, tolto il solo `Q38_INT4_SNAP`) sveglia
  davvero il tier di upstream — `resident 115/24576`, `uploads 115`, hit rate
  **4,9 %** — e quella run ha fatto **TTFT 329,06 s**;
- il braccio `trunk` puro, con **prefetch totale** (`prefetched=41532` su 41532
  weight-range) e cache hit rate **96,3 %**, cioè con le carte migliori, resta
  **−12,39 %**.

Portare gli esperti int4 dentro quella torre significherebbe quindi spendere
banda o lavoro upstream per entrare in un percorso che nel nostro regime è più
lento del nostro.

**Cosa NON viene abbandonato**, perché la parola «tier» confonde due cose: il
container `fmt=4` gs64, il loader `Q38_INT4_SNAP`, il kernel AVX2 (a S=1 batte
l'fp8 di 1,49–1,61×) e **la nostra arena VRAM** restano in pieno servizio. Con
`Q38_INT4_SNAP` attivo il *nostro* tier fa hit rate **92,1 %** e tiene il
**70,1 %** del lavoro esperti sulle schede: gli esperti int4 sono già residenti e
già serviti dalle GPU. Del trunk teniamo il pezzo che vince davvero, il **denso
int8**, attivo in ogni braccio e miglior denso mai misurato qui (42,4+33,2 contro
100,6+69,9 ms/fwd).

L'unica frase che si chiude è: *gli esperti int4 non passeranno mai per
`qt_note` di upstream*.

**Resta in piedi, e vale per chiunque legga:** il −12,39 % è **il numero da non
riusare senza etichetta**. Misurava il trunk *denso* con l'expert tier di
upstream irraggiungibile; sugli esperti il confronto fra le due torri non è mai
avvenuto.
