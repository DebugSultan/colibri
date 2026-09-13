# Campagna Qwen3.8-Flash-Next — registro vivo

> Gemello di `PIANO_38.md` per la linea **colibri / qwen38**, non per la linea
> llama.cpp. Le due non si toccano: qui il modello **non è residente**, gli
> esperti si leggono dal disco a ogni token. I numeri di una linea non valgono
> nell'altra — ci sono già cascato una volta, portando il Q3_K_XL da 88 GB
> dentro un ragionamento dove non significa niente.

Avviato: 07/09/2026. Macchina: **node-01**.

---

## 1. Il bersaglio

`Qwen/Qwen3.8-Flash-Next-FP8`, revisione pinnata
`bcd9f01ddc9cff2316eb84281bebcd5b058bddce`.
**185,6 GB decimali (172,8 GiB), 144 file, 131 shard safetensors.**

Si carica **dagli shard ufficiali, senza conversione**: il formato su disco è
già quello che i kernel di colibri consumano.

### 1.1 Cosa dice la sua `config.json` (misurato, non assunto)

Letta il 07/09/2026 dal checkpoint in corso di download.

| | |
|---|---|
| architettura | `Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp` |
| layer | **48** — 36 `linear_attention` + 12 `full_attention`, intervallo 4 |
| hidden | 2560 · head_dim 256 · 24 teste attn · 2 teste KV · rotary parziale 0,25 |
| attenzione lineare | 16 teste key ×128, 48 teste value ×128, conv kernel 4 |
| MoE | **512 esperti per layer**, **top-k 10**, `moe_intermediate_size` **640** |
| esperto condiviso | uno per layer, intermediate 640 — sempre residente |
| PLE / N-gram | `ngram_size` 3, base vocab **20.000.000**, `split_ngram_parts` 128, iniettato al layer **2**, embed dim 2560 |
| vocabolario | 248.320 · `max_position_embeddings` **262.144** |
| MTP | 1 layer, ibrido |
| **visione** | `vision_config` depth 27, hidden 1152, patch 16, out 2560 — **il modello è multimodale** |
| quantizzazione | `fp8` E4M3, attivazioni **dinamiche**, `weight_block_size` **[128, 128]** |
| `modules_to_not_convert` | **943 voci** |

Tre conseguenze che contano più di quanto sembri:

1. **Solo gli esperti routed sono FP8.** Le 943 voci escluse coprono
   `lm_head`, `embed_tokens`, tutte le proiezioni di attenzione, i mixer
   hyper-connection, gli esperti condivisi e i gate: **tutto BF16**. È
   esattamente la parte densa da 9,2 GiB della ricognizione, e significa che
   qualunque riquantizzazione futura tocca **una sola famiglia di tensori**,
   lasciando intatto il resto. Separazione pulita, senza ambiguità.
2. **`weight_block_size [128, 128]` combacia con `fmt=8` di colibri** (blocco
   FP8 E4M3 + scale f32 per blocco 128×128). Il «carica senza conversione»
   non è una speranza, è verificato sul campo `quantization_config`.
3. **È multimodale, e noi useremo solo la torre testuale.** Il target `qwen38`
   di colibri è testuale; l'encoder visivo va saltato in caricamento. Va messo
   a verbale prima di stupirsi di tensori non riconosciuti.

### 1.2 Il conto degli esperti — letto dagli header, non stimato

`census.py` apre l'header di tutti i 131 shard e somma i byte reali.

- layer che portano esperti: **49, non 48** — i 48 del language model **più il
  layer MTP**, che ha i suoi 512 esperti (3.072 tensori, 2,52 GB).
- esperti totali: 49 × 512 = **25.088**
- un esperto reale (layer 0, id 0):

| tensore | dtype | forma | byte |
|---|---|---|---|
| `gate_proj.weight` | F8_E4M3 | [640, 2560] | 1.638.400 |
| `up_proj.weight` | F8_E4M3 | [640, 2560] | 1.638.400 |
| `down_proj.weight` | F8_E4M3 | [2560, 640] | 1.638.400 |
| i tre `weight_scale_inv` | **BF16** | [5, 20] / [20, 5] | 200 l'uno |
| **totale** | | | **4.915.800 B = 4,6881 MiB** |

- su disco: 120,81 GB (48 layer) + 2,52 GB (MTP) = **123,33 GB = 114,85 GiB**
- letti per token: 10 × 48 = 480 esperti = **2,198 GiB** (il layer MTP è
  speculativo: non entra nel percorso base)

Due correzioni al recon: gli esperti sono **25.088, non 24.576** (+2,1 % di
disco), e le scale FP8 sono **BF16, non F32** — dettaglio che il convertitore
int4 e il loader devono rispettare, perché nel checkpoint non c'è **un solo
tensore F32**.

### 1.3 Il censimento completo del checkpoint (misurato)

| famiglia | GB dec | GiB | quota | tensori |
|---|---|---|---|---|
| esperti routed, 48 layer | 120,81 | 112,51 | 65,1 % | 147.456 |
| **PLE / tabella N-gram** | **51,27** | **47,75** | **27,6 %** | 138 |
| denso restante (BF16) | 6,81 | 6,35 | 3,7 % | 867 |
| embedding + lm_head (BF16) | 2,54 | 2,37 | 1,4 % | 2 |
| esperti routed, layer MTP | 2,52 | 2,34 | 1,4 % | 3.072 |
| vision encoder (BF16) | 0,90 | 0,84 | 0,5 % | 333 |
| shared expert (BF16) | 0,48 | 0,45 | 0,3 % | 196 |
| MTP denso (BF16) | 0,17 | 0,16 | 0,1 % | 25 |
| **totale** | **185,50** | **172,76** | | **152.089** |

**Il terzo blocco che nessuno dei due documenti aveva contato.** La tabella
N-gram del PLE pesa **51,27 GB — il 27,6 % del checkpoint**, più di un quarto.
Sono **128 shard FP8 di forma [2.500.012 × 160]** (400 MB l'uno) con **una sola
scala BF16** per l'intero tensore, più tre tensorini `I64` di indicizzazione
(`layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes`). Vive
sul layer 1.

Non è un peso di matmul: è una **tabella di lookup**. Per token si toccano
pochissime righe da 160 B (`ngram_size: 3`), quindi il traffico è
trascurabile — ma sono 51,27 GB che **non possono stare in RAM** e vanno
mappati su disco accanto agli esperti. È anche il motivo concreto per cui gli
N-gram «restano separati dal modello»: sono letteralmente un archivio a parte,
e **l'int4 sugli esperti non li tocca**.

Il denso davvero residente è quindi **6,81 + 2,54 + 0,48 + 0,17 = 10,00 GB =
9,31 GiB**: la stima del recon (9,2 GiB) regge. Il **vision encoder (0,90 GB)
va saltato** — il target `qwen38` di colibri è solo testo, e il modello è
multimodale.

---

## 2. La macchina

node-01: Ryzen 9 3900X (12c/24t), **64 GB DDR4 a 3200** (da NON rialzare —
sospetta causa del panic), NVMe **Samsung 990 PRO 1 TB** singolo, zram 30,2 G.
GPU: RTX 5070 Ti 16 GB + RTX 5060 Ti 16 GB, Blackwell **sm_120** (CUDA ≥ 12.8;
in casa è pinnata la 13.0).

Vincolo permanente: sulla stessa VRAM gira il **cervello 27B K12 di
produzione**, con picco 30.962 MiB su 32 GiB. La Fase 0 non lo tocca. La
Fase 1 richiede una finestra concordata.

---

## 3. Dove va la memoria

| voce | costo | dove |
|---|---|---|
| denso residente BF16 | **9,2 GiB** fisso | RAM |
| cache esperti routed | **4,6881 MiB × slot × 48 layer** | RAM |
| banca delle scale FP8 | **14,4 MiB** fisso | RAM |
| workspace | ≤ **1,1 GiB** di picco | RAM |
| esperti routed | **123,3 GB** | **disco, in streaming** |
| tabella N-gram PLE | **51,3 GB** | **disco, lookup sparso** |

La banca delle scale è il dettaglio elegante: tutte le scale di blocco di
tutti gli esperti restano residenti, quindi **un miss costa una sola lettura
FP8** — nessun secondo accesso per andare a prendere la scala.

I 28 MiB della ricognizione erano calcolati su scale **F32**: 24.576 × 300 × 4 B
= esattamente 28,125 MiB. Gli header dicono **BF16**, e gli esperti sono 25.088:
25.088 × 300 × 2 B = **14,36 MiB**. La banca costa **metà** di quanto scritto.

**La RAM qui è una manopola, non un soffitto.** È il punto che ribalta tutto
rispetto alla linea llama.cpp: lì i 99 GB di RAM+VRAM erano un tetto e la
quantizzazione era l'unica leva; qui `cap` decide solo *quanto* cache si
compra. Con 64 GB e ~45 GiB utili dopo denso, workspace e sistema, il budget
regge **cap ≈ 205** — il **39 % di residenza** (9.829 esperti su 25.088);
la curva 0.3 si ferma a 192 (≈42 GiB, 9.216 esperti, 36,7 %) per non
saturarlo. Punto di riferimento: al 6 % (cap 32) upstream misura già
**53,8 % di hit**.

---

## 4. L'economia del token

Mediana 140 s/token, scomposta:

| termine | costo | quota |
|---|---|---|
| **disco** | **96 s** | **69 %** |
| matmul esperti | 17 s | 12 % |
| attenzione | 14 s | 10 % |
| LM head | 2,6 s | 2 % |

Azzerare *tutto* il calcolo porta a 106 s: **+33 % per un backend intero.**
Il collo è il disco, e ogni progetto che non lo attacca è un progetto che
compra il 33 % al prezzo di un backend.

Un tier VRAM sposta parte dei 96 s, non li elimina: 32 GiB / 4,6881 MiB ≈
6.989 esperti su 25.088 = **28 % di residenza**.

---

## 5. NVFP4: verificato sui sorgenti, e scartato

Cercato in colibri il 07/09/2026, non a memoria.

- **NVFP4 non esiste.** Zero occorrenze. Nessun formato E2M1 con scala E4M3
  per 16 elementi e scala globale FP32.
- **MXFP4 esiste ma solo denso.** `fmt=7`, standard OCP (nibble `e2m1` +
  esponente `ue8m0` ogni 32): ha decoder CPU (`matmul_mxfp4`), matmul CUDA
  (`coli_cuda_matmul_mxfp4`, `backend_cuda.h:108-113`) e oracolo
  (`tests/test_mxfp4_cuda.cu` contro `tests/mxfp4_ref.c`). Ma **non ha nessun
  kernel `grouped_*`**, e in `expert_group_impl` (`backend_cuda.cu:1851-1866`)
  i formati ammessi sul percorso esperti sono **2, 4, 6, 8**. Il 7 non
  compare in nessuna delle flag.

Quindi introdurre NVFP4 vorrebbe dire: formato nuovo + decoder CPU + famiglia
di kernel grouped + oracoli, per comprare **precisione di calcolo su un
percorso che non è compute-bound**. Vale anche il precedente di casa: NVFP4
bocciato tre volte sul 27B, con la riserva «vivo solo sui MoE grossi». La
riserva si applica — questo *è* un MoE grosso — ma resta che il formato non
esiste nel motore e il collo è altrove.

**Deciso: non si insegue NVFP4.**

---

## 6. int4: la strada che è già costruita

| formato | cos'è | kernel esperti |
|---|---|---|
| **fmt=2** (`w4`) | int4 simmetrico −8..7, scala f32 **per riga** | `grouped_hidden_w4` / `_w4_dual` / `grouped_down_w4` |
| **fmt=4** (`g4`) | id., scale **lungo l'input** — `sc[o*ng + i/gs]`, `gs` pari | `grouped_hidden_g4_dual` / `grouped_down_g4` |
| tensor-core | `grouped_s4_wmma`, dietro `COLI_CUDA_TC_INT4` | richiede `D%32==0 && I%32==0` |

Tre cose:

1. L'epilogo `silu(gate)*up` è **già fuso** nei kernel duali — niente terzo
   kernel, niente round-trip di `up` in memoria globale.
2. I vincoli di forma del percorso tensor-core sono **soddisfatti**: 2560 e
   640 sono entrambi multipli di 32.
3. È **esattamente** il formato che `qwen36_tier.h` già gestisce
   (`expert_is_int4`, fmt=4).

### 6.1 Il conto, e la sorpresa sulle scale

Con `gs = 128` lungo l'input, per esperto:
4.915.200 pesi × 0,5 B = 2,3438 MiB, più 4.915.200/128 = 38.400 scale × 4 B =
150 KiB → **2,4902 MiB contro 4,6881. Rapporto 1,88×.**

| | FP8 (oggi) | int4 g4 |
|---|---|---|
| per esperto | 4,6881 MiB | **2,4902 MiB** |
| esperti su disco | 123,3 GB | **65,5 GB** |
| streaming per token | 2,198 GiB | **1,167 GiB** |
| residenti in 45 GiB di RAM | 9.829 = **39 %** | 18.505 = **74 %** |
| residenti in 32 GiB di VRAM | 6.989 = 28 % | 13.158 = **52 %** |

La sorpresa, ora misurata: **int4 g4 con `gs=128` ha molte più scale della
sorgente FP8**, non meno. La sorgente ha scale per blocco 128×128, e l'header
lo conferma: `weight_scale_inv` di forma **[5, 20] = 100 scale per matrice**.
Il g4 ne ha una per riga per gruppo di 128 lungo l'input → 640 × 20 =
**12.800 per matrice**: **128× più fine sull'asse di uscita** (una scala per
riga invece che una ogni 128 righe). Questo compensa in parte il passaggio da
8 a 4 bit — non è una garanzia, va misurato, ma è un argomento reale.

Da decidere in Fase 1: se le scale g4 possono essere **BF16** invece di f32,
l'esperto scende da 2,4902 a 2,4170 MiB (1,94× invece di 1,88×). Va letto nel
formato `fmt=4` di colibri, non assunto.

### 6.2 Perché è la leva più grossa

L'int4 attacca i 96 s di disco **da due lati insieme**: dimezza i byte di ogni
miss *e* quasi raddoppia gli esperti che un miss non lo fanno mai. Il tier
VRAM, da solo, tocca solo i 33,6 s di calcolo. E i due si sommano.

### 6.3 Il costo, detto onestamente

1. **Non esiste un checkpoint int4 di questo modello**: va prodotto noi dagli
   shard FP8. Il download resta prerequisito, non alternativa.
   il checkpoint FP8 è già a terra (185,5 GB), servono altri **65,5 GB** dei
   **312 GB liberi**. Ci sta, con margine.
2. Serve un **convertitore** FP8-blocco-128×128 → int4-gruppo e un **loader**
   per il nuovo artefatto: il percorso nativo
   (`q38_try_load_native_fp8_expert`, slab contiguo, «a miss is one FP8 read»)
   va replicato per int4. È il lavoro vero, e non è banale.
3. **Problema di metodo, da dire subito**: non abbiamo un riferimento BF16 per
   questo modello — il checkpoint *è* FP8 nativo, un BF16 non esiste. L'arbitro
   diventa **l'FP8 nativo stesso**. È accettabile (si misura lo scostamento da
   ciò che il modello realmente è), ma va dichiarato e mai confuso con la
   metrica della campagna 27B.
4. `w4`/`g4` sono simmetrici, senza zero-point. Sui pesi di esperti MoE di
   solito basta; la coda va **misurata**, non assunta.

---

## 7. Le fasi

### Fase 0 — misurare, CUDA spento, produzione intatta

Nessun rischio per il K12: `COLI_CUDA` resta a 0, non si tocca la VRAM.

- **0.1** ~~build CPU del target `qwen38`~~ **CHIUSO**: build pulita
  (gcc 15.2, `-O3 -march=native -fopenmp`, zero warning), `qwen38-tiny-check`
  verde su tutta la matrice — 8/8 (batch 0/1 × BF16 0/1 × cap 1/4), token
  8/8 e oracolo numerico PASS (cosine 0,99999, max_abs 8,4e-4 < 1e-2) — e
  `qwen38-ple-prefetch-check` verde (16 configurazioni, token identici
  prefetch on/off). Il generatore del banco richiede `torch` +
  `transformers==5.16.1`: venv dedicato in
  `/opt/zyonix/backend/qwen38-tiny-venv`, invocato con
  `make PYTHON=/opt/zyonix/backend/qwen38-tiny-venv/bin/python`.
- **0.2** ~~baseline con `COLI_TIMERS=1`~~ **CHIUSO** (vedi §9, 08/09): la
  scomposizione 96/17/14/2,6 si conferma nelle fasi ma **non nelle quote** —
  su questa macchina il disco pesa 36 %, non il 69 % del box di riferimento.
- **0.2b** ~~`COLI_MAP_EXPERTS=1` contro il percorso a copia~~ **CHIUSO**
  (vedi §9, 08/09): pattern identico bit per bit, 1,38× più veloce in steady
  state, −6,9 GiB di anonimo, +25,9 GiB di page cache ancorata; `RssAnon`
  misurato come da avvertenza (non `VmRSS`).
- **0.3** ~~curva hit-rate contro `cap`: 16 · 32 · 64 · 96 · 128 · 192~~
  **CHIUSA** (08/09, vedi §9). Percorso a copia col prompt lungo (290
  forwards): **7,5 / 10,2 / 26,8 / 49,0 / 61,5 / 71,4 %**; percorso mmap
  ai due gradini alti: **61,5 / 71,4 %** — identici. Il hit rate è
  **indipendente dal percorso**: dipende solo dal numero di slot e dalla
  politica di sfratto, non da come i byte arrivano in memoria.
  ⚠️ i **tempi** di M192 sono invalidi (thrashing: swap di sistema a
  22,4 GiB, `expert-read` che *sale* a 2192 ms mentre i miss si dimezzano) —
  il tempo pulito a 192 lo dà N192, in mmap.
  **Il verdetto — Fase 2 prima di Fase 1.** A cap 128 i 128,2 s si
  scompongono in `resident-mm` 58,4 s + `deltanet` 39,1 s + `expert-read`
  16,5 s: il **disco è sceso al 13 %**, il calcolo denso è al **76 %**, e
  quel pavimento è **piatto** su tutta la curva (3657 → 3654 → 3652 ms/fwd
  da cap 64 a 192). Azzerare del tutto il disco varrebbe ~13 %; oltre 128
  un GiB liberato non vale nulla (N128 135,4 s contro N192 137,4 s, con
  dieci punti di hit rate in più). Un GiB di **VRAM** vale invece l'intero
  pavimento CPU. ⚠️ **correzione**:
  `ColiExpertStoreStats` è la
  struttura dello store **deepseek_v4** — il percorso qwen38 ha la propria
  cache (`q38_expert_get`, contatori `m.hits`/`m.miss` + riga I/O a 6
  statistiche già stampata). Le «otto statistiche» della curva vanno o
  ristampate dalla cache qwen38 (micro-task di stampa) o costruite da
  hit/miss + I/O + geometria nota (4,915,800 B per record); la curva in sé
  non le richiede.
- **0.3a** ~~chiudere il «14 MB per miss»~~ **CHIUSO senza banco**: l'header dice
  4.915.800 B per esperto (§1.2). I 14 MB della ricognizione sono
  **~3 esperti**, non uno: o un miss contava l'intero gruppo top-k parziale, o
  la cifra era per-layer e non per-esperto. Il numero da usare è **4,6881 MiB**.
- **0.3b** ~~verificare come colibri gestisce la tabella N-gram da 51,3 GB~~
  **CHIUSO dal sorgente, non a banco** (SPEC §2.4): `q38_ple_row`
  (`qwen38_core.h:1344`) legge **una riga per volta dallo shard** — 16 teste
  × 160 B = 2.560 B per token — niente residenza, una sola scala BF16
  globale, e prefetch `Q38_PLE_PREFETCH` (default 1) perché gli indici di
  riga sono noti prima di ogni calcolo.

> **Nota di numerazione (13/09).** I due punti qui sopra erano numerati
> «0.4» e «0.5» e collidevano con la misura dei thread (§9, 13/09). Sono
> corollari della curva 0.3, chiusi a tavolino e non a banco: rinumerati
> **0.3a** e **0.3b**. La serie 0.4 / 0.4b / 0.5 resta alle misure vere.

- **0.4** ~~ri-misurare a 20 thread~~ **CHIUSA** (13/09, §9): l'SMT paga,
  **−25,5 %** sui 16 token; e la misura va fatta in una cgroup senza swap (**0.4b**).
- **0.5** ~~rendere i 20 thread il default~~ **CHIUSO** (13/09, §9): la
  politica vive nel runtime del motore, non nel launcher.

**La curva 0.3 era il giudice, e ha sentenziato** (08/09, §9). Un GiB di RAM
liberato — cioè l'int4 — compra hit rate, e il hit rate compra il **13 %** del
tempo che è rimasto al disco; oltre cap 128 non compra nemmeno quello. Un GiB
di **VRAM** attacca il **76 %** che sta nel calcolo denso, un pavimento che
non si è mosso di 5 ms su tutta la curva. **L'ordine è quindi invertito:
Fase 2 prima di Fase 1.** L'int4 resta valido come mezzo per far stare cap
alti in meno RAM (o per liberare RAM al resto della macchina), non come leva
di velocità.

### Fase 1 — int4 (declassata dalla curva: leva di RAM, non di velocità)

1. Convertitore esperti FP8 128×128 → int4 `g4` gs=128. Solo i tensori
   routed: le 943 voci di `modules_to_not_convert` restano BF16, intatte.
2. Loader dell'artefatto int4 con lo stesso schema a slab contiguo.
3. **Misura: KL + top-1 contro l'FP8 nativo. Mai la PPL.** Il precedente
   NVFP4 è categorico — la PPL avrebbe promosso il peggior candidato tre volte
   su tre.
4. Se la coda regge: ri-misura della curva 0.3 sul nuovo artefatto.

### Fase 2 — tier VRAM (**la leva vera**; richiede una finestra sul K12)

`qwen38_tier.c/h` sul modello di `qwen36_tier`, con **il vincolo invertito**:
`qwen36_tier.h:27-28` impone `cap_experts_per_layer == n_experts` perché il
tier tiene puntatori grezzi negli slot RAM, che non devono mai essere
sfrattati. Qui gli slot **sono** una cache con sfratto, quindi il tier deve
possedere la propria copia in VRAM invece di puntare alla RAM. I mixer
restano su CPU.

---

## 8. Regole della campagna

- **Un verde isolato non prova nulla.** Ogni misura ripetuta, il payload di
  prova contato.
- **Caricare non è reggere**: il tetto si misura col picco sotto stress, mai
  col cuscino a riposo.
- **KL + top-1, mai la PPL.**
- Il **K12 di produzione non si ferma** senza via esplicito.
- Ciò che gira su CPU resta **F32 o BF16**.
- **L'inferenza si misura senza swap**: motore in scope cgroup v2 con
  `MemorySwapMax=0`, e il picco di `memory.swap.current` letto *durante* il run.
  Un run che swappa non è una misura lenta, è una misura da buttare (0.4b).
- **I thread si contano, non si assumono**: `qwen38.c` non chiama
  `coli_omp_tune_threads()`, quindi il numero dipende dal percorso di lancio.

---

## 9. Stato

| | |
|---|---|
| 13/09 | **0.5 CHIUSO — i 20 thread sono il default, e la politica vive nel motore** (via esplicito del titolare: «20T fisso, non userò altri modelli»). Tre tocchi, nessuno dei quali cambia il default degli altri motori. (1) **`c/omp_tune.h`**: nuova `coli_omp_tune_threads_smt(engine, reserve_cores)` accanto alla sorella `coli_omp_tune_threads()`, con sopra il perché le due politiche sono *opposte* e la tabella della 0.4 che lo autorizza. Tiene l'SMT e riserva **core fisici interi** — mezzo core non lascia niente, il fratello rimasto continua a contendere l'unità vettoriale — con **pavimento ai core fisici**, così su un host senza SMT non può peggiorare ciò che c'è già: `logical − reserve·(logical/phys)`, che sul 3900X fa **24 − 2·2 = 20**. (2) **`c/qwen38.c`**: `coli_omp_tune_threads_smt("qwen38", 2)` subito dopo il banner, prima di qualsiasi regione parallela — il motore non chiamava **nessun** tuner (era il buco scoperto nella 0.4: a lancio diretto prendeva i 24 logici di libgomp). (3) **`c/coli`**: qwen38 diventa la **seconda eccezione** al default a core fisici, con la stessa forma di `deepseek_v4` e per la stessa ragione — il runtime si dimensiona da sé, e un `setdefault` del launcher sarebbe indistinguibile da un override dell'utente (il motore cede a qualunque `OMP_NUM_THREADS` trovi impostata) e lo ri-inchioderebbe a 12. **Verificato, non assunto**: `coli` *lancia* qwen38 — la ricerca del nome nel wrapper non trovava nulla perché vive nel registro famiglie (`family_registry.py:1131`, `engine_artifact="qwen38"`), risolto da `engine_for()`. A runtime il motore stampa `[OMP] qwen38: 20 threads (SMT inclusa) su 24 CPU logiche`, e con `OMP_NUM_THREADS=12` tace e obbedisce: **i run della 0.4 restano riproducibili identici**. Test: `tests/test_omp_tune.c` copre dimensionamento, riserva nulla come no-op, override e kill-switch (verde, 20/24 su questa macchina); `tests/test_v4_cli.py` copre l'esenzione del launcher *e* che una famiglia sorella continui a ricevere il default (92 test python verdi). |
| 13/09 | **0.4 CHIUSA — l'SMT paga, e paga sul pavimento denso: 20 thread = −25,5 %** (stesso protocollo 0.3, cap 128, prompt lungo 274 t. + 16 nuovi = 290 forwards, `COLI_TIMERS=1`, `COLI_CUDA=0`; unica variabile i thread; 20 run totali, ogni gradino ripetuto; **stessa selezione in tutti**: 138720 scelte su 12032 esperti, hit rate 61,5 % (28599/17933) invariato — ulteriore conferma che è path- *e* thread-independent). **Ancoraggio riuscito**: A12 riproduce M128 dell'08/09 entro lo 0,5 % (128,9 vs 128,2 s; TTFT 119,2 vs 118,8; RssAnon 40.225 vs 40.226 MiB). Questo **risolve per riproduzione una domanda che era rimasta assunta**: `qwen38.c` non chiama mai `coli_omp_tune_threads()` (a differenza di inkling/kimi_k3/olmoe/colibri), quindi il numero di thread dipendeva dal percorso di lancio — `./c/qwen38` diretto avrebbe dato i **24 logici** di default libgomp, non 12. Poiché imporre `OMP_NUM_THREADS=12` riproduce M128 bit per bit, **tutta la Fase 0 girava davvero a 12**. Il sampler ora conta i thread da `/proc/<pid>/Threads` invece di assumerli. **La tabella** (mediana dei run validi, secondi per 16 token): *copia* 12 t **128,5** → 16 t **109,9** → 20 t **95,7** (−25,5 %); *mmap* 12 t **134,2** → 16 t **115,4** → 20 t **103,3** (−23,0 %). **Dove cade il guadagno** (ms/fwd, percorso a copia, 12 t → 20 t): `resident-mm` 3648 → **2296 (−37,1 %)**, `deltanet` 2440 → **1661 (−31,9 %)**, `routed-expert` 2408 → 1756 (−27,1 %), mentre `expert-read` resta **piatto** 1108 → 1087 (−1,9 %: è I/O, i thread non lo toccano) e `qsa-attn` piatto a ~149 (non parallelizzato — diventerà una quota crescente man mano che il resto scende). `resident-mm` e `deltanet` scendono **identici sui due percorsi** (2296/2289 e 1661/1660 a 20 t): il pavimento denso è indipendente dal percorso esperti, come la 0.3 aveva previsto. **Efficienza di scaling ~95 %** (`resident-mm` rende 1,59× con 1,67× di thread; 12→16 rende il 98 %, 16→20 il 97 %): a 20 **non c'è segno di saturazione**, il che è la firma del regime bandwidth-bound — l'SMT riempie gli stalli di memoria, esattamente l'ipotesi che la 0.3 aveva lasciato aperta. Il gradino a **24 è escluso per scelta del titolare** (saturare la CPU è controproducente: al K12 di produzione servono thread liberi); 20 resta lo sweet spot operativo. **Il pavimento denso è piatto rispetto al cap, non rispetto ai thread** — ed è la leva più economica della campagna: una variabile d'ambiente sul 76 % del tempo. |
| 13/09 | **0.4b — lo swap falsava le misure: l'inferenza va confinata in una cgroup senza swap** (emerso durante la 0.4, su segnalazione del titolare). Sul percorso **mmap a 16 t** il primo run ha spinto **12,5 GiB** in swap e ha chiuso in **124,7 s**; rifatto con il motore dentro una scope `systemd-run --user --scope -p MemorySwapMax=0` chiude in **115,2-115,5 s** (mediana 115,4) — lo swap costava **~8 %** di tempo su una misura che sembrava valida. Il regime è ora **provato, non dedotto**: il sampler legge `memory.swap.current` della scope *durante* il run (a fine run la scope è già rimossa) e riporta **0 MiB per l'inferenza** in tutti i run del nuovo regime, con `memory.events` a `oom 0 / oom_kill 0` anche col percorso a copia a **40,2 GiB di RssAnon** su 60. Il resto del nodo resta libero di usare zram: è il «swappa tutto tranne l'inferenza» chiesto dal titolare, e non richiede né sudo né `vm.swappiness` globale. **Nota sullo storico**: lo swap di questa macchina è **zram 30,2 GiB compresso in RAM** + swapfile 4 GiB — i «22,4 GiB di swap» del cap 192 (0.3) erano quindi compressione in RAM, non I/O su disco. **Nota metodologica**: a 12 e 16 t il percorso a copia non cambia (128,5 / 109,9 s), perché non swappava comunque (+5 MiB sul baseline zram preesistente); un run a 20 t ha dato **91,9 s** contro i 95,7 canonici e sembrava un guadagno della cgroup — la ripetizione l'ha smentito (95,7 s) e la causa è stata isolata: quel run aveva `expert-read` a **841 ms/fwd** invece di 1087, cioè page cache favorevole sul percorso esperti, 266 ms × 16 fwd ≈ 4,3 s = esattamente il divario. **Un verde isolato non prova nulla**, di nuovo. |
| 08/09 | **0.3 CHIUSA — gradini 96/128/192 su due percorsi, e il verdetto d'ordine** (stesso prompt lungo 274 t. + 16 nuovi = 290 forwards, 12 thread, stessa selezione in tutti i run: 138720 scelte su 12032 esperti distinti). **Copia**: cap 96 → **49,0 %** (22778/23754), expert-read **1329 ms/fwd**, TTFT 123,8 s, 133,7 s per 16 token, RssAnon 33.025 MiB. cap 128 → **61,5 %** (28599/17933), expert-read **1031**, TTFT 118,8 s, **128,2 s** (il minimo della curva), RssAnon 40.226 MiB. cap 192 → **71,4 %** (33225/13307) ma ⚠️ **tempi invalidi**: RssAnon 49.672 MiB su 60 GB, swap di sistema fino a 22,4 GiB, e `expert-read` che *sale* a **2192 ms** mentre i miss si dimezzano — sintomo inequivocabile di thrashing, non di cache. **Mmap** (`COLI_MAP_EXPERTS=1`, che dà il tempo pulito a 192 perché tiene l'anonimo piatto): N128 → 61,5 %, TTFT 125,7 s, **135,4 s**, expert-read **32,5 ms**, routed-expert 3921, RssAnon **12.282** MiB + RssFile 37.772, picco swap 2,3 GiB. N192 → 71,4 %, TTFT 122,7 s, **137,4 s**, expert-read 34,9 ms, routed-expert 4031, RssAnon **13.314** MiB + RssFile 39.864, picco swap 8,6 GiB. **Tre letture.** (1) *Il hit rate è path-independent*: M128/N128 danno 28599/17933 identici, M192/N192 33225/13307 identici — è una grandezza logica (slot × politica), il percorso cambia solo i tempi. (2) *Mmap sposta il costo del fault fra i contatori, non lo elimina*: `expert-read` crolla 1031 → 32 ms ma `routed-expert` sale 2466 → 3921, e il totale è praticamente lo stesso (128,2 vs 135,4 s) — il page fault ora si paga dentro il matmul. Il suo valore vero è **l'anonimo**: 12-13 GiB contro 40-49, cioè la differenza fra girare e swappare. (3) *La curva ha risposto alla domanda per cui esisteva*. A cap 128: `resident-mm` 58,4 s + `deltanet` 39,1 s + `expert-read` 16,5 s su 128,2 → **disco 13 %, denso 76 %**, con il pavimento denso **piatto** (3657/3654/3652 ms/fwd a cap 64/128/192). Il disco non è più il collo di bottiglia: azzerarlo del tutto vale ~13 %, e oltre cap 128 un GiB liberato vale **zero** (N128 135,4 vs N192 137,4 s con +10 punti di hit rate). ⇒ **Fase 2 (VRAM) prima di Fase 1 (int4)**, l'ordine opposto a quello che il piano lasciava aperto: l'int4 comprerebbe RAM e hit rate, cioè il 13 % che resta; solo la GPU attacca il 76 %. Fase 0 **completa**. |
| 08/09 | **0.3 — curva col prompt lungo** (274 token + 16 decode = 290 forwards, stessa selezione dei 3 run: 46532 eventi; il prompt corto lasciava il decode fermo perché il working set ~118/layer non churnava): cap 16 → **7,5 %** (hit 3509 / miss 43023), expert-read **2601 ms/fwd**, decode **10,0 s/tok**, TTFT 148 s. cap 32 → **10,2 %**, expert-read 2307, decode 9,6 s/tok, TTFT 143 s. cap 64 → **26,8 %** (12486/34046), expert-read **1886**, decode **9,2 s/tok**, TTFT 136 s. Il gradiente ora muove il decode: 16→64 = −27,5 % di disco. Asimmetria forte: +2,7 punti da 16→32 ma **+16,6 da 32→64** → la popolarità degli esperti è skewed, gli slot marginali 32-64 sono caldi: 128/192 dovrebbero pagare ancora (da confermare nella finestra K12). **Due fatti architetturali**: (1) `routed-expert` a 2692-2752 ms/fwd, 6,7× più lento che col prompt corto (408) → il matmul esperti non è FLOP-bound ma **bandwidth-bound sulle pagine appena faultate** (4,7 MiB nuovi a esperto, zero locality): VRAM-residenza (Fase 2) e cap alto attaccano la stessa radice; (2) prefetch cala sotto il 100 % sotto carico (78936/86046 → 55416/68092 = 81-92 %) → parte dei miss diventa síncrona. TTFT 136-148 s su prefill 274 a working set freddo (~16-17k esperti unici, 56-80 GiB di disco): questo è il cold-start reale del prompt lungo su questa macchina. PEAK RSS invariato (14,68/18,19/25,22 GB) |
| 08/09 | **0.3 parziale — curva hit-rate, gradini 16/32/64** (stesso prompt 28+16, 43 forwards, percorso copia, 12 thread; working set del run: 5687 esperti distinti = ~118/layer). cap 16 (768 slot): **22,4 %** (hit 2677 / miss 9252), 0,65 tok/s, TTFT 13,3 s, PEAK 14,62 GB. cap 32 (1536): **32,5 %** (3871/8058), 0,61 tok/s, TTFT 15,2 s, PEAK 18,14 GB. cap 64 (3072): **41,3 %** (4928/7001), 0,63 tok/s, TTFT 15,3 s, PEAK 25,17 GB. Gains +10,1 poi +8,8 punti per raddoppio — decrescenti ma vivi; I/O coerente (weight-ranges 18504→16116→14002, prefetched 100 %, batch 1021→828→649). **Lettura**: con 43 forwards il working set per layer (~118 esperti) sta SOTTO cap 192 (192 slot/layer) → a 192 questo prompt arriverebbe a ~70-75 % di hit rate (mancano solo i 5687 primi contatti); su prompt lunghi il working set per layer cresce fino a 512 e la curva va ri-misurata. Decode flat 0,61-0,65 tok/s a questi cap: a 43 forwards il hit rate incide su TTFT/cold-start, non sul decode (disco ~0,5 s su 1,6). 96/128/192: finestra K12 a riposo (comunica il titolare) |
| 08/09 | **0.2 chiuso — scomposizione per token di decode** (cap 32, 12 thread, prompt 28 + 16 nuovi, 16 forwards; stessa riga I/O in tutti i run: weight-ranges 16116, coalesced-gate-up 8058, prefetched 16116 = 100 %, parallel-batches 828, resident-scales 28,12 MiB; hit rate 32,5 % = hit 3871 / miss 8058, 5687 esperti distinti di 25088). Copia, steady state: expert-read **498 ms**, routed-expert 408, shared 40, resident-mm 561 (sovrapposto), deltanet 378, qsa 0,3+4,7, ple 6,2, lm-head 38,2 ms/fwd → **1,64 s/token** (0,61 tok/s), TTFT 15,3 s (prefill 28), RssAnon picco **18,1 GiB** (RssFile ~0). Mappatura sui 4 termini del piano: disco ≈ 498 ms (36 %), expert-matmul ≈ 448 ms (33 %), attention ≈ 382 ms (28 %), lm-head ≈ 38 ms (3 %). **vs recon 96/17/14/2,6 s (disco 69 %)**: stessa fase n.1 a disco freddo, quote molto diverse — 990 PRO + prefetch al 100 % + gate-up coalesced = ~0,5 s/token di disco, ~30× meno del box di riferimento; le fasi CPU (matmul + deltanet + dense) diventano la quota visibile. Conseguenza: su questa macchina la Fase 2 deve attaccare il matmul CPU (routed+resident ≈ 1,38 s/fwd sovrapposto), non solo il disco; il disco resta il problema di TTFT/cold-start (11,5–15,4 s) |
| 08/09 | **0.2b chiuso — copia vs mmap**: pattern di accesso identico bit per bit (riga I/O e hit rate uguali in tutti e 8 i run). Mmap steady state (dal 2° pass): expert-read **26 ms** vs 498 → **1,18 s/token** (0,85 tok/s, 1,38× più veloce), TTFT 11,5 s; primo pass dopo evizione paga re-fault + first-touch (routed-expert 947 ms). RAM: `RssAnon` picco **11,2 GiB vs 18,1** (lo slab mmap non esiste più, −6,9 GiB di anon) + **RssFile fino a 25,9 GiB** di page cache ancorata dai mapping (VmRSS 37,0 GB = somma; file pages recuperabili, ma in pratica sopravvivono alla pressione — a differenza della cache del percorso copia, che qui viene evictata: il run copia «warm» resta lento come il cold, 498 ms). La `PEAK RSS` del binario è `ru_maxrss` (VmRSS, include file pages) — per l'anonimo vero serve `RssAnon` da `/proc/<pid>/status` |
| 08/09 | **0.1 chiuso**: build CPU pulita sul 3900X reale, `qwen38-tiny-check` 8/8 verde (token 8/8 + oracolo numerico) e prefetch-check 16/16 verde; venv torch+transformers 5.16.1 in `/opt/zyonix/backend/qwen38-tiny-venv` |
| 08/09 | **verifica referenze**: censimento e `config.json` ripresi dai 131 header a terra — tutto confermato (152.089 tensori, 25.088 esperti, zero F32, 943 voci non convertite, top-k 10, PLE al layer 2); tutte le citazioni di sorgente reggono sulla working tree (branch `qwen38cuda` = `fd93c41` + solo doc, nessun sorgente toccato) |
| 08/09 | **correzioni**: GPU di node-01 = 5060 Ti + 5070 Ti (sm_120, 32 GiB) — la SPEC aveva per errore le 2×1070 di node-02, cascata §3/§5.2/§6 corretta; §3 riordinato (cap 192 = 36,7 % ≈42 GiB, cap 205 = 39 % ≈45 GiB); 0.3b chiuso dal sorgente (allora numerato 0.5), aggiunto 0.2b, statistiche = 8; totale checkpoint precisato (185,56 GB / 172,82 GiB) |
| 07/09 23:11 | **download COMPLETO e verificato per montaggio** — 185.563.800.832 B, 144 file, 131 shard, 0 residui `.incomplete`; tutti i 131 header safetensors si aprono, tutti i **152.089** offset cadono dentro i file |
| 07/09 23:2x | **censimento dei byte per famiglia** (§1.2, §1.3): esperti **25.088 non 24.576** (c'è il layer MTP), scale FP8 **BF16 non F32**, e una **tabella N-gram da 51,27 GB = 27,6 %** che nessuno aveva contato. Punto 0.3a chiuso (allora numerato 0.4), nuovo punto 0.3b (allora 0.5) |
| 07/09 22:4x | download avviato — 185,6 GB, revisione pinnata, riavviabile, `~/models/Qwen3.8-Flash-Next-FP8`, log in `~/models/.dl-qwen38next.log` |
| 07/09 | `config.json` letta e messa a verbale (§1.1) — multimodalità e le 943 voci non-convertite sono novità rispetto alla ricognizione |
| 07/09 | NVFP4 verificato sui sorgenti e **scartato** (§5); int4 `g4` promosso a Fase 1 (§6) |
| — | **Fase 0 sbloccata** (il checkpoint c'è): attende il via |
