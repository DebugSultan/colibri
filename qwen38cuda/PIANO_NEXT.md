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

node-01: Ryzen 9 3900X (12c/24t, Zen 2: **AVX2 nativo 256 bit, niente AVX512**),
**64 GB DDR4 a 3200** (da NON rialzare — sospetta causa del panic), NVMe
**Samsung 990 PRO 1 TB** singolo, zram 30,2 G.
GPU: RTX 5070 Ti 16 GB + RTX 5060 Ti 16 GB, Blackwell **sm_120** (CUDA ≥ 12.8;
in casa è pinnata la 13.0).

Vincolo di **sviluppo** (sparisce al cutover, §6.5): finché colibri non
sostituisce llama.cpp, sulla stessa VRAM gira il cervello 27B K12 di
produzione, con picco 30.962 MiB su 32 GiB. La Fase 0 non lo tocca; le fasi
che toccano GPU richiedono finestra concordata SOLO nel regime di sviluppo.

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

**Aggiornamento 14/09 — il tetto sale.** Col cutover B2/B5 node-01 è
dedicata all'inferenza: a K12 spento il sistema occupa ~3 GiB su 60 e il
budget cache sale a **50-55 GiB** (cap fp8 ≈ 237-260, int4 gs64 ≈ 420-445).
Onestà sui due regimi: **in produzione K12 torna su** e il tetto riparte da
~45 GiB, salvo rinegoziare il `--cache-ram` di llama-server (default 8 GiB di
prompt cache host: leva reale, documentata). I banchi della Fase 1 corrono in
finestra K12-spento; la config di produzione si tarerà al regime che
coesiste col cervello.

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

| | FP8 (oggi) | int4 g4 gs64 |
|---|---|---|
| per esperto | 4,6881 MiB | **2,6367 MiB** |
| esperti su disco (25.088, MTP incluso) | 123,3 GB | **~69,4 GB** |
| streaming per token | 2,198 GiB | **1,236 GiB** |
| residenti in 45 GiB di RAM | 9.829 = 39 % | 17.454 = **70 %** |
| residenti in 50-55 GiB (K12 spento) | 10.919-12.012 = 43-48 % | 19.394-21.358 = **77-85 %** |
| residenti in 32 GiB di VRAM | 6.989 = 28 % | 12.424 = **50 %** |

Le percentuali sono su **25.088 esperti** (48 layer + MTP; il titolare ha
deciso il 14/09: **anche MTP va a int4** — un container uniforme, quando il
percorso speculativo si attiva i suoi esperti sono già lì). I numeri int4
precedenti di questa tabella (2,4902 MiB, 65,5 GB) erano stime gs128:
corretti al formato gs64 verificato col convertitore (2.457.600 B packed +
307.200 B scale).

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
   metrica della campagna 27B. (⚠️ **corretto 14/09, vedi §6.4**: il BF16
   esiste — è il repo base `Qwen/Qwen3.8-Flash-Next`, rev `de4b8e4d`, stessa
   struttura a 131 shard del nostro FP8. Resta valido il protocollo: l'arbitro
   di banco resta l'FP8 nativo, il BF16 entra come fonte del convertitore e
   come ancore esterne pubblicate.)
4. `w4`/`g4` sono simmetrici, senza zero-point. Sui pesi di esperti MoE di
   solito basta; la coda va **misurata**, non assunta.

### 6.4 La ricognizione dell'ecosistema (14/09) — il design di Fase 1 rivisto

Quattro repo HF esaminati (API + README), e la ricognizione del nostro albero.
Il design di Fase 1 cambia in tre punti, ciascuno con misura alla mano.

**1. gs64, non gs128.** Lo standard dell'ecosistema è il group-scaled int4 a
64 (`expert_gs` nel meta del container): Kreuzzelg su Qwen3.6-35B misura
**KL 0,0795 e cosine 0,99313** contro ancora int8 (per-row: 0,1091/0,98777 —
il group scaling taglia l'errore di logit del 44 %); JustVugg sceglie 64 per
GLM-5.3-Flash «group-scaled int4 within noise of int8, per-row scales cost
materially more». Il nostro gs=128 (calcolato sulle scale sorgente FP8) è
superato: le scale in più costano 2,65 MiB/esperto contro 2,49 (+4 GB su ~70)
e comprano qualità misurata. **Il percorso è già merged nel nostro albero**:
`matmul_q_gs` + `expert_gs` in `qwen36.c:1022/1416-1417` (contatore scale
group-scaled), meta `expert_gs` a :602/:1208. Il container: nibble U8 + `.qs`
F32 `[O][I/gs]` row-major = esattamente il fmt=4 di colibri.

**2. Sorgente BF16 in streaming.** Il repo base `Qwen/Qwen3.8-Flash-Next`
(rev `de4b8e4d`) **è il checkpoint BF16**: il §6.3 «un BF16 non esiste» era
falso — abbiamo scaricato il derivato FP8, non il master. Il convertitore
legge BF16 shard-a-shard (pattern `convert_glm53.py`: picco disco = output +
un solo shard, mai i ~330 GB sorgente) e fa **un solo salto** di
quantizzazione invece di due (FP8→int4 ne compone due). Fallback: FP8 se il
torrent BF16 è lento. Il BF16 completo come arbitro di banco resterebbe un
lusso da 330 GB che non ci sta accanto all'FP8 — non serve: l'arbitro di
banco resta l'FP8 nativo (ciò che il modello è in deploy), con le **ancore
esterne pubblicate** per calibrare le aspettative: ivanfioravanti (DS4-Q4,
Q4_K imatrix gate/up + MXFP4 down, vs BF16) MAE 0,0431 / top-1 96,6 %;
Q8_0 MAE 0,0211 / 98,7 %.

**3. Container = soli esperti.** ~69,5 GB (gs64) — dense e PLE restano letti
dagli shard FP8 come oggi: nessuna duplicazione, e la PLE (51,3 GB) non si
toca. Edge case noto dalla ricetta ivanfioravanti: **24 esperti a zero**
(16 gate/up, 8 down) → fallback deterministico per colonna a energia di peso,
mai divisione per zero nello scale. Riserva qualità se la coda RTN simmetrica
fallisce il banco per-tensore: scale fit pesate imatrix (calibrazione unsloth
pinnata, `unsloth/Qwen3.8-Flash-Next-GGUF` rev `c8b5954a`).

**Cosa NON si usa, e perché.**
- `cyankiwi/...AWQ-INT4` (93,85 GB, 38 shard): compressed-tensors
  pack-quantized su **tutti i Linear** + PLE compressa — per usarlo
  bisognerebbe scrivere un decoder AWQ per l'intero modello invece del nostro
  g4 per i soli esperti. Vale come prova sociale che int4 non collassa il
  modello e come eventuale comparatore/calibrazione, non come artefatto.
- `ivanfioravanti/...DS4-Q4`: engine **ds4/Metal** (Apple Silicon), non
  colibri. Il suo valore è tutto informativo: ricetta, numeri di qualità vs
  BF16, e il **primo dato MTP misurato su questo modello** — 45,4 → 55-56
  tok/s a **un solo draft token per ciclo** (98 % acceptance su prompt
  matematici, ~66 tok/s; `--mtp-draft 7` accettato ma non abilita catene:
  le chain multi-token con verifica S=k+1 sono territorio nostro, e il
  batched verify oltre a moltiplicare i tok/s tira il routed-expert fuori dal
  tetto DRAM di S=1 — D1: S=32 7,5×). PLE come sidecar Q4_1 demand-paged:
  conferma l'architettura «PLE mai residente».
- `nvidia/Qwen3.8-Flash-Next-NVFP4`: ufficiale, esperti-only, 110k download —
  ma la CPU non decodifica NVFP4: resta opzione della sola combo VRAM
  (tensor-core Blackwell nativi), mai del prefill.
- MXFP4 per gli esperti: colibri ha il **solo** percorso denso (fmt=7), nessun
  kernel `grouped_*` — ricontrollato post-merge. Stesso vicolo di NVFP4.

**Il perimetro operativo di Fase 1** (tutto CPU-side, zero finestra):
convertitore (adattare `c/tools/convert_qwen36.py` / `convert_fp8_to_int4.py`
alla geometria qwen38: naming, layer MTP, zero-expert, meta `expert_gs: 64`;
prima un subset col layer 0 per lo smoke, poi l'artefatto completo overnight)
→ loader container int4 in `qwen38.c` (gemella di
`q38_try_load_native_fp8_expert`) → dispatch GEMV group-scaled nel percorso
esperti (pattern D2, `matmul_q_gs` come riferimento) → banchi: per-tensore
(coda + zero-expert) → d3 KL vs FP8 → curva 0.3 + d5 storm + decode steady.
(MTP resta la fase successiva, con spec separata: i numeri ds4 sono il
riferimento onesto di partenza.)

---

### 6.5 La Fase 2 rivista (14/09): due livelli, NVFP4 in VRAM, residenza totale

**Il perimetro prima di tutto: colibri sostituisce llama.cpp, non convive**
(decisione del titolare). Il passaggio di modello è 27B → Flash; a cutover
avvenuto K12 esce e le sue risorse tornano tutte qui: 32 GiB VRAM + ~55 GiB
RAM (sistema ~3 GiB). Le finestre con K12 caricato sono solo il regime di
sviluppo. ⇒ l'obiettivo «TUTTI gli esperti + MTP + KV in GPU+RAM» è il regime
di produzione, non un lusso da finestra.

**La matematica (verificata; gs64 ≈ NVFP4 = 2,64 MiB/esperto — stessi byte:
packed 2,46 MB + scale 307 KB; il NVFP4 scala E4M3 per 16 contro la nostra
f32 per 64):**

| livello | contenuto | GiB |
|---|---|---|
| VRAM | ~8,5k esperti caldi **int4** (~22,4) + MTP int4 (~1,5) + head (1,2) + KV q8_0 2×200k (~4,9, vedi riga 15/09) | ~30 |
| RAM | ~16,5k esperti int4 (~44) + dense (5,4) + workspace (2-3) | ~50-52 |
| SSD | PLE (47,75) + ViT + fallback FP8 | — |

Totale esperti residenti: **25.088/25.088**. Lo storm sparisce; il prefill
batched diventa compute-bound ⇒ lì il GEMM accelera, con la strada standard
su sm_120: dequant-in-kernel + INT8 MMA (i tensor core INT4 non esistono più
da sm_89; il `grouped_s4_wmma` wmma-s4 del codice è path pre-Ada). Per il
decode GEMV i formati valgono lo stesso (memory-bound, byte uguali): la
velocità decode viene dalla residenza, non dal formato.

**NVFP4 declassato a ipotesi condizionata (revisione del pomeriggio 14/09,
spinta dal titolare: «prima di fare casini con la quant»).** Analisi per
ruolo: (a) decode GEMV memory-bound a byte identici ⇒ vantaggio ZERO — e il
decode è l'orizzonte della campagna; (b) capacità VRAM: byte identici ⇒ zero;
(c) qualità vs int4-gs64: non provata (sul 27B il danno NVFP4 era TUTTO nelle
FFN — cioè ciò che gli esperti sono qui); (d) GEMM prefill: l'unico vantaggio
vero, ma smussato — il percorso int4 standard (dequant + INT8 MMA) già copre
sm_120 e il NVFP4 vale ~2× SOLO su quel punto, che diventa collo solo dopo la
residenza totale. E il doppio formato espone il rischio che il titolare ha
nominato: gli esperti caldi (tool-call, Python, italiano) in int4 in RAM
mentre il NVFP4 sta sul freddo = tutti i costi, nessun beneficio. Soprattutto:
**lo skew risolve il piazzamento, non il formato** — se l'hotness deriva, la
promozione dinamica RAM→VRAM deve costare una memcpy, ed è vero solo a formato
uniforme. **Deciso: quant UNICA int4-gs64 ovunque** — un convertitore, un
container, una validazione d3, stage() memcpy. Le lezioni di metodo del 27B
restano in tasca (KL mediana + top-1, mai PPL: 3/3 avrebbe promosso i
peggiori; imatrix su NVFP4 peggiora, +38 %; q8_0 sulla cache KV è gratis,
0,99× — limite: KL vista a ctx 512-8192, il long-ctx va provato sul compito).
**Condizioni di riapertura (tutte e quattro):** (1) residenza totale raggiunta
e prefill-GEMM misurato come nuovo collo; (2) istogramma hits: hot-set
abbastanza stabile da far correre ai residenti VRAM il grosso del GEMM;
(3) d3 verde di qualità vs int4-gs64; (4) il ~2× GEMM giustifica secondo
formato + container + convertitore.

**Gli esperti caldi non sono un set statico**: l'hotness deriva col carico ⇒
il tier diventa il livello sommitale dell'LFRU esistente (ammissione/eviction
dinamiche, copia int4 uploadata con una memcpy); l'istogramma per-esperto
su carichi rappresentativi dimensiona il set VRAM (ancore: upstream 6 %
residenza → 53,8 % hit; le nostre curve 0.3). Serve export del registro hits
(piccolo patch).

**La testa, e l'anomalia che la guarda** (da risolvere prima della Fase 2):
`lm_head` è UN solo tensore BF16 da **1,18 GiB** (~248k vocab) e viene
attraversato per intero a ogni token. Ma 1,18 GiB/step a ~45 GB/s di DRAM
vorrrebbe ~26 ms/step = ~38 tok/s SOLO per la testa, mentre il banco misurato
dice 54 tok/s TOTALI con la testa su CPU. L'accounting non torna: o il motore
la tiene già in un formato più stretto (qdw-int8?), o c'è riuso/cache non
contabilizzato. **Prima di spostarla in VRAM (candidata naturale), una
riga-timer la accerta** — altrimenti si progetta la Fase 2 su un
presupposto falso.

**Riferimenti implementativi**: il percorso standard int4 su sm_120
(dequant-in-kernel + INT8 MMA) non ha bisogno di nulla di nuovo; per la
riapertura condizionata del NVFP4 restano in tasca llama.cpp b10133 (GGML_TYPE_NVFP4 = 40, `mmq-config-blackwell.cuh`, gate MMA
`>= 1200`) come sorgente di studio per i kernel grouped FP4, così
come cuBLASLt FP4 per il GEMM di prefill. Gotchi ereditati che restano: P2P NS
fra le schede + 5060 Ti su link chipset x4 (upload una tantum per stage,
compute residente ⇒ PCIe ammortizzato); `CUDA_VISIBLE_DEVICES` invertito
(logico 0 = 5070 Ti).

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
  `transformers==5.16.1`: venv dedicato torch+transformers, invocato con
  `make PYTHON=<venv-torch>/bin/python`.
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

> **Fatta e misurata il 13/09 — verdetto e numeri in §9.** In breve: paga solo
> quando la fase di riempimento si chiude (a 8 e 32 token di generazione è più
> lenta del baseline, a 128 è `routed-expert` −16/−24 %), e la prossima leva
> non è più il decode ma il prefill, che è ~85 % del wall clock e non passa
> per il tier.
>
> **Rivista il 14/09 — §6.5**: colibri sostituirà llama.cpp (mai convivenza) ⇒
> a cutover la residenza è TOTALE e il prefill compute diventa lavoro del
> tier; il formato VRAM (NVFP4 gate d3, fallback int4) decide lì.

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
| 14/09 | **PORT int4 ATERRATO: loader, batch, prefetch, kernel AVX2 bloccato — e la lezione del falso fallback** (`q38_core.h` +205, `qwen38_matmul.h` +55, `Q38_INT4_SNAP=<dir>` apre il container come secondo store `S4`: stessa machineria `st`, nomi `model.language_model.layers.N.mlp.experts.M.{merged_weight,qs}`). Il kernel `q38_matmul_i4_avx2` replica il blocking fp8-D2 (tile per gruppo, accumulo float nel gruppo e double fra i gruppi): test discriminante **PASS su 4 forme** (gate/up 2560·640, down 640·2560, due code irregolari) — identità bit-per-bit riga-sola vs batch-130 e prefisso batch-100 (**lo split Q38_MV_ROWS=64 attraversato, il caveat del banco è coperto**), errore backward 7,8e-09 vs 2,5e-08 dello scalare (il vettoriale batte il riferimento anche qui), concordanza 1,5e-07, scala NaN avvelena solo la sua colonna, controprove beccate (complemento a due rel 1,9; scale trasposte rel 4e-02). Il loader deriva **gs dai byte di qs** (12·I·H/qb), non da un meta: un container gs diverso si aggancia al suo gs vero o non si aggancia. Priorità int4→fp8→per-expert, batch parallelo e prefetch estesi, scale-bank saltata sui layer int4. **Il falso verde che lo smoke ha beccato**: la prima run mise `merged->dtype!=4` e `qs->dtype!=0` — ma 4 è F8_E4M3 e 0 è BF16; U8 è **3** e F32 è **2**. Risultato: int4 mai agganciato, TUTTO cascato su fp8 **senza** batch paralleli (prefetched=0, parallel-batches=0): −2,3× e nessun crash — senza i contatori I/O sarebbe passato per «funziona». Corretto: la run mista (26/49 layer int4, container ancora in scrittura) dà `routed-expert` **19,2→16,3 s (−15 %** a metà copertura, il verdetto del banco regge nel motore), fasi di controllo piatte, ma `expert-read` 53,9→84,2 s **inquinata dal convertitore che scrive lo stesso NVMe** — la misura A/B pulita (container completo, convertitore fermo, atteso: byte/miss dimezzati) è il prossimo gesto. |
| 14/09 | **A/B pulita a container completo (fp8 vs int4, cap 32, convertitore fermo): +38 % end-to-end, guadagno byte-contabile** — 0,61→**0,84 tok/s** (105,7→76,6 s per 64 tok); `expert-read` 831,6→465,6 ms/fwd = **−43,9 %, esattamente il rapporto di byte del modello** (2,76 vs 4,92 MB/esperto: il guadagno I/O è contabilità, non fortuna); `routed-expert` 18,1→13,2 s (**−27 %** a copertura piena, coerente col −15 % della run mista a metà); PEAK RSS 18,19→15,10 GB (slab più piccole); machineria intatta (prefetched 92k, parallel-batches 4,1k) e **scale-bank azzerata** (96→0 range: le scale int4 viaggiano col tensore). Nota di metodo: la prima A/B era doppiamente inquinata — anche il braccio fp8 girò col convertitore che scriveva lo stesso NVMe; la misura valida è questa, a disco fermo, bracci consecutivi sullo stesso prompt. Prossimo gesto: **d3 qualità** (KL mediana+top-1 vs FP8, banco CPU del 27B). |
| 14/09 notte | **d3: traccia teacher-forced aggiunta al `tf_nll`, DA SMOCCARE al risveglio** — `Q38_TF_TRACE=<f>` scrive per posizione (nll f32, argmax i32); `Q38_TF_REF=<f>` legge la traccia dell'altro braccio e stampa **KL surrogata per posizione (nll_int4−nll_fp8: in media = esatta in nats/tok) mediana/media/max + top-1 agreement**. Protocollo: ref.json dal corpus del 27B (`corpus_calib38.txt`, `tokenizers` py disponibile) → braccio FP8 con TRACE → braccio int4 con TRACE+REF. ⚠️ **il gancio è compilato ma mai eseguito: prima prova del protocollo, poi i numeri** (un verde isolato non prova nulla — il metodo lo richiede due volte oggi). |
| 14/09 | **Banco GEMV AVX2 CHIUSO — int4-gs64 NON è penalizzato su AVX2: a S=1 BATTE l'fp8 di 1,49-1,61×, e il verdetto è slot packed** (`c/tests/bench_qwen38_int4_gemv.c`, due run indipendenti che concordano entro l'1 %). Protocollo: 20 thread, `systemd-run --user --scope -p MemorySwapMax=0`, forme esatte degli esperti (gate/up I=2560·O=640, down I=640·O=2560), S=1/4/12/32, arbitro = il kernel fp8 del motore (`q38_matmul_fp8_avx2` di `qwen38_matmul.h`, che è header-only e quindi pilotabile da un driver giocattolo). **Il banco gira su un pool, non su una matrice**: una proiezione d'esperto è 1,6 MB in fp8 e un ciclo di ripetizioni su di essa misurerebbe la L3 del 3900X (64 MB), mentre il regime che decide è quello DRAM-bound — pool dimensionato a ~384 MiB di pesi fp8 (la stessa cifra di D1, così i numeri fp8 restano confrontabili con quelli già a registro: qui 83,4 GFLOP/s a S=1 contro i 73-75 di D1). **Risultati gate/up, GFLOP/s, scalare / `matmul_i4_grouped` / int4 con riuso del tile / fp8-D2**: S=1 **39,5 / 134,0 / 133,7 / 83,4**; S=4 40,7 / 172,9 / **313,6** / 261,4; S=12 41,0 / 178,9 / **359,6** / 341,4; S=32 41,1 / 180,9 / 333,8 / 336,3. **down**: S=1 38,7 / 121,8 / 126,2 / 81,6; S=4 40,1 / 170,1 / **316,2** / 273,3; S=12 40,7 / 178,1 / **427,4** / 401,1; S=32 40,9 / 183,0 / **418,0** / 391,5. **Lettura**: a S=1 il prior teorico si realizza esattamente — l'fp8 sta sul tetto DRAM (**41,7 GB/s** misurati, banda nominale ~45) e l'int4 gli stessi FLOP li fa leggendo **metà byte** (37,7 GB/s su 0,92 MB invece di 1,6), quindi **1,61× su gate/up e 1,49× su down**: sul decode, che è l'orizzonte della campagna, **int4-in-slot batte fp8-in-slot**, non lo pareggia. A S≥4 il quadro si ribalta ma **non per il formato**: `matmul_i4_grouped` così com'è crolla a 0,44-0,66× perché il suo ciclo sulle righe sta DENTRO quello sulle uscite e ri-decodifica gli stessi nibble una volta per posizione — la leva che D1 aveva già misurato come la più grossa (riuso 3,1× contro vettorizzazione 2,0×) lì semplicemente non c'è. **Per questo il banco ha un quarto braccio**: stesso decode, bloccato come l'fp8 (un gruppo decodificato una volta serve tutte le righe del batch) ⇒ **1,05-1,20× sull'fp8 a S=4/12 e pari a S=32** — senza questo braccio un verdetto rosso sarebbe stato sul kernel che ci troviamo, non sul formato. **Conseguenza per il port**: slot packed confermati (il fallback unpack-at-load int8 di §9, 5,2 MiB/slot, non serve e non va pagato), ma il kernel int4 del prefill va bloccato — è ~40 righe sul modello di `q38_matmul_fp8_avx2`, non un kernel nuovo. **Verifiche di regime** (il banco misura la DRAM o la cache?): pool 32 MiB ⇒ fp8 a 59,2 GB/s, sopra il tetto DRAM, e il vantaggio int4 scende a 1,30× (siamo in L3); pool 384 MiB ⇒ 41,7 GB/s; pool 1024 MiB ⇒ 40,9 GB/s con rapporto **identico** (1,59×) ⇒ 384 MiB è già saturo, la misura è onesta. **Gate di correttezza prima di ogni timing**: i tre bracci int4 devono concordare (max\|scalare−avx2\| = 2,86e-06, max\|scalare−tile\| = 4,77e-06 su rms 4,2172), altrimenti staremmo cronometrando tre calcoli diversi. **Controprove**: (1) offset di gruppo sbagliato nel decode del tile ⇒ max\|scalare−tile\| = **2,60e+01** e uscita 1 *prima* di misurare — il gate discrimina; (2) primo tentativo di controprova, un off-by-`r0` sull'indirizzo di riga, **non beccato** — e la ragione è la stessa già annotata in D2: `Q38_MV_ROWS` è 64 e il banco arriva a S=32, quindi il blocco non si spezza mai e quel bug è *irraggiungibile per costruzione*. Annotato perché vale per il port: chi porterà il kernel bloccato deve testarlo a S>64, altrimenti lo split non è coperto. |
| 14/09 | **Fase 1 — convertitore container int4-gs64 scritto e CHIUSO col gate cross-language** (`c/tools/convert_qwen38_int4.py`, commit `59d3d0c`). Sorgente di default il **BF16** `Qwen/Qwen3.8-Flash-Next` rev `de4b8e4d` scaricato shard-a-shard (curl a blocchi da 256 MB, resume via `.converted.json`, cancellazione di uno shard solo quando il refcount del piano scende a zero); `--indir` sul checkpoint FP8 **solo come sorgente di test** — e non è lo stesso percorso di codice, perché i due layout differiscono strutturalmente (BF16: tensori 3D FUSI `gate_up_proj` [512,1280,2560] con gate = righe 0:640 e up = 640:1280, verificato per cosine 0,99964 contro 0,0086 scambiati / 0,0024 interlacciati; FP8: tensori 2D per-esperto con `_scale_inv` a blocchi 128×128 da espandere e **moltiplicare**). Tutto numpy: torch non è installato da nessuna parte su questa macchina, quindi BF16 (`uint16 << 16` riletto f32) ed e4m3 (LUT a 256 voci che riproduce `E4M3_LUT` di `quant.h`, OCP E4M3-FN, solo 0x7F/0xFF NaN, max finito 448) sono decodificati a mano. **Convenzione del nibble decisa e pinnata: offset binary** (`u = q + 8`), NON il complemento a due di `qwen36.c` — i due differiscono per XOR 0x8 e nessuno dei due lato crasherebbe: la ragione è che `matmul_i4_grouped()` decodifica già esattamente questo layout in AVX2, e che in un decode-dentro-GEMV l'offset binary costa `and`+`sub` per vettore contro `and`+`xor`+`sub` (su AVX2 non esiste lo shift a 8 bit). **Piano scoperto sul checkpoint vero: «49 MoE units x 512 experts = 25088 experts, gs=64»** — coincide col numero di §6.2 senza che sia stato cablato (48 layer + MTP, unità = un layer MoE, e per la MTP i due shard gate+up / down **non sono adiacenti**, quindi il piano risolve un *insieme* di shard per unità invece di assumere uno stride). **Layer 0 convertito davvero** (da FP8, `--limit-shards 1`): `conv 16s  out 1.416 GB  experts 512  zero-proj 0` ⇒ ×49 = **69,4 GB**, il totale atteso. Container verificato sui byte: 1024 tensori, ogni `merged_weight` uint8 **2.457.600 B**, ogni `qs` float32 **307.200 B** (76.800 scale) = 2,6367 MiB/esperto. **Fedeltà contro l'FP8 dequantizzato**, 12 proiezioni su esperti 0/1/255/511: cosine **0,992114-0,994207**, rel **0,1090-0,1272** — che è il valore *teorico* di int4/gs64 su pesi gaussiani (amax ≈ 2,7σ ⇒ step = amax/7 ≈ 0,386σ ⇒ rms uniforme = step/√12 ≈ 0,111σ), non un difetto: la prima soglia che avevo scritto (`rel < 0,05`) era sbagliata **io**, sostituita da un bound bilaterale `0,08 < rel < 0,14` perché un errore sospettosamente *piccolo* vorrebbe dire che il test non misura niente. **Il buco vero era l'auto-consistenza**: il `--selftest` quantizza e dequantizza con due funzioni scritte fianco a fianco, quindi una lettura *coerentemente sbagliata* della specifica ci passa. Proprietà che conta: **cross-language** — i byte che il convertitore scrive devono essere i byte che il motore legge. Pinnata come **dato**: il tool emette `c/tests/fixtures/qwen38_int4_gs64.bin` (`--emit-fixture`, 3604 B, O=32 I=128 S=2 gs=64, `y` di riferimento in float64) e `c/tests/test_qwen38_int4_container.c` lo rigioca dentro `matmul_i4_grouped()` vero. Esito: **max\|dy\| = 1,907e-06 su rms(yref) = 10,9419** (1,7e-7 relativo = puro ordine di somma f32), mentre le due controprove interne mancano di **7,909e+01** (nibble letto in complemento a due) e **9,770e+00** (mappa scale trasposta a [I/gs][O], stessi byte, assegnazione sbagliata). **Controprove di protocollo, tutte beccate**: packing in complemento a due (rel 0,585 → 4,95), floor `s>=1e-8` tolto (divide-by-zero, scale a 0 sui 24 zero-expert), nibble basso/alto scambiati (4,91), byte dell'esperto 0 contro sorgente dell'esperto 1 (**cos 0,000145**), e la divergenza guardia↔registry introdotta apposta (`convert_qwen38.py` vs `convert_qwen38_int4.py`) che il test di guardia ha nominato per esteso. **Guardia e registro aggiornati IN COPPIA** (`convert_fp8_to_int4.py` + `family_registry.py`, che `tests/test_convert_guard.py` confronta): il test che asseriva il letterale `"NOT converted"` è stato sostituito da uno che cicla su `FAMILIES` e pretende `family.converter in message` — un letterale in un test non è un contratto, è una nota; il contratto è che le due mappe non divergano. **E qwen38 diventa coli-driven**: `converter="convert_qwen38_int4.py"`, `converter_accepts=("group_size",)`, `converter_mtp_pass=False` — letto `cmd_convert` di `coli`, costruisce `--repo/--outdir` più `--group-size`, cioè esattamente questa CLI, quindi la ragione documentata per cui la famiglia aveva `converter=""` non vale più (la testa MTP è un'unità del piano, non un secondo passaggio). Il test nuovo è **gate automatico** senza toccare liste condivise (`TEST_RULES` si auto-deriva dalle regole del Makefile). Sottoinsieme layer-0 per il banco della prossima riga: generato dall'FP8 locale in 16 s invece di scaricare ~5 GB di BF16 — legittimo perché **il banco misura tempo di kernel, non qualità**; il container definitivo resta quello dal BF16. |
| 14/09 | **MTP a int4 nel container + banco AVX2 che precede il port** (direttive del titolare). MTP: container uniforme da **25.088 esperti (~69,4 GB)** — quando il percorso speculativo si attiverà i suoi 512 esperti sono già lì in fmt=4 (la mia correzione-tabella di ieri li aveva esclusi in modo incoerente colla riga del re-design a ~69,5 GB: riportati coerenti, percentuali §6.2 su 25.088). AVX2: domanda del titolare — «int4-gs64 è penalizzante su CPU solo-AVX2?» — trasformata in **banco GEMV che precede il port del loader**: forme esatte (gate/up I=2560·O=640, down I=640·O=2560), scalare vs AVX2 vs fp8-D2, S=1/4/12/32. Prior teorico sul 3900X (Zen 2, 256 bit nativi, niente AVX512 e nessuna cosa del disegno lo chiede): a S=1 è DRAM-bound ⇒ −45 % di byte/token (2,25→1,24 GiB) con decode-ALU comparabile (e4m3_decode8 ≈ unpack nibble branchless) e overhead gs64 ≈ +1 mul/64 MAC — **int4-in-slot dovrebbe BATTERE fp8-in-slot**; il rischio vero è la qualità del kernel (pipelining del decode-in-GEMV), quindi banco prima, poi decisione slot packed vs unpack-at-load. Fallback quantificato se banco rosso: unpack-at-load int8 (kernel provato) ⇒ slot 5,2 MiB ⇒ residenza 70 % → ~35-40 %: perde la promessa, non è una via libera. Best case da inseguire: decode pipelinato (decodifica i prossimi 64 B mentre l'fma macina i correnti) + riuso tile per S>1 (D2: riuso 3,1× > vettorizzazione 2,0×). |
| 14/09 | **NVFP4 declassato: quant UNICA int4-gs64 ovunque** (revisione del pomeriggio, spinta dal titolare: «prima di fare casini con la quant»). Analisi per ruolo: decode GEMV memory-bound a byte identici ⇒ vantaggio zero (ed è l'orizzonte della campagna); capacità identica ⇒ zero; qualità non provata vs gs64 (sul 27B il danno era tutto nelle FFN = ciò che gli esperti sono qui); GEMM prefill = unico vantaggio, ma il percorso int4 standard su sm_120 (dequant-in-kernel + INT8 MMA — tensor core INT4 tolti da sm_89 in poi, `grouped_s4_wmma` = path pre-Ada) lo copre già, NVFP4 ~2× SOLO lì e solo dopo la residenza totale. Il rischio doppio-formato che il titolare ha nominato è reale: caldi in int4-RAM con NVFP4 sul freddo = tutti i costi nessun beneficio — e **lo skew risolve il piazzamento, non il formato**: la promozione dinamica RAM→VRAM deve costare una memcpy, vero solo a formato uniforme. Riapertura solo a 4 condizioni (§6.5). Il tier parte int4: stage() memcpy, kernel fmt=4 già merged. |
| 14/09 | **Fase 2 ridisegnata: tier a due livelli, obiettivo «residenza totale»** — decisione del titolare: **colibri sostituisce llama.cpp, non convive mai** (il passaggio di modello È 27B → Flash); a cutover la macchina è tutta nostra (32 GiB VRAM + ~55 RAM, sistema ~3) ⇒ **24.576/24.576 esperti residenti in produzione**: VRAM = caldi NVFP4 (~8-9k) + MTP (~2) + head (1,2) + KV q8_0 (~1,6), RAM = resto int4-gs64 (40) + dense (5,4), SSD = solo PLE/ViT/fallback FP8. Lo storm muore, il prefill diventa compute-bound: **lì** il NVFP4 (tensor core sm_120) conta davvero — per il decode GEMV i formati valgono identici (memory-bound, 2,64 MiB/esperto entrambi), la velocità viene dalla residenza. **NVFP4: il verdetto 27B non si eredita** (era denso vs BF16 — la memoria lo delimitava già: «su un MoE si rimisura da zero»), **il metodo sì**: gate d3 KL mediana+top-1 (mai PPL) su `nvidia/...NVFP4` vs int4 nostro vs FP8, imatrix con e senza (sul 27B la PEGGIORAVA, +38 %); rosso ⇒ tier parte int4 (kernel pronti). Bonus ereditato: **q8_0 sulla KV misurato gratis** (0,99×) e f16-K vantaggio inesistente. Hot set = LFRU a tre livelli, non set statico: istogramma hits per dimensionare (serve export del registro). Riferimenti kernel NVFP4 sm_120 già nei sorgenti llama.cpp b10133. Dettagli in §6.5; correzioni tetto RAM (§3, 50-55 GiB a K12 spento) e tabella gs64 (§6.2) committate insieme. |
| 14/09 | **Fase 1 ri-disegnata sull'ecosistema — gs64 group-scaled, sorgente BF16 in streaming, container esperti-only; e il BF16 esiste** (il §6.3 «un BF16 non esiste» era falso: abbiamo scaricato il derivato FP8, il master `Qwen/Qwen3.8-Flash-Next` rev `de4b8e4d` È il BF16, stessa struttura a 131 shard). Quattro repo HF esaminati — Kreuzzelg qwen36 i4-gs64 (KL 0,0795 / cosine 0,99313 vs int8, il group scaling taglia l'errore logit del 44 % sul per-row), JustVugg GLM-5.3-Flash int4-gs64 («within noise of int8», convertitore `tools/convert_glm53.py`: 25 h per 194,7 GB, picco disco = output + un shard), ivanfioravanti DS4-Q4 **del nostro modello** (Q4_K imatrix gate/up + MXFP4 down, vs BF16: MAE 0,0431 / top-1 96,6 %, Q8_0 0,0211/98,7 %; e il primo dato **MTP misurato**: 45,4 → 55-56 tok/s a un solo draft token per ciclo, 98 % acceptance su math — le chain multi-token con verify S=k+1 sono territorio nostro, D1: S=32 7,5×), nvidia NVFP4 ufficiale esperti-only (CPU non lo decodifica: solo combo VRAM). **Decisioni**: gs64 al posto di gs128 (+4 GB su ~70, qualità misurata); sorgente BF16 shard-a-shard (un solo salto di quant, mai i 330 GB a terra; fallback FP8); container = soli esperti ~69,5 GB con dense/PLE sugli shard FP8 esistenti (PLE mai toccata, 24 zero-expert con fallback deterministico); arbitro di banco = FP8 nativo + ancore esterne. **E il perimetro si è sgonfiato**: `matmul_q_gs` + `expert_gs` sono GIÀ merged nel nostro albero (`qwen36.c:1022/1416`), i convertitori esistono (`c/tools/convert_qwen36.py`, `convert_fp8_to_int4.py`), disco 334 GB liberi — la Fase 1 è adattare, non scrivere da zero. Dettagli e scarti (cyankiwi AWQ full-model, MXFP4 grouped assente) in §6.4. Finestra chiusa fino alle 18: il tier A/B, gli oracle CUDA e lo split 4/4 aspettano la riapertura; nel frattempo ricognizione del port e convertitore subset. |
| 14/09 | **Scomposizione di `routed-expert` CHIUSA — la fault storm è I/O sincrono, non aritmetica, e nemmeno lavoro di page-table** (la domanda aperta della leva D: 27,1 s su 42,7 di TTFT di cui solo 0,95 s di servizio disco misurato; due diagnosi opposte, due leve opposte — matmul fp8 lento o fault impliciti del mapping — e finché non erano separate ogni mossa successiva era una scommessa). **Disegno**: quattro run `N_NEW=1` cap 32, D2 acceso ovunque, `COLI_CUDA=0`, 20 thread, stessa selezione in tutte (hit 404 / miss 38928, riga I/O identica); unica variabile `COLI_MAP_EXPERTS` — sotto mmap il primo tocco dei byte dell'esperto avviene DENTRO il matmul e il fault viene addebitato a `routed-expert`, sotto copia i byte arrivano espliciti e finiscono in `expert-read`; la differenza fra i due `routed-expert` è il tempo di fault (coppie adiacenti M1-C1, M2-C2). **Risultato**: mmap **33,76/29,88 s** contro copia **9,87/9,88 s** — l'aritmetica fp8 vera è ~9,9 s, i fault valgono **20,0-23,9 s = 44-53 % del TTFT**: il primo costo del prefill, più grosso del matmul stesso. **Il discrimine major/minor** (`/usr/bin/time -v` + Δ`pgmajfault` di sistema, scope `MemorySwapMax=0`): run mmap **848.410 major fault** (Δ sistema 849.723 — il processo li rende conto al 99,9 %), run copia **8**; minor comparable (4,08 M contro 4,92 M) ⇒ non è granularità di pagina: **sono stalli di I/O veri dentro il matmul** — le leve hugepage/`MADV_HUGEPAGE` muoiono qui (il costo non è la tabella, è il disco dietro), e il readahead del kernel ci sta già provando: è il motivo per cui mmap vince comunque il TTFT (50,0 contro 58,5 s) pur pagando la stessa roba dentro il timer. `fp8-expand` a zero in tutte le run: nessun lavoro di miss fuori dal matmul, come da nota d'architettura. **Riconciliazione con la nota 0.3** («mmap sposta il costo del fault fra i contatori, non lo elimina»): ora quantificato — mmap 0,94 s di `expert-read` + 34,1 s di `routed-expert` contro copia 30,1 + 12,4 (aritmetica più rumorata in questa coppia: banda 9,9-12,4 s). **Conseguenza di roadmap**: il costo #1 del prefill è latenza di I/O seriale, non calcolo — la leva che lo attacca è la sovrapposizione acquisizione/calcolo (prefill expert-major + acquisizione batched, 410842e+871d793), finora parcheggiata come pezzo del percorso GPU ma il cui beneficio qui misurato è **CPU-side e misurabile senza finestra**. Banco: `d5_fault.sh` (scomposizione) + `d6_fault.sh` (discrimine) con i run `d5_*.txt` e `d6_{M_mmap,C_copy}.txt`. |
| 14/09 | **Leva D3 CHIUSA — i kernel vettoriali sono validati sul modello vero: KL 9,4e-11 nat, sequenze identiche, margine/spostamento 52.181×** (il debito di correttezza di D2: i test unitari dicevano 2,7-10× di accuratezza in più del riferimento long double, ma la distribuzione del modello reale non era stata toccata — finché aperto, D era misurata ma non validata). Protocollo da regola campagna, **mai la PPL**: quattro run greedy ref/ref/vec/vec sul prompt lungo 274 t. + 16 token, `Q38_SIMD_MATMUL` unica variabile nello stesso binario; `DUMP=<path>` scrive i logit float32 dell'ultima posizione — lo strumento esisteva già in `qwen38.c:1861`, zero modifiche al motore; il decode a S=1 passa comunque dai kernel, quindi il banco copre prefill e decode. **Determinismo prima di tutto**: le due run per ramo sono identiche bit per bit sia nei logit sia nella sequenza generata — senza questo il confronto fra rami non sarebbe interpretabile. Esito: sequenze **identiche**; KL(rif‖vet) = KL(vet‖rif) = **9,4e-11 nat** sulla distribuzione completa a 248.320 voci (rumore di rounding float32); top-1 d'accordo (p=0,614225 contro 0,614219), top-5 stesso insieme e stesso ordine; spostamento massimo di probabilità 6,2e-6 contro margine top-1 0,3246 = **52.181×**: lo spostamento dovrebbe essere cinquantaduemila volte più grande per ribaltare la decisione a questa posizione. Con l'accuratezza unitaria già misurata (il vettoriale è più vicino al vero dello scalare), D2 ora è misurata **e** validata. Banco: `d3_kl.sh` + `d3_analyse.py`. |
| 14/09 | **Riprezzamento A1/B1 e del verdetto «Fase 2 prima di Fase 1» — il denominatore della curva 0.3 non esiste più, e il nuovo costo #1 non è il denso** (la curva aveva sentenziato col denso al 76 % del prefill: `resident-mm` 58,4 + `deltanet` 39,1 su 128,2 s; i microbench CUDA `coli_cuda_expert_mlp`/`coli_cuda_matmul` erano stati pensati contro una CPU da 51 GFLOP/s — oggi ne fa 380-427 e S=1 è DRAM-bound, D1: 48,5→73 GFLOP/s è il tetto DRAM, non il kernel). **Dopo D2, stesso protocollo di scomposizione**: `resident-mm` 4,6-5,0 s su 45-50 di TTFT = **10 %**; denso totale col deltanet (6,7-6,9 s) = **25 %**. Il 76 % è memoria storica ⇒ **il soffitto di un denso su GPU è ~25 % del prefill e 0 % del decode**: A1/B1 restano parcheggiati in finestra ma a questo prezzo, e il verdetto d'ordine va ri-derivato, non ritirato per abitudine. La nuova scomposizione dice: il costo #1 è la **fault storm** (20-24 s, riga sopra) — né il tetto denso la tocca, né la Fase 1 la spegne (l'int4 dimezza i byte per esperto, 4,92→2,46 MB: meno byte unici da faultare a freddo e slot a raddoppio di hit rate, ma i 38.928 miss restano miss); il tier VRAM resta giustificato dal decode (−16/−24 % misurato a 128 token), non più dal denso. **Prossima leva candidata, CPU-side e senza finestra**: sovrapposizione acquisizione/calcolo nel prefill (expert-major + acquisizione batched) — da portare col banco toy e misurare QUI prima di qualunque atterraggio; se il guadagno regge, ridefinisce anche quanto vale la prossima finestra GPU. |
| 14/09 | **Leva D chiusa: A/B end-to-end misurato, prefill da 100,7 s a 42,7-53,2 s.** Quattro run alternate OFF/ON/OFF/ON nello **stesso binario** (`Q38_SIMD_MATMUL`), `N_NEW=1` (zero forward di decode: ogni numero è prefill), cap 32, `COLI_CUDA=0`, 20 thread, K12 fermo, stesso token emesso da tutte e quattro (`<|im_end|>`), swap mai toccato (SwapFree ~34,4 GB costante). **TTFT, solo coppie adiacenti**: A(off) 100,82 → B(on) 53,24 s = **−47,2 %, 1,89×**; C(off) 100,50 → D(on) 42,72 s = **−57,5 %, 2,35×**. Le due run OFF concordano entro lo **0,3 %**, le due ON divergono del 25 %: tolto il collo di bottiglia aritmetico la run diventa I/O-sensibile e la varianza si sposta lì, quindi **il numero onesto è la coppia peggiore, 1,89×**. **Dove va il guadagno** (s/forward, off → on): `resident-mm` 33,94/34,12 → 5,11/4,73 = **6,6-7,2×**, cioè **esattamente il 6,3-7,5× del microbench D1** — la leva rende sul suo bersaglio diretto per intero; `deltanet` 24,60/24,72 → 6,92/6,76 = 3,6×; `shared-expert` 2,40/2,42 → 0,43/0,39 = 5,6-6,2×; `ple` 0,39 → 0,13/0,14 = 2,7×; `routed-expert` 53,66/52,81 → 37,01/27,06 = **1,45-1,95× soltanto**, perché è l'unica fase che paga anche il disco. `qsa-attn` 2,12 → 2,12 **invariato**, come deve essere: non passa da `q38_weight_matmul`, ed è il controllo negativo che dimostra che stiamo misurando la leva e non il rumore. `expert-read` 1,06/1,38 → 0,95/0,95 (servizio disco, invariato). **Perché anche gli esperti guadagnano**: nel prefill batched gli esperti routed passano da `q38_weight_matmul` con `count` righe (qwen38_core.h:2021-2028), quindi prendono anche loro il riuso del tile decodificato — non solo la vettorizzazione. **Limite noto e invariato**: a S=1 entrambi i formati stanno sul tetto DRAM, quindi **D non fa nulla per la generazione di token**; il `lm-head` (una riga sola) lo conferma restando nel rumore. Cache esperti 1,0 % di hit e I/O identico in tutte e quattro le run (38928 miss, 64488 prefetch): il modello è 185 GB su 60 GB di RAM, l'A/B non ha spostato nulla sul disco — e questo è ciò che rende il confronto pulito. |
| 14/09 | **Leva D (matmul densi vettoriali) — D1 misurata, D2 atterrata nel motore** (il guadagno end-to-end, misurato subito dopo, è nella riga sopra). **D1 su banco** (20 thread, pool 384 MiB, due build indipendenti che concordano entro 2-6 %): le leve sono **due e indipendenti** — vettorizzare il decode (8 pesi e4m3 → `__m256`) e **riusare il tile decodificato** servendo tutte le righe del batch da un blocco decodificato una volta sola. Forma esperto gate/up (I=2560, O=640), GFLOP/s scalare → vettore → vettore+riuso: S=1 **48,5 / 75,2 / 73,1**; S=4 50,5 / 100,8 / 205,5; S=12 51,1 / 104,2 / **323,5 (6,33×)**; S=32 51,4 / 105,5 / **383,8 (7,47×)**. Esperto down (640→2560) S=32: 51,0 / 105,1 / **377,6**. bf16 v-proj (2560→6144) S=274: 56,2 / **427,3 (7,60×)**; o-proj (6144→2560) S=274: **390,7 (6,95×)**. **Il riuso vale più della vettorizzazione** (3,1× contro 2,0×): il collo non era l'aritmetica, era decodificare lo stesso peso una volta per posizione. **A S=1 non cambia nulla** — 48→73 GFLOP/s è il tetto DRAM (~45 GB/s), non il kernel: **D è una leva di prefill e non tocca la generazione token**, ed è il limite onesto della leva. Due ipotesi mie smentite dalla misura: il blocking su bf16 **peggiora** (pesi doppi, decode = uno shift) e il gather `vpgatherdd` su Zen 2 è microcodato, vicolo cieco. **D2 nel motore**: nuovo `c/qwen38_matmul.h` header-only (zero dipendenze da `Model` — è ciò che lo rende pilotabile da un banco giocattolo, dato che `qwen38_core.h` non compila da solo), due kernel AVX2, e `q38_weight_matmul` che dispatcha su `Q38_SIMD_MATMUL` (default 1, `0` = percorso di riferimento): **un solo punto di dispatch per tutti i matmul densi** (verificato col grep: ~305, 1807-1817, 1970-1977, 2017-2024, 2139), quindi i due rami di un A/B stanno **nello stesso binario**. Semplificazione rispetto a D1: **un solo kernel fp8 blocked** invece di due con dispatch su S — a S=1 degenera in «vettore con un buffer» (73,1 contro 75,2: −2,8 % su un percorso comunque DRAM-bound) in cambio di metà codice. **Riuso a monte**: `quant.h` spediva già `e4m3_decode8`/`bf16_decode8` (lo stesso trucco sui bit derivato in D1) e `deepseek_v41.c` era arrivato indipendentemente alle stesse due conclusioni in `mv8_rows`/`mvb`; `matmul_fp8` è lasciata intatta perché condivisa con altri motori. **Accuratezza — questa sì misurata**: contro riferimento `long double`, errore backward (Σ\|w·x\| al denominatore, non \|risultato\|: con attivazioni casuali un prodotto scalare da 2560 termini si cancella e dividere per il risultato misurerebbe la cancellazione, non il kernel) il vettoriale è **2,71×** meglio su fp8 gate/up S=1, 3,08× a S=100, 3,65× su down, **10,18×** su bf16 v-proj — quattro catene di accumulo parallele arrotondano meno di una seriale. Il 2,71× coincide col 2,7× che D1 aveva misurato per strada indipendente. **Test discriminanti, non verdi isolati**: `tests/test_qwen38_simd_matmul.c` pinna (1) identità bit per bit di una riga calcolata da sola contro la stessa dentro un batch di 100 — `Q38_MV_ROWS` è 64, quindi il batch **si spezza davvero**; (2) che il vettoriale sia più vicino al vero dello scalare (una sola tolleranza promuoverebbe un kernel diventato peggiore); (3) che un peso NaN avveleni la sua uscita e **solo** quella; geometrie con code irregolari (I non multiplo di 128, ultimo blocco non multiplo di 8) per i percorsi di resto. Controprova: iniettando un off-by-one nell'offset di riga → 9 fallimenti, con l'asserzione di identità-sotto-split che scatta su ogni caso batched e **tace a S=1**, dove quel bug è per costruzione invisibile. **Due difetti trovati dalla suite e corretti**: (a) `test_qwen38_native_weights` asserisce col `memcmp` che l'fp8 del dispatcher sia **bit-identico** a un riferimento indipendente — invariante che il vettoriale non può rispettare (ordine di somma diverso, e migliore). Riparato preservandone l'intento: il test **forka e gira entrambi i rami** (il flag è letto e cachato una volta, non si ribalta in-process; il fork precede ogni regione OpenMP), ramo di riferimento bit-esatto come prima, ramo vettoriale bit-esatto **ovunque tranne** i due soli confronti col riferimento indipendente, dove passa a errore relativo. I confronti percorso-contro-percorso (batched vs seriale, prefill vs decode, parità deltanet) **restano bit-esatti in entrambi i rami** ed è la proprietà più forte: i kernel vettoriali sono bit-identici **a sé stessi** sotto qualunque split. Controprova con difetto sulle scale di blocco: il ramo di riferimento passa (il difetto è solo nel vettoriale) e il vettoriale lo becca. (b) **Il Makefile non elencava `qwen38_matmul.h` fra le dipendenze di nessuna regola** tranne quella del suo test: la prima controprova è girata su un binario **stale** e ha dato un falso verde. Aggiunto alle 8 regole che dipendono da `qwen38_core.h`. **Metodologia**: la prima «suite verde» era anch'essa falsa — `make test-c \| tail -25` restituisce l'exit di `tail`, non di `make`; la coda diceva `FAILED: tests/test_qwen38_native_weights`. **Debito A/B end-to-end chiuso dalla riga sopra** (misurato subito dopo, stessa finestra): il microbench non si traduce 1:1 perché nel prefill reale i matmul densi convivono con l'I/O da disco degli esperti, e infatti il guadagno end-to-end (1,9-2,4×) è minore di quello sul kernel isolato (6-7×), che però si ritrova **intatto** nel timer `resident-mm`. |
| 14/09 | **Porte n.2-5 atterrate senza finestra GPU — e la prima fake-backend test suite del tier** (tutta la verifica gira su CPU; i debiti di misura restano in coda, vedi parcheggi). **(A) `inflight` per `q38t_fill_wait`** (df44b87): il tier aveva lo stesso difetto — `qn` libera l'anello alla DEQUEUE, prima del copy, quindi «coda vuota ≠ residente» e il warmstart contava miss un esperto ancora in upload (lettura disco ridondante + offerta sprecata al primo token). Nuovo contatore `inflight`: incrementato all'enqueue, decrementato dall'uploader SOLO a upload completato o abbandonato, `fill_wait` attende su quello; broadcast di `cv_take` anche a completamento (prima solo alla dequeue). **Controprova col bug rimesso**: col vecchio `while(G.qn>0)` il nuovo test fallisce 3/3 run in modo deterministico; col fix torna verde. **(B) La suite che mancava** (93c981b): `tests/qwen38_fake_cuda.h` (fake backend che registra il traffico, con hook slow-upload per rendere deterministica la race) + `test_qwen38_tier.c` (accounting = somma footprint ≠ vecchia formula a slack piatto; budget auto = headroom misurato; clamp del budget esplicito; `COLI_GPU` singolare; piano che riserva, `q38t_cancel_plan` che rilascia; used ≤ budget; split home 4/4 su due device) + `test_qwen38_tier_fill_wait.c` (la proprietà «al ritorno del wait tutto è residente», senza poll). Auto-greggi via `TEST_RULES` dalle regole Makefile: aggiungere la regola basta. **(C) Budget esplicito clampato** (lezione 6cbd5aa): `CUDA_EXPERT_GB` sopra l'headroom misurato (`free − Q38T_DEV_RESERVE`) viene clampato e il banner lo annuncia — prima la coda falliva esperto dopo esperto finché il clamp non congelava il tetto. **(D) Affinity** (792b6f2): `q38t_aff_widen/restore` — con OMP_PROC_BIND l'uploader pthread e i thread del runtime CUDA nascevano intrappolati sulla maschera del master OpenMP (qui 20 thread di default): misurato upstream +64 % di CPU per upload. Maschera allargata a tutti i CPU online attorno a `coli_cuda_init` e alla creazione dell'uploader, ripristinata dopo. **(E) Igiene**: fallback `COLI_GPU` (il singolare che scrive il planner) in `q38t_init`; commento dell'header allineato all'accounting vero (6,02 MiB a granularità cudaMalloc ⇒ ~4.300 esperti su ~12,5 GiB per card da 16, non più «4,69 MiB ⇒ 6.300»). **La suite ha poi pescato un difetto di merge**: `test_omp_tune.c` merged chiamava `env_set`/`env_unset` (gli helper statici della versione upstream, persi nell'auto-merge) — normalizzato su `setenv`/`unsetenv` di compat.h; e la prima «verifica» verde era il binario STALE pre-merge: cancellato e ricostruito prima di fidarsi. Build `CUDA=1` verde, `test-c` in suite verde. **PARCHENZE IN FINESTRA GPU** (non si può misurare senza K12 in pausa): (1) misura end-to-end del budget post-accounting — determinismo del tetto, `failed 0`, residenza % reale a 128 token; (2) ri-misura fadvise-prefetch e reader pool parallela sotto saturazione (53669e2: entrambi possono essere net-negativi — se lo sono qui, si tolgono); (3) prefill expert-major + acquisizione batched (410842e+871d793) ed esposizione del `coli_cuda_expert_group()` sincrono — il refactoring è portabile e verificabile bit-identical col banco toy, ma atterrarlo senza la misura sarebbe scommettere contro la metodologia della campagna; (4) Fase 3 placement shared-expert/router a prezzo bytes-per-byte (ff13134). |
| 14/09 | **Porta n.1 atterrata — l'accounting del tier addebita alla granularità `cudaMalloc`, senza finestra GPU** (la misura end-to-end del nuovo budget resta in debito per la prossima finestra). Venduta da upstream la `dev_alloc_footprint()` (d0a382d+40ff645, curva misurata su driver 5xx: >1 MiB → multipli di 2 MiB, >512 KiB → 1 MiB, sotto → +1/16 a passi di 8 KiB, minimo 8 KiB) e sostituito il vecchio `3*mat + 3*sc*4 + 4096` — che dichiarava **4,69 MiB** a esperto dove l'allocatore ne prende **6,02** (+28,3 %, geometria fp8 3×1,56 MiB + 3 scale da 400 B): ogni budget era sovracompresso e il clamp `G.budget[hd]=G.used[hd]` congelava per sempre il tetto alla prima OOM. Curva verificata **identica bit per bit** a quella di `qwen36_tier.c` su sweep a 14 punti (test usa e getta in /tmp); build `CUDA=1` verde. Ora il numero pianificato è il numero residente; il ~22-28 % che la granularità costa è reale — solo un'arena unica per device lo recupererebbe (aperto, come da nota upstream). |
| 14/09 | **Allineamento a `upstream/main` (JustVugg, +173 commit da `fd93c41`) — merge verde, un solo conflitto, e cinque porte identificate nel registro upstream**. Procedura: ff sul ramo thread di `origin` (7badc1c), poi merge di upstream; l'unico conflitto è `c/Makefile`, dove upstream aveva reso il target `qwen38` **CPU-only deliberato** («no CUDA/Metal tier advertised by the control plane») nella stessa regione del nostro blocco `ifeq (CUDA,1)` — risolto tenendo il blocco CUDA-aware del ramo e riprendendo da upstream il target `deepseek_v41` che affiancava. Tutto il resto auto-merge: `qwen38.c` (+47: EMAP/HITS, 7a8eec2), `qwen38_core.h` (+29: HITS lazy sotto lock, fc6e7bb — **segfault classe CUDA+warmstart, esattamente la nostra configurazione**: preso intatto), `coli` (+40), clang-warning cleanup. **Verifica**: build CPU verde, `qwen38-ple-prefetch-check` 16/16 e `qwen38-tiny-check` 8/8 con il venv torch dedicato, build `CUDA=1` verde (tier linkato a cublas). **La ricognizione dei 173 commit** dà un top-5 di porte per `qwen38_tier`: (1) **accounting a granularità cudaMalloc** (d0a382d+40ff645): addebitiamo 4,70 MiB a esperto ma cudaMalloc ne prende 6,03 ⇒ budget sovracompresso ~28 % e il clamp `G.budget[hd]=G.used[hd]` congela il budget per sempre alla prima OOM — la stessa geometria fp8 misurata da upstream su qwen36; (2) **contatore `inflight`** per `q38t_fill_wait` (df44b87): dequeue *prima* del copy ⇒ «coda vuota ≠ residente», un esperto ancora in upload conta come miss al primo token; (3) **fake-backend test suite** (93c981b, `tests/qwen36_fake_cuda.h`): il tier non ha alcun test — invarianti budget/residui/coda senza GPU; (4) **prefill expert-major + batched acquisition** (410842e+871d793) e i due **esiti negativi** documentati (53669e2): il nostro fadvise-prefetch e la reader pool parallela possono essere net-negativi sotto saturazione — da rimisurare, non dare per scontati; (5) **placement shared-expert/router a prezzo bytes-per-byte** (ff13134) come Fase 3, più il fix affinity OMP (792b6f2: con 20 thread l'uploader pthread eredita la maschera del master e perde ~64 % di CPU) e il fallback `COLI_GPU` in `q38t_init`. Igiene merge: fc6e7bb e 7a8eec2 presi intatti; `TRUNK_RESIDENT_LAYERS` (d4788a7) rifiuta correttamente i backend GPU — interazione nulla col tier. |
| 13/09 | **Fase 2 — tier VRAM implementato e misurato: l'ammortamento esiste, ma solo dopo che la coda di riempimento si chiude** (via esplicito del titolare: «20T fisso + inizio del supporto cuda»; K12 di produzione fermo nella finestra concessa). **Il pezzo**: `c/qwen38_tier.{c,h}`, tutto sotto `#ifdef COLI_CUDA` con stub inline nell'`#else` — la build CPU non compila nemmeno il `.c` e resta a costo zero, verificato ricostruendo entrambi i target. **Il vincolo è invertito rispetto a `qwen36_tier`**: lì `qwen36_tier.h:27-28` impone `cap_experts_per_layer == n_experts` perché il tier tiene puntatori grezzi negli slot RAM; qui gli slot **sono** una cache con sfratto (185 GB di modello su 64 GB di macchina, e con `COLI_MAP_EXPERTS=1` lo slot punta dentro un mapping di file), quindi il tier **possiede la propria copia in VRAM** e **non può andarsela a prendere da sé**: il motore gliela **offre**, e la promozione si decide dentro `q38t_offer` — nessun tick periodico LFRU, solo il decadimento del calore. **Zero conversione**: con `native_fp8` lo slot RAM è già E4M3 grezzo + scale per blocco 128×128, bit per bit ciò che `fmt=8` del backend consuma — `stage()` è tre `memcpy` e uno per le scale, niente XOR di nibble int4 come in `qwen36_tier.c:63`. **La curva dell'ammortamento** (prompt lungo 274 t., `OMP_NUM_THREADS=20`, `COLI_MAP_EXPERTS=1`, unica variabile `COLI_CUDA`, coppie **adiacenti** OFF/ON ripetute): 8 token **+8,0 %** (più lento) → 32 token **+8,8 % / +11,3 %** (più lento) → 128 token **−16,9 % / −24,2 %** sul `routed-expert`, cioè wall clock 163,7→147,7 s (−9,8 %) e 150,6→138,2 s (−8,2 %). **Il perché è nella telemetria, non nell'interpretazione**: a 32 token `offers 4858, promotions 4854` = 99,9 % di promozioni, il budget (~6236 esperti) non si riempie **mai** e ogni token paga staging nuovo senza ancora raccogliere; a 128 token `promotions 5219, swaps 832, queue-full 0` — la fase di riempimento **chiude** verso il passo ~45, da lì in poi restano il 69,4 % di righe servite da GPU e i 13449 miss non costano più nulla. **Determinismo provato**: D/D2/D3 danno telemetria identica bit per bit (`gpu 10022, cpu 4858, 67,4 %, row-overflow 4, take-fail 0`), G1/G2 idem (`gpu 42292, cpu 18668, 69,4 %, row-overflow 18, take-fail 0`); l'aritmetica torna, 10022+4858 = 14880 = 31×48×10. **Il tetto a 8 righe morde davvero**: `coli_cuda_expert_group_issue` rifiuta `total > 8` righe per device e tiene un solo issue in volo, quindi un issue oltre il tetto non è divisibile — le righe in eccesso cadono su CPU via maschera, 4 su 14880 a 32 token e 18 a 128. Era previsto in `qwen38_tier.h:119-128`, ora è **misurato**. **Nota metodologica che vale per tutta la campagna**: la page cache dell'NVMe si scalda fra run consecutivi e produce una deriva monotona (TTFT 94,78 → 88,40 dentro la stessa tornata; il baseline OFF a 128 token è sceso da solo 163,7 → 150,6 → 146,9 s in tre tornate) — **solo le coppie adiacenti sono confrontabili**, il primo run di una tornata è sistematicamente penalizzato, e la grandezza stabile fra tornate è il `routed-expert`, non il wall clock. È esattamente questo che mi ha fatto annunciare un −3,7 % a 32 token che non esisteva: con le coppie adiacenti il tier lì è **più lento** del 2,2/2,4 %. **Prossima leva, e non è più il decode**: il TTFT è ~85 % del wall clock (88,40 s su 103,7 a 32 token) e il tier tocca **solo** il decode — il prefill passa per il `coli_cuda_expert_group()` sincrono che l'intestazione del tier aveva già segnato come «si fa dopo aver misurato il decode». Il decode ora è misurato. |
| 13/09 | **Fase 2b — due difetti trovati col warmstart, corretti e ri-misurati; uno dei due non ha dato ciò che avevo previsto.** (1) **La riserva di VRAM era una stima a occhio, sbagliata di 2,3 GB — ma il margine che speravo di recuperare non c'era.** Il budget si calcolava come `free − 1 GiB`, dichiarando 14,2/14,3 GB; il run produceva due `[CUDA] tensor allocation: out of memory` e il clamp adattivo dell'uploader (`G.budget[hd]=G.used[hd]`) si assestava a **11,9/12,0 GB**. La causa è di **ordine**, non di quantità: il backend alloca i propri tensori di lavoro e i workspace cuBLASLt **dopo** che il tier ha già preso i pesi, quindi il «free» letto in `q38t_init` è il valore più ottimistico dell'intero run. Corretto a `Q38T_DEV_RESERVE = 3,5 GiB`, col clamp lasciato come rete. **Esito: `failed 0` (da 2 a 128 token e 40 nel warmstart), zero OOM, budget 11,7/11,8 GiB stabile dall'inizio — ma residenza 20,9 %, cioè 0,3 punti in MENO del 21,2 % che il clamp raggiungeva.** La mia previsione di «~25 % di residenza sprecata» era sbagliata: al backend servono davvero ~3,3 GiB e ~12 GiB è il tetto vero. Quello che la correzione compra non è residenza, è che il budget ora è **deterministico** invece di essere scoperto andando in OOM — e il clamp torna a essere una rete invece del meccanismo. (2) **Il warmstart buttava il 46 % del proprio lavoro di disco.** Telemetria prima: `offers 11367, promotions 5257, queue-full 4535`, col motore che stampa `warmstart: 9792 experts in 19,0 s` — **5257 + 4535 = 9792 esatti**. La coda è `Q38T_QCAP 16`, dimensionata per il percorso caldo; in warmstart il motore offre più in fretta di quanto l'uploader dreni, l'offerta viene scartata, lo scarto **restituisce correttamente il budget**, e il giro dopo `q38t_plan_fill` ripianifica lo stesso esperto: letto dal disco, copiato e buttato. La correzione è **asimmetrica di proposito** — le offerte con `planned=1` ora **aspettano** posto sulla condvar `cv_take` invece di rinunciare (il warmstart è per definizione una fase «riempi e aspetta», `q38t_fill_wait()` esiste per quello), mentre il percorso caldo continua a scartare perché lì c'è un token che aspetta; dopo l'attesa lo slot viene ricontrollato, il lock è stato rilasciato. **Esito: `queue-full 4535 → 0` e `warmstart: 5145 experts in 16,6 s`** — il 47 % di letture da disco in meno, 2,4 s più veloce, e i 5145 sono **esattamente** il budget. **Coppia adiacente a 128 token dopo le correzioni**: 146,9 → 142,1 s (−3,3 % wall), `routed-expert` 509,06 → 426,40 ms/fwd (**−16,2 %**), 69,0 % su GPU, `failed 0, queue-full 0, take-fail 0`. Col warmstart acceso: 136,5 s, `routed-expert` 409,79 ms/fwd, **72,3 % su GPU**, `swaps 1458` — il warmstart riempie il budget prima del primo token e durante il decode restano solo gli scambi. |
| 13/09 | **0.5 CHIUSO — i 20 thread sono il default, e la politica vive nel motore** (via esplicito del titolare: «20T fisso, non userò altri modelli»). Tre tocchi, nessuno dei quali cambia il default degli altri motori. (1) **`c/omp_tune.h`**: nuova `coli_omp_tune_threads_smt(engine, reserve_cores)` accanto alla sorella `coli_omp_tune_threads()`, con sopra il perché le due politiche sono *opposte* e la tabella della 0.4 che lo autorizza. Tiene l'SMT e riserva **core fisici interi** — mezzo core non lascia niente, il fratello rimasto continua a contendere l'unità vettoriale — con **pavimento ai core fisici**, così su un host senza SMT non può peggiorare ciò che c'è già: `logical − reserve·(logical/phys)`, che sul 3900X fa **24 − 2·2 = 20**. (2) **`c/qwen38.c`**: `coli_omp_tune_threads_smt("qwen38", 2)` subito dopo il banner, prima di qualsiasi regione parallela — il motore non chiamava **nessun** tuner (era il buco scoperto nella 0.4: a lancio diretto prendeva i 24 logici di libgomp). (3) **`c/coli`**: qwen38 diventa la **seconda eccezione** al default a core fisici, con la stessa forma di `deepseek_v4` e per la stessa ragione — il runtime si dimensiona da sé, e un `setdefault` del launcher sarebbe indistinguibile da un override dell'utente (il motore cede a qualunque `OMP_NUM_THREADS` trovi impostata) e lo ri-inchioderebbe a 12. **Verificato, non assunto**: `coli` *lancia* qwen38 — la ricerca del nome nel wrapper non trovava nulla perché vive nel registro famiglie (`family_registry.py:1131`, `engine_artifact="qwen38"`), risolto da `engine_for()`. A runtime il motore stampa `[OMP] qwen38: 20 threads (SMT inclusa) su 24 CPU logiche`, e con `OMP_NUM_THREADS=12` tace e obbedisce: **i run della 0.4 restano riproducibili identici**. Test: `tests/test_omp_tune.c` copre dimensionamento, riserva nulla come no-op, override e kill-switch (verde, 20/24 su questa macchina); `tests/test_v4_cli.py` copre l'esenzione del launcher *e* che una famiglia sorella continui a ricevere il default (92 test python verdi). |
| 13/09 | **0.4 CHIUSA — l'SMT paga, e paga sul pavimento denso: 20 thread = −25,5 %** (stesso protocollo 0.3, cap 128, prompt lungo 274 t. + 16 nuovi = 290 forwards, `COLI_TIMERS=1`, `COLI_CUDA=0`; unica variabile i thread; 20 run totali, ogni gradino ripetuto; **stessa selezione in tutti**: 138720 scelte su 12032 esperti, hit rate 61,5 % (28599/17933) invariato — ulteriore conferma che è path- *e* thread-independent). **Ancoraggio riuscito**: A12 riproduce M128 dell'08/09 entro lo 0,5 % (128,9 vs 128,2 s; TTFT 119,2 vs 118,8; RssAnon 40.225 vs 40.226 MiB). Questo **risolve per riproduzione una domanda che era rimasta assunta**: `qwen38.c` non chiama mai `coli_omp_tune_threads()` (a differenza di inkling/kimi_k3/olmoe/colibri), quindi il numero di thread dipendeva dal percorso di lancio — `./c/qwen38` diretto avrebbe dato i **24 logici** di default libgomp, non 12. Poiché imporre `OMP_NUM_THREADS=12` riproduce M128 bit per bit, **tutta la Fase 0 girava davvero a 12**. Il sampler ora conta i thread da `/proc/<pid>/Threads` invece di assumerli. **La tabella** (mediana dei run validi, secondi per 16 token): *copia* 12 t **128,5** → 16 t **109,9** → 20 t **95,7** (−25,5 %); *mmap* 12 t **134,2** → 16 t **115,4** → 20 t **103,3** (−23,0 %). **Dove cade il guadagno** (ms/fwd, percorso a copia, 12 t → 20 t): `resident-mm` 3648 → **2296 (−37,1 %)**, `deltanet` 2440 → **1661 (−31,9 %)**, `routed-expert` 2408 → 1756 (−27,1 %), mentre `expert-read` resta **piatto** 1108 → 1087 (−1,9 %: è I/O, i thread non lo toccano) e `qsa-attn` piatto a ~149 (non parallelizzato — diventerà una quota crescente man mano che il resto scende). `resident-mm` e `deltanet` scendono **identici sui due percorsi** (2296/2289 e 1661/1660 a 20 t): il pavimento denso è indipendente dal percorso esperti, come la 0.3 aveva previsto. **Efficienza di scaling ~95 %** (`resident-mm` rende 1,59× con 1,67× di thread; 12→16 rende il 98 %, 16→20 il 97 %): a 20 **non c'è segno di saturazione**, il che è la firma del regime bandwidth-bound — l'SMT riempie gli stalli di memoria, esattamente l'ipotesi che la 0.3 aveva lasciato aperta. Il gradino a **24 è escluso per scelta del titolare** (saturare la CPU è controproducente: al K12 di produzione servono thread liberi); 20 resta lo sweet spot operativo. **Il pavimento denso è piatto rispetto al cap, non rispetto ai thread** — ed è la leva più economica della campagna: una variabile d'ambiente sul 76 % del tempo. |
| 13/09 | **0.4b — lo swap falsava le misure: l'inferenza va confinata in una cgroup senza swap** (emerso durante la 0.4, su segnalazione del titolare). Sul percorso **mmap a 16 t** il primo run ha spinto **12,5 GiB** in swap e ha chiuso in **124,7 s**; rifatto con il motore dentro una scope `systemd-run --user --scope -p MemorySwapMax=0` chiude in **115,2-115,5 s** (mediana 115,4) — lo swap costava **~8 %** di tempo su una misura che sembrava valida. Il regime è ora **provato, non dedotto**: il sampler legge `memory.swap.current` della scope *durante* il run (a fine run la scope è già rimossa) e riporta **0 MiB per l'inferenza** in tutti i run del nuovo regime, con `memory.events` a `oom 0 / oom_kill 0` anche col percorso a copia a **40,2 GiB di RssAnon** su 60. Il resto del nodo resta libero di usare zram: è il «swappa tutto tranne l'inferenza» chiesto dal titolare, e non richiede né sudo né `vm.swappiness` globale. **Nota sullo storico**: lo swap di questa macchina è **zram 30,2 GiB compresso in RAM** + swapfile 4 GiB — i «22,4 GiB di swap» del cap 192 (0.3) erano quindi compressione in RAM, non I/O su disco. **Nota metodologica**: a 12 e 16 t il percorso a copia non cambia (128,5 / 109,9 s), perché non swappava comunque (+5 MiB sul baseline zram preesistente); un run a 20 t ha dato **91,9 s** contro i 95,7 canonici e sembrava un guadagno della cgroup — la ripetizione l'ha smentito (95,7 s) e la causa è stata isolata: quel run aveva `expert-read` a **841 ms/fwd** invece di 1087, cioè page cache favorevole sul percorso esperti, 266 ms × 16 fwd ≈ 4,3 s = esattamente il divario. **Un verde isolato non prova nulla**, di nuovo. |
| 08/09 | **0.3 CHIUSA — gradini 96/128/192 su due percorsi, e il verdetto d'ordine** (stesso prompt lungo 274 t. + 16 nuovi = 290 forwards, 12 thread, stessa selezione in tutti i run: 138720 scelte su 12032 esperti distinti). **Copia**: cap 96 → **49,0 %** (22778/23754), expert-read **1329 ms/fwd**, TTFT 123,8 s, 133,7 s per 16 token, RssAnon 33.025 MiB. cap 128 → **61,5 %** (28599/17933), expert-read **1031**, TTFT 118,8 s, **128,2 s** (il minimo della curva), RssAnon 40.226 MiB. cap 192 → **71,4 %** (33225/13307) ma ⚠️ **tempi invalidi**: RssAnon 49.672 MiB su 60 GB, swap di sistema fino a 22,4 GiB, e `expert-read` che *sale* a **2192 ms** mentre i miss si dimezzano — sintomo inequivocabile di thrashing, non di cache. **Mmap** (`COLI_MAP_EXPERTS=1`, che dà il tempo pulito a 192 perché tiene l'anonimo piatto): N128 → 61,5 %, TTFT 125,7 s, **135,4 s**, expert-read **32,5 ms**, routed-expert 3921, RssAnon **12.282** MiB + RssFile 37.772, picco swap 2,3 GiB. N192 → 71,4 %, TTFT 122,7 s, **137,4 s**, expert-read 34,9 ms, routed-expert 4031, RssAnon **13.314** MiB + RssFile 39.864, picco swap 8,6 GiB. **Tre letture.** (1) *Il hit rate è path-independent*: M128/N128 danno 28599/17933 identici, M192/N192 33225/13307 identici — è una grandezza logica (slot × politica), il percorso cambia solo i tempi. (2) *Mmap sposta il costo del fault fra i contatori, non lo elimina*: `expert-read` crolla 1031 → 32 ms ma `routed-expert` sale 2466 → 3921, e il totale è praticamente lo stesso (128,2 vs 135,4 s) — il page fault ora si paga dentro il matmul. Il suo valore vero è **l'anonimo**: 12-13 GiB contro 40-49, cioè la differenza fra girare e swappare. (3) *La curva ha risposto alla domanda per cui esisteva*. A cap 128: `resident-mm` 58,4 s + `deltanet` 39,1 s + `expert-read` 16,5 s su 128,2 → **disco 13 %, denso 76 %**, con il pavimento denso **piatto** (3657/3654/3652 ms/fwd a cap 64/128/192). Il disco non è più il collo di bottiglia: azzerarlo del tutto vale ~13 %, e oltre cap 128 un GiB liberato vale **zero** (N128 135,4 vs N192 137,4 s con +10 punti di hit rate). ⇒ **Fase 2 (VRAM) prima di Fase 1 (int4)**, l'ordine opposto a quello che il piano lasciava aperto: l'int4 comprerebbe RAM e hit rate, cioè il 13 % che resta; solo la GPU attacca il 76 %. Fase 0 **completa**. |
| 08/09 | **0.3 — curva col prompt lungo** (274 token + 16 decode = 290 forwards, stessa selezione dei 3 run: 46532 eventi; il prompt corto lasciava il decode fermo perché il working set ~118/layer non churnava): cap 16 → **7,5 %** (hit 3509 / miss 43023), expert-read **2601 ms/fwd**, decode **10,0 s/tok**, TTFT 148 s. cap 32 → **10,2 %**, expert-read 2307, decode 9,6 s/tok, TTFT 143 s. cap 64 → **26,8 %** (12486/34046), expert-read **1886**, decode **9,2 s/tok**, TTFT 136 s. Il gradiente ora muove il decode: 16→64 = −27,5 % di disco. Asimmetria forte: +2,7 punti da 16→32 ma **+16,6 da 32→64** → la popolarità degli esperti è skewed, gli slot marginali 32-64 sono caldi: 128/192 dovrebbero pagare ancora (da confermare nella finestra K12). **Due fatti architetturali**: (1) `routed-expert` a 2692-2752 ms/fwd, 6,7× più lento che col prompt corto (408) → il matmul esperti non è FLOP-bound ma **bandwidth-bound sulle pagine appena faultate** (4,7 MiB nuovi a esperto, zero locality): VRAM-residenza (Fase 2) e cap alto attaccano la stessa radice; (2) prefetch cala sotto il 100 % sotto carico (78936/86046 → 55416/68092 = 81-92 %) → parte dei miss diventa síncrona. TTFT 136-148 s su prefill 274 a working set freddo (~16-17k esperti unici, 56-80 GiB di disco): questo è il cold-start reale del prompt lungo su questa macchina. PEAK RSS invariato (14,68/18,19/25,22 GB) |
| 08/09 | **0.3 parziale — curva hit-rate, gradini 16/32/64** (stesso prompt 28+16, 43 forwards, percorso copia, 12 thread; working set del run: 5687 esperti distinti = ~118/layer). cap 16 (768 slot): **22,4 %** (hit 2677 / miss 9252), 0,65 tok/s, TTFT 13,3 s, PEAK 14,62 GB. cap 32 (1536): **32,5 %** (3871/8058), 0,61 tok/s, TTFT 15,2 s, PEAK 18,14 GB. cap 64 (3072): **41,3 %** (4928/7001), 0,63 tok/s, TTFT 15,3 s, PEAK 25,17 GB. Gains +10,1 poi +8,8 punti per raddoppio — decrescenti ma vivi; I/O coerente (weight-ranges 18504→16116→14002, prefetched 100 %, batch 1021→828→649). **Lettura**: con 43 forwards il working set per layer (~118 esperti) sta SOTTO cap 192 (192 slot/layer) → a 192 questo prompt arriverebbe a ~70-75 % di hit rate (mancano solo i 5687 primi contatti); su prompt lunghi il working set per layer cresce fino a 512 e la curva va ri-misurata. Decode flat 0,61-0,65 tok/s a questi cap: a 43 forwards il hit rate incide su TTFT/cold-start, non sul decode (disco ~0,5 s su 1,6). 96/128/192: finestra K12 a riposo (comunica il titolare) |
| 08/09 | **0.2 chiuso — scomposizione per token di decode** (cap 32, 12 thread, prompt 28 + 16 nuovi, 16 forwards; stessa riga I/O in tutti i run: weight-ranges 16116, coalesced-gate-up 8058, prefetched 16116 = 100 %, parallel-batches 828, resident-scales 28,12 MiB; hit rate 32,5 % = hit 3871 / miss 8058, 5687 esperti distinti di 25088). Copia, steady state: expert-read **498 ms**, routed-expert 408, shared 40, resident-mm 561 (sovrapposto), deltanet 378, qsa 0,3+4,7, ple 6,2, lm-head 38,2 ms/fwd → **1,64 s/token** (0,61 tok/s), TTFT 15,3 s (prefill 28), RssAnon picco **18,1 GiB** (RssFile ~0). Mappatura sui 4 termini del piano: disco ≈ 498 ms (36 %), expert-matmul ≈ 448 ms (33 %), attention ≈ 382 ms (28 %), lm-head ≈ 38 ms (3 %). **vs recon 96/17/14/2,6 s (disco 69 %)**: stessa fase n.1 a disco freddo, quote molto diverse — 990 PRO + prefetch al 100 % + gate-up coalesced = ~0,5 s/token di disco, ~30× meno del box di riferimento; le fasi CPU (matmul + deltanet + dense) diventano la quota visibile. Conseguenza: su questa macchina la Fase 2 deve attaccare il matmul CPU (routed+resident ≈ 1,38 s/fwd sovrapposto), non solo il disco; il disco resta il problema di TTFT/cold-start (11,5–15,4 s) |
| 08/09 | **0.2b chiuso — copia vs mmap**: pattern di accesso identico bit per bit (riga I/O e hit rate uguali in tutti e 8 i run). Mmap steady state (dal 2° pass): expert-read **26 ms** vs 498 → **1,18 s/token** (0,85 tok/s, 1,38× più veloce), TTFT 11,5 s; primo pass dopo evizione paga re-fault + first-touch (routed-expert 947 ms). RAM: `RssAnon` picco **11,2 GiB vs 18,1** (lo slab mmap non esiste più, −6,9 GiB di anon) + **RssFile fino a 25,9 GiB** di page cache ancorata dai mapping (VmRSS 37,0 GB = somma; file pages recuperabili, ma in pratica sopravvivono alla pressione — a differenza della cache del percorso copia, che qui viene evictata: il run copia «warm» resta lento come il cold, 498 ms). La `PEAK RSS` del binario è `ru_maxrss` (VmRSS, include file pages) — per l'anonimo vero serve `RssAnon` da `/proc/<pid>/status` |
| 08/09 | **0.1 chiuso**: build CPU pulita sul 3900X reale, `qwen38-tiny-check` 8/8 verde (token 8/8 + oracolo numerico) e prefetch-check 16/16 verde; venv torch+transformers 5.16.1 dedicato (passato via `make PYTHON=…`) |
| 08/09 | **verifica referenze**: censimento e `config.json` ripresi dai 131 header a terra — tutto confermato (152.089 tensori, 25.088 esperti, zero F32, 943 voci non convertite, top-k 10, PLE al layer 2); tutte le citazioni di sorgente reggono sulla working tree (branch `qwen38cuda` = `fd93c41` + solo doc, nessun sorgente toccato) |
| 08/09 | **correzioni**: GPU di node-01 = 5060 Ti + 5070 Ti (sm_120, 32 GiB) — la SPEC aveva per errore le 2×1070 di node-02, cascata §3/§5.2/§6 corretta; §3 riordinato (cap 192 = 36,7 % ≈42 GiB, cap 205 = 39 % ≈45 GiB); 0.3b chiuso dal sorgente (allora numerato 0.5), aggiunto 0.2b, statistiche = 8; totale checkpoint precisato (185,56 GB / 172,82 GiB) |
| 07/09 23:11 | **download COMPLETO e verificato per montaggio** — 185.563.800.832 B, 144 file, 131 shard, 0 residui `.incomplete`; tutti i 131 header safetensors si aprono, tutti i **152.089** offset cadono dentro i file |
| 07/09 23:2x | **censimento dei byte per famiglia** (§1.2, §1.3): esperti **25.088 non 24.576** (c'è il layer MTP), scale FP8 **BF16 non F32**, e una **tabella N-gram da 51,27 GB = 27,6 %** che nessuno aveva contato. Punto 0.3a chiuso (allora numerato 0.4), nuovo punto 0.3b (allora 0.5) |
| 07/09 22:4x | download avviato — 185,6 GB, revisione pinnata, riavviabile, `~/models/Qwen3.8-Flash-Next-FP8`, log in `~/models/.dl-qwen38next.log` |
| 07/09 | `config.json` letta e messa a verbale (§1.1) — multimodalità e le 943 voci non-convertite sono novità rispetto alla ricognizione |
| 07/09 | NVFP4 verificato sui sorgenti e **scartato** (§5); int4 `g4` promosso a Fase 1 (§6) |
| — | **Fase 0 sbloccata** (il checkpoint c'è): attende il via |
| 15/09 | **KV: requisito del titolare 2 parallele × 180-200k + bench «KV in RAM vs KV in VRAM» in coda finestra** — aritmetica dal config (48 layer: 36 DeltaNet a stato costante ~113 MB/seq f32, 12 full-attn GQA 2×256, interval 4; indexer budget-capped ~24 MB costante): KV per token 24 KiB f16 / 12 KiB q8 ⇒ **2×200k: 9,8 GiB f16 / 4,9 GiB q8** (il «KV ~1,6» della tabella era a contesto corto: corretto). **La fisica**: la lettura KV per token è O(context) ⇒ a 2×200k ogni token rilegge 4,9 GB: in VRAM ~5 ms/token (rumore), da RAM su PCIe ~200 ms/token ⇒ **a residenza piena il KV-in-RAM diventa il collo, più degli esperti** (col 27B denso era peggiorativa per lo stesso motivo ma 48/48 layer). Capienza decide: tier int4 22,4 + MTP 1,5 + head 1,2 + KV q8 4,9 = **30/32 ci sta al soffitto**; con F16 34,9 NON ci sta (solo con tier ridotto ~6,5k). Bench disegnato a 3 bracci (tutti 2×200k, misura tg): **KV-VRAM q8** / KV-RAM q8 / KV-RAM f16; F32 mai (condiviso); F16-in-VRAM scartato a priori per capienza salvo interesse esplicito per il tier ridotto. Controllo ereditato q8_0-gratis (0,99× sul 27B) si RIMISURA qui, non si eredita. |
| 15/09 | **d3 ESEGUITA: VERDE a doppia verifica — la quantizzazione esperti-only costa poco** — protocollo: 3 bracci teacher-forced (2000 posizioni, slice di `corpus_calib38.txt`, cap 64): FP8×2 per il controllo di determinismo (**bit-identici**, macchina pulita) + int4 con `Q38_TF_REF`. Frame giusto (correzione del titolare): **qui si quantizza SOLO gli esperti** (denso/DeltaNet/PLE/head nativi) — non paragonabile ai quant full-model del 27B; lo scope è lo stesso della gate per-tensore del convertitore (cos 0,9921-0,9942) e la d3 dice la stessa cosa a livello modello. Numeri (int4 vs FP8, costo incrementale): **KL mediana 0,0049**, media 0,1369 (= delta NLL esatto, il cross-check quadra), p90 0,75, max 6,8 (coda = flip di gating occasionale), **top-1 83,4%**, ppl 4,56→5,23. La misurazione è vs FP8 (base ufficiale Qwen, BF16 archiviato): è il costo incrementale, non quello assoluto. **Il bug del gancio beccato dalla coerenza interna**: la prima run stampava media 0,6634 ≠ delta aggregato 0,1369 e top1 1,6% — layout interleaved scritto ma letto flat; fixato, il C ora riproduce il ricalcolo python indipendente **cifra per cifra** (mediana/media/max/top1 esatti). Verdetto: **d3 verde ⇒ finestra GPU proponibile** (tier A/B + oracle + bench KV 3 bracci). |
