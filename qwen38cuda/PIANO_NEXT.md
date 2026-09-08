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
compra. Con 64 GB e ~45 GiB utili dopo denso, workspace e sistema, si arriva a
**cap ≈ 192** — cioè il **39 % di residenza** (9.829 esperti su 25.088), contro
il 6 % (cap 32) su cui upstream misura già **53,8 % di hit**.

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

- **0.1** build CPU del target `qwen38` sul 3900X reale.
- **0.2** baseline con `COLI_TIMERS=1`: confermare (o smentire) la
  scomposizione 96/17/14/2,6 su *questa* macchina, non su quella della
  ricognizione.
- **0.3** **curva hit-rate contro `cap`**: 16 · 32 · 64 · 96 · 128 · **192**.
  Spinta fino a ~42 GiB, non fermata a 96 — la nostra RAM regge il 40 % di
  residenza contro il 6 % su cui upstream ha misurato. Da
  `ColiExpertStoreStats` (`requests`, `hits`, `misses`, `bytes_read`,
  `resident_bytes`) — la struttura c'è già, non va scritta.
- **0.4** ~~chiudere il «14 MB per miss»~~ **CHIUSO senza banco**: l'header dice
  4.915.800 B per esperto (§1.2). I 14 MB della ricognizione sono
  **~3 esperti**, non uno: o un miss contava l'intero gruppo top-k parziale, o
  la cifra era per-layer e non per-esperto. Il numero da usare è **4,6881 MiB**.
- **0.5** verificare come colibri gestisce la **tabella N-gram da 51,3 GB**
  (§1.3): mmap, streaming o caricamento? È un quarto del checkpoint e nessuno
  dei due documenti la contava. Se pretende residenza, il piano cambia.

**La curva 0.3 è il giudice.** Dice quanto vale un GiB liberato — cioè quanto
vale l'int4 — e quanto vale un GiB di VRAM. Prima della curva, ogni scelta
d'ordine è un'opinione.

### Fase 1 — int4 (condizionata alla curva)

1. Convertitore esperti FP8 128×128 → int4 `g4` gs=128. Solo i tensori
   routed: le 943 voci di `modules_to_not_convert` restano BF16, intatte.
2. Loader dell'artefatto int4 con lo stesso schema a slab contiguo.
3. **Misura: KL + top-1 contro l'FP8 nativo. Mai la PPL.** Il precedente
   NVFP4 è categorico — la PPL avrebbe promosso il peggior candidato tre volte
   su tre.
4. Se la coda regge: ri-misura della curva 0.3 sul nuovo artefatto.

### Fase 2 — tier VRAM (richiede una finestra sul K12)

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

---

## 9. Stato

| | |
|---|---|
| 07/09 23:11 | **download COMPLETO e verificato per montaggio** — 185.563.800.832 B, 144 file, 131 shard, 0 residui `.incomplete`; tutti i 131 header safetensors si aprono, tutti i **152.089** offset cadono dentro i file |
| 07/09 23:2x | **censimento dei byte per famiglia** (§1.2, §1.3): esperti **25.088 non 24.576** (c'è il layer MTP), scale FP8 **BF16 non F32**, e una **tabella N-gram da 51,27 GB = 27,6 %** che nessuno aveva contato. Punto 0.4 chiuso, nuovo punto 0.5 |
| 07/09 22:4x | download avviato — 185,6 GB, revisione pinnata, riavviabile, `~/models/Qwen3.8-Flash-Next-FP8`, log in `~/models/.dl-qwen38next.log` |
| 07/09 | `config.json` letta e messa a verbale (§1.1) — multimodalità e le 943 voci non-convertite sono novità rispetto alla ricognizione |
| 07/09 | NVFP4 verificato sui sorgenti e **scartato** (§5); int4 `g4` promosso a Fase 1 (§6) |
| — | **Fase 0 sbloccata** (il checkpoint c'è): attende il via |
