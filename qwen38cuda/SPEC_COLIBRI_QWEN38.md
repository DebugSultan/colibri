# Colibri × Qwen3.8-Flash-Next — analisi e specifica

**Documento**: analisi tecnica + piano/specifica di lavoro
**Data**: 2026-09-08
**Autore**: Claude (Opus 5), su mandato del titolare
**Registro di campagna correlato**: `~/bench-llama/PIANO_NEXT.md` (diario, non spec)

> **Che cos'è questo file e che cosa non è.**
> `PIANO_NEXT.md` è un *diario di campagna*: registra in ordine cronologico cosa
> è stato provato e cosa è stato misurato. Questo documento è un'altra cosa —
> è la **specifica**: dice com'è fatta la macchina oggi, che cosa vogliamo
> ottenere, in quale ordine, e **con quale prova si dichiara chiuso ogni passo**.
> Dove i due divergono, questo file è più recente e cita il sorgente.

---

## 0. Sommario esecutivo

Colibri è un motore di inferenza in C che esegue Qwen3.8-Flash-Next
**direttamente dai safetensors FP8 ufficiali**, senza conversione a GGUF e
senza tenere il modello in memoria: gli esperti MoE vengono **letti dal disco
a ogni token** e trattenuti in una cache a slot di capacità configurabile.

Questa è la ragione per cui il modello è eseguibile su una macchina da 64 GB
di RAM quando il checkpoint pesa 172,76 GiB. Ed è anche la ragione per cui
oggi è lento: **il 69 % del tempo per token è disco**.

Il lavoro si divide in tre movimenti, in quest'ordine:

| Fase | Obiettivo | Costo | Rischio |
|---|---|---|---|
| **0** | Misurare la macchina com'è, senza toccare nulla | ore | nullo |
| **1** | Ridurre i byte per token (int4 `g4`) | giorni | medio |
| **2** | Portare gli esperti sulla GPU | giorni | alto |

**Due scoperte fatte leggendo il sorgente riordinano il piano** rispetto alla
ricognizione iniziale, e sono descritte al §5. In breve: (a) esiste già un
percorso **mmap** per gli esperti che potrebbe attaccare il termine di disco
**a costo zero**, e va provato *prima* di scrivere qualunque convertitore;
(b) i kernel CUDA raggruppati per **FP8 nativo esistono già**, quindi la
Fase 2 non è «scrivere kernel», è «collegare qwen38 al backend che c'è».

---

## 1. Il bersaglio

### 1.1 Checkpoint

| voce | valore | come lo so |
|---|---|---|
| percorso | `~/models/Qwen3.8-Flash-Next-FP8` | in locale |
| dimensione | **185.563.800.832 B** = 185,56 GB dec = **172,82 GiB** (i soli dati tensori: 185,50 GB / 172,76 GiB; la differenza è header + file non-shard) | `du` + somma degli header |
| file | 144 (di cui **131 shard** safetensors) | conteggio |
| tensori | **152.089** | parsing degli header |
| formato pesi | **FP8 E4M3 nativo** + scale, misto BF16 sui moduli densi | `config.json` + dtype degli header |
| offset | tutti verificati dentro il file che li dichiara | controllo esplicito |

Il checkpoint **è nativamente FP8**: non è una quantizzazione di qualcun altro.
Questo ha una conseguenza che ritorna al §7 — **non esiste un riferimento BF16**
contro cui misurare la perdita. L'arbitro di ogni quantizzazione futura è
l'FP8 stesso.

### 1.2 Censimento dei byte

| blocco | GB dec | quota |
|---|---:|---:|
| esperti routed (48 layer) | 120,81 | **65,1 %** |
| **tabella N-gram / PLE** | **51,27** | **27,6 %** |
| densi BF16 | 6,81 | 3,7 % |
| embed + lm_head | 2,54 | 1,4 % |
| esperti MTP | 2,52 | 1,4 % |
| torre vision (si salta) | 0,90 | 0,5 % |
| shared expert | 0,48 | 0,3 % |
| densi MTP | 0,17 | 0,1 % |

Due numeri che nessuna documentazione a monte riportava e che sono stati
ricavati contando:

- **gli esperti sono 25.088**, non 24.576: i layer sono **49**, cioè 48 più il
  layer **MTP** (multi-token prediction). Chi conta 48×512 sbaglia di 512
  esperti;
- **la tabella N-gram pesa 51,27 GB, il 27,6 % del checkpoint.** È il secondo
  blocco per dimensione dopo gli esperti, e la ricognizione iniziale non
  l'aveva pesata.

**Un esperto = 4.915.200 B = 4,6875 MiB** (gate 1.638.400 + up 1.638.400 +
down 1.638.400, in FP8).

> ⚠️ **Discrepanza da segnalare.** Un commento nel sorgente
> (`qwen38_core.h:1098`) parla di «14 MB per miss». Gli header dicono
> 4,69 MiB per esperto. I 14 MB corrispondono a 3× quel valore, o a
> un'espansione che nel percorso `Q38_NATIVE_FP8=1` non avviene.
> **La fonte autorevole sono gli header, non il commento**: il numero
> operativo è 4,69 MiB. Va comunque confermato a banco alla misura 0.2,
> perché se per qualche ragione lo slab fosse davvero 14 MB, tutta
> l'aritmetica della residenza cambia di 3×.

### 1.3 Memoria residente

I pesi **densi** restano in memoria per tutta l'esecuzione: **10,00 GB dec =
9,31 GiB**. La banca delle scale degli esperti aggiunge **14,36 MiB**
(sono BF16, non F32 — metà di quanto stimava la ricognizione).

Tutto il resto — esperti e PLE — **non è residente**. È questo il punto
architetturale che rende il modello eseguibile.

---

## 2. Come funziona oggi (verificato dal sorgente)

Sorgenti letti: `c/qwen38.c`, `c/qwen38_core.h`, `c/st.h`, `c/expert_store.h`,
`c/backend_cuda.cu`, `c/qwen36_tier.h`, `c/Makefile`.
Clone locale: `/home/sultano/colibri`, branch `main`, HEAD `fd93c41`.
⚠️ È il **clone upstream JustVugg**, non il fork `DebugSultan/qwen38cuda`.

### 2.1 Il percorso degli esperti

Per ogni token, per ogni layer, il router sceglie `topk` esperti. Per ciascuno:

1. si cerca lo slot in cache (`ColiExpertStore::lookup`);
2. se manca, `q38_try_load_native_fp8_expert` porta i tre tensori dal disco;
3. si esegue il MLP;
4. si rilascia il lease.

Il caricamento ha **due percorsi**, ed è qui la prima scoperta (§5.1):

```c
/* Se i tre intervalli sono mappati (COLI_MAP_EXPERTS=1), lo slot li PUNTA
 * invece di copiarli: niente slab, niente 14 MB per miss. */
const uint8_t *pg = st_map_shard_range(weight[0]->fd, weight[0]->off, weight[0]->nbytes);
...
if (pg && pu && pd) {
    q38_bind_borrowed_fp8(&slot->gate, (void*)pg, scales, c->inter, c->hidden);
    ...
    if (slot->fp8_slab) { free(slot->fp8_slab); ... }
    return;
}
/* fallback: slab + due pread (gate+up contigui, poi down) */
```
— `qwen38_core.h:1092-1132`

Il fallback è già ottimizzato: gate e up sono **contigui su disco** e si
leggono con **una sola** `pread` (`pair_bytes`), poi una seconda per down.
Due syscall per esperto, non tre.

### 2.2 La cache a slot e il contratto di lease

`expert_store.h` definisce un contratto stretto, che qualsiasi modifica deve
rispettare:

- dopo un `lookup()` riuscito il chiamante **deve** chiamare `release()`
  esattamente una volta sulla stessa view;
- le view **non si copiano** (il lease non è condivisibile);
- in caso di fallimento la view è azzerata e **non va rilasciata**;
- `destroy()` pretende **zero lease attivi** (asserito nelle build di debug);
- la thread-safety è specifica dell'implementazione.

Le statistiche disponibili sono **otto**, non cinque come indicato nel
registro:

```c
uint64_t requests, hits, misses, prefetched, prefetch_hits,
         bytes_read, resident_bytes, capacity_bytes;
```

`prefetched` / `prefetch_hits` / `capacity_bytes` erano sfuggite. Sono
esattamente le tre che servono a giudicare se il prefetch paga.

### 2.3 Il guardiano della RAM

`colibri.c:817-839` documenta una trappola già disinnescata a monte
(issue #1325) e che vale la pena conoscere prima di accendere la mappatura:

> `rss_guard` **sfratta esperti** quando la misura supera il budget. Con la
> mappatura accesa `VmRSS` sale di gigabyte **senza che un byte in più sia
> sottratto al sistema** — e il guardiano si metterebbe a sfrattare per
> liberare memoria che non stava occupando. «Non un avviso cosmetico: cache
> distrutta e lavoro rifatto.»

La soluzione già in albero: misurare **`RssAnon`**, non `VmRSS`. Misurato su
GLM-5.3 (391 GB di container, 25 GB di RAM) il pianificatore riportava
18,2 GB e segnalava uno sforamento di 0,8 GB **che non esisteva**.

**Corollario per noi**: con `COLI_MAP_EXPERTS=1` qualunque monitoraggio di
memoria che guardi `VmRSS` mente. Questo vale anche per gli occhi umani su
`htop`.

### 2.4 La tabella N-gram (PLE) — **punto 0.5 chiuso**

Era la domanda aperta più grossa: 51,27 GB, il 27,6 % del checkpoint,
devono stare in memoria? **No.** Chiuso leggendo il sorgente, non a banco.

```c
static void q38_ple_row(Model *m, int64_t row, float *out) {
    ...
    if (t->dtype == 4) {
        uint8_t raw[512];
        st_read_slice_raw_cap(&m->S, nm, local * c->ngram_head_dim,
                              c->ngram_head_dim, raw, sizeof raw, 1);
        for (int d = 0; d < c->ngram_head_dim; d++)
            out[d] = e4m3_decode(raw[d]) * m->ple_weight_scale;
    } else st_read_slice_f32(...);
}
```
— `qwen38_core.h:1344`

**Una riga per volta, direttamente dallo shard.** `ngram_head_dim` = 160 byte.
Con 16 teste sono **2.560 byte per token**. Niente è residente, e c'è **una
sola** `ple_weight_scale` globale.

Esiste inoltre `q38_ple_prefetch` (env `Q38_PLE_PREFETCH`, **default 1**), e la
sua motivazione è essa stessa materiale da spec:

> Gli indici di riga PLE sono una **funzione pura degli id dei token** e dei
> due token precedenti — `q38_hash_row` non legge stato nascosto — quindi
> sono noti **prima** di qualsiasi calcolo. Gli esperti no: dipendono dal
> router del layer precedente, e il loro lookahead è di un layer solo.
> Il PLE cade sul layer 2 di 48: due layer interi di streaming esperti
> coprono le letture da 16×160 B. Il prefetch trasforma 16 letture seriali a
> profondità di coda 1 in letture parallele.
> **«Riordino puro: stessi byte, letti prima.»**

La funzione di hash, per completezza:
`x = cur*mult[0] ^ p1*mult[1] ^ (ngram==3 ? p2*mult[2] : 0)`, poi
`r = x % ple_head_vocab[head]` (reso positivo), infine
`ple_head_offset[head] + r`.

**Conseguenza sul piano: nessuna.** I 51,27 GB non chiedono residenza. La
voce 0.5 si chiude senza modifiche al progetto.

### 2.5 Come si costruisce

```make
qwen38$(EXE): qwen38.c cli_args.h qwen38_core.h ...
	$(CC) $(NOCUDA_CFLAGS) qwen38.c -o qwen38$(EXE) $(NOCUDA_LDFLAGS)
```
— `c/Makefile:1035-1036`

Il target qwen38 è compilato **deliberatamente senza CUDA**. E la verifica
diretta lo conferma: `qwen38.c` e `qwen38_core.h` **non contengono una sola
occorrenza di `COLI_CUDA`**. Il percorso qwen38 è interamente CPU, a differenza
di qwen36 che ha il suo tier GPU.

**Questa è la risposta alla domanda «perché la GPU non funziona», che il
titolare aveva osservato non essere spiegata da nessuna parte: non è un bug e
non è una regressione. Il gancio GPU per qwen38 non è mai stato scritto.**

### 2.6 Il banco giocattolo

Il Makefile offre qualcosa che cambia la pianificazione della Fase 0:

```make
qwen38-tiny-generate:      # tools/make_qwen38_tiny.py --out ./qwen38_tiny
qwen38-tiny-check:         # esegue contro ref.json, matrice cap × batch × bf16
qwen38-ple-prefetch-check: # confronta prefetch on/off, stesso ref.json
```

Esiste cioè un **checkpoint sintetico e un riferimento deterministico** che
provano la correttezza del percorso qwen38 **senza toccare i 173 GB**.
Ogni modifica al loader o al formato dei pesi va provata prima qui.

---

## 3. La macchina

| | node-01 |
|---|---|
| CPU | Ryzen 9 3900X (12c/24t) |
| RAM | 64 GB (≈45 GiB utilizzabili col K12 in produzione) |
| GPU | RTX 5060 Ti 16 GB + RTX 5070 Ti 16 GB, **Blackwell sm_120**, 32 GiB totali |
| disco | NVMe Samsung 990 PRO 1 TB, 312 GB liberi |

Vincoli hardware che non si negoziano:

- le due schede **non hanno NVLink** e stanno su PCIe stretto: la
  collocazione degli esperti tra le due VRAM è una decisione di progetto;
- sm_120 richiede **CUDA ≥ 12.8** (in casa è pinnata la 13.0);
- l'FP8 su sm_120 è **aritmetica nativa** (tensor core), non storage:
  il guadagno del tier VRAM è di banda, residenza *e* FLOPS.

Le 2× GTX 1070 (8 GB, Pascal sm_61) sono su **node-02** e non sono nel
percorso di questa campagna: la prima stesura di questa sezione le
attribuiva a node-01 ed era errata (corretto il 08/09).

**Vincolo operativo assoluto: il cervello di produzione K12 (27B) sul
node-01 non si ferma senza un via esplicito del titolare.** Ogni misura di
Fase 0 e 1 è progettata per convivere con lui.

---

## 4. Economia del token (baseline dichiarata)

Per un token, mediana **140 s**:

| voce | s | quota |
|---|---:|---:|
| **disco (servizio esperti sincrono)** | **96** | **69 %** |
| matmul esperti | 17 | 12 % |
| attenzione | 14 | 10 % |
| LM head | 2,6 | 2 % |

Traffico esperti per token: **2,198 GiB**.

> ⚠️ **Questi numeri vengono dalla ricognizione a monte, non da una misura
> nostra**, e in particolare **non sappiamo se furono presi con
> `COLI_MAP_EXPERTS=1` o sul percorso a copia**. È esattamente per questo che
> la Fase 0 esiste, e per questo la misura 0.2 viene prima di tutto il resto.
> *Un verde isolato non prova nulla; un numero ereditato ancora meno.*

**Il bersaglio è il 69 %.** Tutto il resto sommato non arriva a un quarto del
tempo. Qualunque lavoro che non tocchi i byte letti dal disco, o il modo in
cui vengono letti, sta ottimizzando il rumore.

---

## 5. Le due scoperte che riordinano il piano

### 5.1 `COLI_MAP_EXPERTS=1` — il termine di disco potrebbe già essere attaccabile a costo zero

Descritto al §2.1. La mappatura è **opt-in**, spenta di default
(`st.h:1007`), con una cache di mapping **per file** tenuta viva per tutta la
durata del processo, e con memoria del fallimento per fd (non si ritenta a
ogni chiamata). Un `NULL` di ritorno significa «usa pread», **non** errore.

Perché è importante: nel percorso mappato lo slot **punta** ai byte invece di
copiarli. Il costo di un miss non è più «leggi 4,69 MiB in un buffer», è
«tocca pagine che potrebbero già essere nel page cache del kernel». Con
~45 GiB di RAM libera, il page cache può ospitare una frazione consistente
dei 120,81 GB di esperti — **e lo fa in modo ortogonale alla cache a slot**,
cioè in aggiunta a `cap`, non al suo posto.

**Conseguenza sul piano: la misura di `COLI_MAP_EXPERTS` diventa un elemento
di Fase 0, prima di qualunque riga di convertitore int4.** Se sposta il
termine da 96 s, sposta il bersaglio dell'intero progetto, gratis.

### 5.2 I kernel CUDA FP8 raggruppati esistono già

`backend_cuda.cu` ammette sul percorso esperti raggruppati quattro formati:

| `fmt` | formato |
|---:|---|
| 2 | s4 |
| **4** | **g4 — int4 raggruppato** (il bersaglio di Fase 1) |
| 6 | e8 |
| **8** | **FP8** |

e i kernel per FP8 sono in albero e nominati:
`grouped_hidden_f8_dual` (`:892`), `grouped_down_f8` (`:911`),
più le varianti pesate `grouped_hidden_f8w_dual` (`:933`) e
`grouped_down_f8w` (`:968`), selezionate a `:1918` e `:2113` sui rami
`all_f8`. Gruppi misti E8/FP8 ricadono sul percorso per-esperto.

**Conseguenza sul piano: la Fase 2 non è «scrivere kernel FP8».
È collegare il percorso qwen38 — oggi NOCUDA per costruzione — al backend
CUDA che esiste già.** È un lavoro di plumbing e di gestione della memoria,
non di aritmetica. Su sm_120 l'FP8 è aritmetica nativa: il guadagno atteso
è di **banda, residenza e FLOPS**. La recon documenta in albero anche il
percorso DeepGEMM `sm_120a` (opt-in `DEEPGEMM=1`, blocchi FP8) da valutare
per i GEMM degli esperti.

### 5.3 Il vincolo del tier che va invertito

```c
/* cap_experts_per_layer must equal n_experts (full RAM residency): the tier
 * stores raw pointers into the expert slots, which must never be evicted. */
```
— `qwen36_tier.h:27-28`

Il tier GPU di qwen36 **assume la residenza totale** in RAM e tiene puntatori
grezzi negli slot. Per qwen38 questa assunzione è **falsa per costruzione**:
gli slot sono una cache che sfratta.

**Un `qwen38_tier` non può essere un copia-incolla di `qwen36_tier`.** Deve
possedere una propria copia in VRAM, con un proprio ciclo di vita, e non
puntare mai dentro uno slot RAM sfrattabile. Questo è il rischio tecnico
principale della Fase 2 ed è noto **prima** di cominciare.

Nota utile dallo stesso header: il chiamante determina int4 vs int8 **dalla
taglia su disco**, non da `meta.ebits`, «che su qualche container mente».

---

## 6. Il piano

### Fase 0 — misurare, senza toccare niente

Tutto a CUDA spenta, produzione intatta. Nessun elemento di questa fase
richiede di fermare il K12.

| # | cosa | prova di chiusura | stato |
|---|---|---|---|
| 0.1 | build del target CPU `qwen38` | il binario esiste e `qwen38-tiny-check` passa su tutta la matrice | ✅ chiuso 08/09: 8/8 configurazioni, token 8/8 + oracolo numerico, prefetch 16/16 |
| 0.2 | baseline con `COLI_TIMERS=1` sul 3900X vero | scomposizione del tempo per token misurata **da noi**, confrontata col §4 | ✅ chiuso 08/09: disco 36 % / expert 33 % / attn 28 % / lm-head 3 %, 1,64 s/token (copia) |
| **0.2b** | **`COLI_MAP_EXPERTS=1` contro il percorso a copia** | **stessa scomposizione, due configurazioni, stesso prompt; ⚠️ leggere `RssAnon`, non `VmRSS` (§2.3)** | ✅ chiuso 08/09: 1,18 s/token, `RssAnon` 11,2 vs 18,1 GiB, +25,9 GiB file-backed |
| 0.3 | **curva hit-rate vs `cap`** a 16·32·64·96·128·192 | curva completa con le otto statistiche di `ColiExpertStoreStats` | da fare |
| 0.4 | censimento byte e offset | ✅ chiuso (§1.2) | fatto |
| 0.5 | residenza della tabella N-gram | ✅ **chiuso dal sorgente** (§2.4): streaming per riga, mai residente | fatto |

**La curva 0.3 è il giudice.** È lei a dire se int4 serve davvero: se a
`cap` realistici l'hit-rate è già alto, dimezzare i byte per esperto sposta
poco; se è basso, raddoppiare la residenza a parità di RAM è la leva
principale. La decisione di entrare in Fase 1 **si prende dopo aver visto
0.2b e 0.3, non prima**.

### Fase 1 — int4 `g4` (condizionata all'esito di 0.3)

**Bersaglio**: dimezzare i byte per esperto.

| voce | FP8 oggi | int4 `g4` gs=128 | fattore |
|---|---:|---:|---:|
| per esperto | 4,6875 MiB | **2,4902 MiB** | 1,88× |
| esperti su disco | 120,81 GB | **65,5 GB** | |
| traffico per token | 2,198 GiB | **1,167 GiB** | |
| residenza in 45 GiB RAM | 39 % | **74 %** | |
| residenza in 32 GiB VRAM | 28 % | **52 %** | |

Serve spazio per **65,5 GB** sui 312 GB liberi: capiente.

Lavoro:

1. **convertitore** FP8 (blocchi 128×128) → int4 `g4` con `gs=128`,
   **solo per gli esperti routed**. I 943 moduli in
   `modules_to_not_convert` restano BF16 — e il titolare ha già posto la
   regola: *quello che gira su CPU è meglio lasciarlo F32 o BF16*;
2. **loader** con lo stesso schema a slab contiguo di oggi, così che gate+up
   restino una sola lettura;
3. decidere se le scale `g4` possono essere BF16 (2,4902 → **2,4170 MiB**,
   1,94×) **leggendo la definizione di `fmt=4`**, non assumendola;
4. provare prima sul banco giocattolo (§2.6), poi sul checkpoint vero.

**NVFP4 è stato scartato e non torna**: non esiste in colibri, e sul 27B era
già stato bocciato tre volte. MXFP4 (`fmt=7`) esiste solo sul percorso denso,
non su quello degli esperti.

### Fase 2 — tier VRAM (richiede una finestra concordata sul K12)

Prerequisito: **via esplicito del titolare**, perché le 1070 e la RAM sono
condivise con la produzione.

1. `qwen38_tier.c/h` modellato su `qwen36_tier`, **con il vincolo invertito**
   (§5.3): il tier possiede la propria copia in VRAM e non punta mai dentro
   uno slot sfrattabile;
2. collegare il percorso qwen38 ai kernel raggruppati esistenti — `fmt=8`
   subito, `fmt=4` se la Fase 1 è passata;
3. 32 GiB di VRAM su due schede senza NVLink, **condivisa col K12 di
   produzione** (picco misurato 30.962 MiB): la politica di collocazione
   degli esperti fra le due schede è essa stessa una decisione di progetto,
   non un dettaglio.

### Fase 3 — non pianificata

Non si scrive finché la Fase 1 non ha un numero.

---

## 7. Criteri di accettazione

Regole vincolanti, ereditate da campagne che le hanno pagate care.

1. **KL + top-1, mai PPL.** Sul 27B la perplessità avrebbe promosso il
   candidato peggiore **tre volte su tre**. Non è un'opinione sulla metrica:
   è un risultato misurato in casa.
2. **L'arbitro è l'FP8 nativo.** Non esiste un riferimento BF16 per questo
   modello: il checkpoint *è* FP8. Ogni delta si misura contro l'FP8, sugli
   stessi prompt, con lo stesso seme.
3. **Un verde isolato non prova nulla.** Ogni misura vuole una controprova:
   il banco giocattolo *e* il modello vero, il prefetch acceso *e* spento,
   la mappatura accesa *e* spenta.
4. **Caricare non è reggere.** Il tetto si misura col **picco sotto stress**,
   mai col cuscino a riposo. Il payload di prova va contato prima.
5. **`RssAnon`, non `VmRSS`**, in ogni misura di memoria fatta con la
   mappatura accesa (§2.3). Vale per gli script e per gli occhi.
6. **Il K12 non si ferma senza un via esplicito.**
7. Nessun numero ereditato entra in una decisione senza essere stato
   rimisurato qui. I numeri del §4 sono, oggi, ereditati.

---

## 8. Rischi noti

| rischio | dove | mitigazione |
|---|---|---|
| il tier assume residenza totale | §5.3 | inversione del vincolo progettata prima di scrivere |
| `rss_guard` sfratta per memoria mappata | §2.3 | leggere `RssAnon`; già corretto a monte (#1325) |
| `meta.ebits` mente su alcuni container | `qwen36_tier.h` | dedurre int4/int8 dalla taglia su disco |
| «14 MB per esperto» nel commento vs 4,69 MiB negli header | §1.2 | confermare a banco in 0.2 prima di fidarsi dell'aritmetica |
| il baseline 140 s potrebbe essere stato preso col percorso a copia | §4 | 0.2 e 0.2b lo rimisurano entrambe le vie |
| clone locale ≠ fork con la ricognizione | §2 | il lavoro nasce sull'upstream; ogni divergenza col fork va riconciliata prima di scrivere codice |
| contratto di lease violato → assert in debug | §2.2 | ogni `lookup` riuscito ha esattamente un `release` |

---

## 9. Stato

- **Fase 0**: 0.1, 0.2, **0.2b**, 0.4 e 0.5 chiuse. 0.3 aperta.
- **Fase 1**: non iniziata, condizionata a 0.3.
- **Fase 2**: non iniziata, richiede un via sul K12.
- **Produzione**: intatta. Nulla di questo lavoro ha toccato il K12 né il 9B.
