# Handoff — Fase 0.4: ri-misurare a 20 thread

> **CHIUSA il 13/09.** Esito: l'SMT paga — 20 thread portano i 16 token da
> 128,5 a **95,7 s (−25,5 %)** sul percorso a copia, con `resident-mm` −37,1 %
> e `deltanet` −31,9 %; `expert-read` piatto. Il gradino a 24 è stato escluso
> dal titolare. Emersa e risolta una seconda cosa: **l'inferenza va misurata in
> una cgroup senza swap** (0.4b). Verbale completo in `PIANO_NEXT.md` §9.

> Documento di consegna per una nuova sessione. Registro completo: `PIANO_NEXT.md`
> (stesso cartella) — §9 «Stato» è la tabella cronologica delle misure.

## Il compito
Rifare la misura di riferimento **a 20 thread**. Tutta la Fase 0 è girata a **12**,
che sono i core *fisici* del 3900X (12c/**24t**): scelta conservativa da carico
compute-bound, mai messa in discussione. Ma la Fase 0 ha poi misurato che il tempo
è denso-dominato e **bandwidth-bound sulle pagine appena faultate** — ed è proprio
il regime in cui l'SMT può pagare. Quindi va misurato, non assunto. 20 lascia
4 thread al sistema (K12 di produzione compreso).

## Stato attuale — Fase 0 CHIUSA (08/09)
Bersaglio: **Qwen3.8-Flash-Next** MoE su colibri (fork JustVugg), branch `qwen38cuda`,
checkpoint 185 GB verificato. Macchina: node-01, Ryzen 9 3900X, 64 GB DDR4-3200
(**non rialzare**), NVMe 990 PRO, RTX 5070 Ti + 5060 Ti (sm_120).

Il risultato che governa tutto, a cap 128 / 290 forwards / 12 thread:

| voce | tempo | quota |
|---|---|---|
| `resident-mm` (denso CPU) | 58,4 s | **76 % col deltanet** |
| `deltanet` | 39,1 s | |
| `expert-read` (disco) | 16,5 s | **13 %** |
| **totale 16 token** | **128,2 s** | |

Tre fatti da non ri-derivare:
1. **Il pavimento denso è piatto** — 3657/3654/3652 ms/fwd a cap 64/128/192.
   È lui il bersaglio, ed è CPU: ecco perché i thread contano.
2. **Il hit rate è path-independent** (copia e mmap danno gli stessi hit/miss):
   è una grandezza logica. I *tempi* di un run che thrasha si buttano, il hit rate no.
3. **Oltre cap 128 un GiB liberato vale zero.** ⇒ verdetto d'ordine:
   **Fase 2 (VRAM) prima di Fase 1 (int4)**, l'opposto di quanto il piano lasciava aperto.

## Come misurare
Stesso protocollo dei run 0.3, cambiando **solo** i thread — altrimenti non è un confronto:
prompt lungo **274 token + 16 nuovi = 290 forwards**, `COLI_TIMERS=1`, `COLI_CUDA=0`,
percorso a copia **e** `COLI_MAP_EXPERTS=1`, cap **128** (il minimo della curva).
Gradini utili: **12 (baseline già in tabella) → 16 → 20 → 24**.

Cosa guardare: `resident-mm` e `deltanet` — se l'SMT paga, paga lì. Occhio a `RssAnon`
e allo swap: a 12 thread cap 192 arrivò a 49 GiB anonimi e **22,4 GiB di swap**, e i
tempi diventarono spazzatura. Più thread = più workspace.

## Regole della campagna (non negoziabili)
- **Un verde isolato non prova nulla**: ogni misura ripetuta, il payload contato.
- **Caricare non è reggere**: il tetto si misura col picco sotto stress.
- **KL + top-1, mai la PPL** (avrebbe promosso i peggiori 3 volte su 3).
- **Il K12 di produzione non si ferma** senza via esplicito del titolare.
- Ciò che gira su CPU resta **F32 o BF16**.
- Si lavora **solo in `/opt/zyonix/backend`**, mai in `/home/sultano`.
- **Fase 1 e Fase 2 non partono senza il via esplicito** del titolare.

## Dove vivono le memorie
- Registro vivo, misure e razionale: **`/opt/zyonix/backend/colibri/qwen38cuda/PIANO_NEXT.md`**
  (+ `SPEC_COLIBRI_QWEN38.md`, `QWEN38-GPU-RECON.md`). Branch `qwen38cuda`, ultimo commit `21477d8`.
- Memoria persistente in `/home/sultano/.claude/projects/-opt-zyonix/memory/`, indice `MEMORY.md`.
  Le voci che servono qui: `project_qwen38_next_fattibilita` (questa campagna),
  `reference_caricare_non_e_reggere`, `project_nvfp4_layerpin_plan` (perché la PPL è bandita),
  `project_hardware` (RAM a 3200, non toccare), `project_k4_kquant_prod` e
  `project_qwen38_campagna` (il K12 in produzione), `project_llama_cpp_cuda`.

## Direzione generale
Il north-star resta l'autonomia. Su questo fronte: portare in casa un MoE grosso che
oggi non ci sta, **senza toccare la produzione**. La Fase 0 ha detto dove attaccare —
la GPU, non il disco. Prima però si chiude questa domanda aperta sui thread: è la leva
più economica che esista (una variabile d'ambiente) su una quota del 76 %.
