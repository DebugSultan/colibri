# Stato della PR fmt=9 — preparata il 2026-09-19, gate chiuso in finestra

## Cosa c'è qui
- `fmt9-vs-upstream-dev-5a0b725.patch` — patch completa, 4 file, 369 righe, in
  formato `git` (header `diff --git`, index reali, `new file mode` per il test),
  contro `upstream/dev` @ `5a0b725` (#1582, merge più recente al momento).
- `PR_BODY.md` — descrizione pronta.

## Verifiche fatte
- `git apply --check` pulito su un albero `git archive 5a0b725` vergine; applicato
  davvero, i 4 file risultanti sono **identici byte per byte** all'albero su cui è
  stato compilato ed eseguito il test.
- `nvcc -O2 -std=c++14 -c backend_cuda.cu` → 0 errori, 0 warning.
- test compilato con i **flag esatti della CI upstream**
  (`-O3 -std=c++17 -ftz=false -arch=native -Xcompiler=-Wall,-Wextra`) → 0 warning.
- `make -n CUDA=1 cuda-test` mostra la ricetta bf16 agganciata correttamente.
- **`./bf16_cuda_test` eseguito a prod ferma → 0 failure, RC=0**, tutte le claim
  stampate verdi (decode su 65536 pattern, parity vs riferimento `double`,
  scale-free upload + refresh, controllo negativo su `fmt=1`).

## Il difetto trovato dal test (ed è l'argomento della PR)
Alla prima esecuzione la claim 3 è **fallita**:
`[CUDA] scale refresh: invalid argument` → `FAIL refresh`.
Causa: `coli_cuda_tensor_update()` scriveva la regola scale-free come
`!fmt || tensor->fmt==6`, cioè la stessa regola **nella grafia opposta** rispetto
ai `fmt && fmt != 6` degli altri sette siti. La sweep su una grafia non poteva
trovare l'altra, e `fmt=9` cadeva dritto in una `cudaMemcpy` da puntatore host
NULL. Risolto instradando anche quel sito su `fmt_scale_free()`; ricompilato coi
flag CI (0 warning) e test riverde.
Il nostro albero (`c/backend_cuda.cu:1806`) era già corretto: il difetto esisteva
solo nella trasposizione su upstream, nessun impatto su motore e batteria.

## Correzione a una previsione precedente
Avevo annunciato "due bugfix NULL-deref upstream" come punto forte. Verificato:
**quei due non sono raggiungibili** — `absorb_scale` è bloccato a monte da
`absorb_fmt_ok()` → `coli_cuda_weight_at_supported()` = {0,1,2,3,4}, e l'epilogo
del matmul escludeva già `fmt != 6`. Restano hardening. Il bug vero è il terzo,
quello sopra, e non l'avevo previsto: l'ha trovato il test.

## Decisione che resta al titolare
La regola di campagna vieta branch laterali in questo repo. Una PR upstream
richiede un branch basato su `upstream/dev`. Patch e descrizione sono pronte come
file: il gesto di creare il branch e aprire la PR è tuo.
