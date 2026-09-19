# Stato della PR fmt=9 — preparata fuori-finestra il 2026-09-19

## Cosa c'è qui
- `fmt9-vs-upstream-dev-5a0b725.patch` — patch completa, 4 file, 347 righe,
  contro `upstream/dev` @ `5a0b725` (#1582, merge più recente al momento).
- `PR_BODY.md` — descrizione pronta.

## Verifiche già fatte (fuori-finestra, senza GPU)
- `git apply --check` pulito su un albero `git archive upstream/dev` vergine.
- `nvcc -O2 -std=c++14 -c backend_cuda.cu` → 0 errori, 0 warning.
- test compilato con i **flag esatti della CI upstream**
  (`-O3 -std=c++17 -ftz=false -arch=native -Xcompiler=-Wall,-Wextra`) → 0 warning.
- `make -n CUDA=1 cuda-test` mostra la ricetta bf16 agganciata correttamente.

## Gate ancora aperto — DA FARE NELLA FINESTRA
`./bf16_cuda_test` **non è ancora stato eseguito su questo albero upstream**:
la prod occupa le due schede (≈600 MiB liberi) e l'init CUDA va in OOM.
La PR non si apre finché le 4 claim non passano verdi.
Comando, a prod ferma:

    cd <albero upstream con patch applicata>/c
    make CUDA=1 cuda-test      # oppure solo: ./bf16_cuda_test

## Correzione a una previsione precedente
Avevo annunciato "due bugfix NULL-deref upstream" come punto forte della PR.
Verificato sul codice: **non sono raggiungibili**.
- `absorb_scale` con un formato scale-free è bloccato a monte da
  `absorb_fmt_ok()` → `coli_cuda_weight_at_supported()` = {0,1,2,3,4}.
- l'epilogo del matmul escludeva già `fmt != 6` esplicitamente.
La descrizione li presenta quindi come *hardening*, non come fix.
L'argomento forte della PR è un altro, ed è vero: bf16 è già un dtype di prima
classe sul lato CPU di upstream (`st.h:122`), il backend CUDA non lo sa decodificare.

## Decisione che resta al titolare
La regola di campagna vieta branch laterali in questo repo. Una PR upstream
richiede un branch basato su `upstream/dev`. Patch e descrizione sono pronte
come file: il gesto di creare il branch e aprire la PR è tuo.
