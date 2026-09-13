#!/usr/bin/env bash
# Fase 0.4 — ri-misura a N thread. Protocollo identico ai run 0.3 (cap 128, prompt lungo
# 274 t. + 16 nuovi = 290 forwards, COLI_TIMERS=1, COLI_CUDA=0). Unica variabile: i thread.
# NOSWAP=1 -> il motore gira in una scope cgroup v2 con memory.swap.max=0: l'inferenza
# non swappa mai, il resto del nodo resta libero di usare zram.
set -u
TAG="$1"; THREADS="$2"; PATHMODE="$3"; CAP="${4:-128}"; NOSWAP="${NOSWAP:-0}"
SCRATCH=/opt/zyonix/backend/colibri-q38-scratch
ENG=/opt/zyonix/backend/colibri/c/qwen38
LOG="$SCRATCH/log-$TAG.txt"

export SNAP=/home/sultano/models/Qwen3.8-Flash-Next-FP8 N_NEW=16 COLI_TIMERS=1 COLI_CUDA=0
export OMP_NUM_THREADS="$THREADS"
export COLI_USAGE="$SCRATCH/usage-$TAG"
[ "$PATHMODE" = "mmap" ] && export COLI_MAP_EXPERTS=1 || unset COLI_MAP_EXPERTS

SW0=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
{
  echo "### run $TAG | threads=$THREADS | path=$PATHMODE | cap=$CAP | noswap=$NOSWAP | $(date -Is)"
  echo "### page-cache prima: $(awk '/^Cached:/{print $2/1048576" GiB"}' /proc/meminfo) | swap gia' in uso: $SW0 MiB"
} > "$LOG"

if [ "$NOSWAP" = "1" ]; then
  systemd-run --user --scope -p MemorySwapMax=0 --quiet -- \
    "$ENG" "$CAP" 8 "$SCRATCH/prompt-lungo.txt" >> "$LOG" 2>&1 &
else
  "$ENG" "$CAP" 8 "$SCRATCH/prompt-lungo.txt" >> "$LOG" 2>&1 &
fi
WAITPID=$!

# il PID del motore: con la scope non coincide col job di shell
PID=""; for _ in $(seq 40); do PID=$(pgrep -n -u "$(id -u)" -x qwen38 || true); [ -n "$PID" ] && break; sleep 0.25; done
[ -z "$PID" ] && PID=$WAITPID
CG=/sys/fs/cgroup$(awk -F: '/^0:/{print $3}' /proc/$PID/cgroup 2>/dev/null)
echo "### cgroup: $CG" >> "$LOG"

MAXA=0; MAXF=0; MAXV=0; MAXT=0; MAXSW=0; MAXCG=0
while kill -0 $PID 2>/dev/null; do
  if [ -r /proc/$PID/status ]; then
    eval "$(awk '/^RssAnon:/{a=$2}/^RssFile:/{f=$2}/^VmRSS:/{v=$2}/^Threads:/{t=$2}
                 END{printf "A=%d;F=%d;V=%d;T=%d",a,f,v,t}' /proc/$PID/status 2>/dev/null)" || continue
    SW=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
    [ "$A" -gt "$MAXA" ] && MAXA=$A; [ "$F" -gt "$MAXF" ] && MAXF=$F
    [ "$V" -gt "$MAXV" ] && MAXV=$V; [ "$T" -gt "$MAXT" ] && MAXT=$T
    [ "$SW" -gt "$MAXSW" ] && MAXSW=$SW
    C=$(cat "$CG/memory.swap.current" 2>/dev/null || echo 0)
    [ "${C:-0}" -gt "$MAXCG" ] && MAXCG=$C
  fi
  sleep 0.5
done
CGSW="$((MAXCG/1048576)) MiB (picco in-cgroup)"
wait $WAITPID; RC=$?

{
  echo "[sampler] RssAnon picco: $((MAXA/1024)) MiB"
  echo "[sampler] RssFile picco: $((MAXF/1024)) MiB"
  echo "[sampler] VmRSS   picco: $((MAXV/1024)) MiB"
  echo "[sampler] THREAD  picco: $MAXT   (OMP_NUM_THREADS=$THREADS)"
  echo "[sampler] swap di sistema: $SW0 MiB prima -> $MAXSW MiB picco (delta $((MAXSW-SW0)) MiB)"
  echo "[sampler] swap DELL'INFERENZA: $CGSW"
  echo "[sampler] exit code: $RC"
} >> "$LOG"
echo "=== $TAG fatto (rc=$RC) ==="
grep -E "hit rate|Speed:|TTFT|THREAD|RssAnon|RssFile|swap|cgroup" "$LOG"
