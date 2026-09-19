#!/usr/bin/env bash
# ribatteria.sh -- fixed-memory-regime A/B runner for the qwen38 campaign.
#
# Why this exists: notte-17 showed that swap/reclaim pressure was a confounder of
# every A/B measured in notte-15 and notte-16 (ours 3.64 -> 2.73 s TTFT with the
# swap valve alone). A number is only comparable to another number taken under a
# DECLARED and VERIFIED memory regime, so this script refuses to run until the
# regime it was asked for is the regime the machine is actually in, and it records
# the evidence next to every result.
#
# What it records per run (notte-18 inheritance):
#   - pswpin/pswpout deltas from /proc/vmstat and the swap regime in force
#   - read_bytes / write_bytes deltas from /proc/<pid>/io   (real NVMe traffic)
#   - the `[qwen38 expert I/O] ... parallel-batches=N` line. N == 0 means the
#     per-layer expert cache refused to batch (the #1594 cap<topk cliff) and the
#     run is INVALID as a comparison -- the script says so instead of reporting it.
#   - an iobench storage probe before and after the battery, same shape the #1594
#     reporter used: 64 random reads x 4.7 MB, 10 threads, O_DIRECT.
#
# Usage:
#   ./ribatteria.sh --swap off --arms "ours hybrid" --prompt prompt-2k.txt
#   ./ribatteria.sh --swap off --dry-run          # print the plan, touch nothing
#
# Needs root for drop_caches and swapoff/swapon; everything else runs as $RUN_USER.

set -uo pipefail

# ---- fixed paths (campaign constants) ---------------------------------------
REPO=/opt/zyonix/backend/colibri
SCRATCH=/opt/zyonix/backend/colibri-q38-scratch
SNAP_FP8=/home/sultano/models/Qwen3.8-Flash-Next-FP8
SNAP_INT4=$SCRATCH/qwen38-int4-gs64
HEAT=$SCRATCH/heat_qth1.bin
BIN=$REPO/c/qwen38
IOBENCH=$REPO/c/iobench
PROBE_SHARD=$SNAP_INT4/model-00041.safetensors
PROD_SVC=llama-server-qwen38-prod.service
RUN_USER=sultano

# ---- run shape --------------------------------------------------------------
CACHE=8192          # KV cache per layer
BITS=8
N_NEW=64
OMP=20
DEV_RESERVE_MB=1024

# ---- arms: name -> extra env. Keep the names stable: they become log names. --
declare -A ARM_ENV=(
  [ours]=""                                             # our int4 arena + host-pinned store
  [hybrid]="Q38_TIER_TRUNK=2"                           # our arena for experts + their int8 trunk
  [trunk]="Q38_TIER_TRUNK=1"                            # upstream tier as-is
  [hybrid_pf]="Q38_TIER_TRUNK=2 Q38_TRUNK_PREFILL=1"    # + batched GPU prefill (needs the trunk)
)

SWAP_MODE=""; ARMS="ours hybrid"; PROMPT=prompt-2k.txt; DRY=0
OUT=$SCRATCH/ribatteria-$(date +%Y%m%d-%H%M%S)

while [ $# -gt 0 ]; do
  case "$1" in
    --swap)    SWAP_MODE="${2:-}"; shift 2 ;;
    --arms)    ARMS="${2:-}"; shift 2 ;;
    --prompt)  PROMPT="${2:-}"; shift 2 ;;
    --out)     OUT="${2:-}"; shift 2 ;;
    --n-new)   N_NEW="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "REFUSED: $*" >&2; exit 1; }
say() { echo "[ribatteria] $*"; }

[ -n "$SWAP_MODE" ] || die "--swap on|off is mandatory: the regime must be DECLARED, not inherited."
case "$SWAP_MODE" in on|off) ;; *) die "--swap takes on or off, got '$SWAP_MODE'" ;; esac

# ---- preflight: every condition that silently ruined a past measurement ------
preflight() {
  local fail=0

  # 1. prod must be down: it holds ~15.7 GiB on each card.
  if systemctl is-active --quiet "$PROD_SVC"; then
    echo "  [x] $PROD_SVC is ACTIVE -- the GPU window is not open"; fail=1
  else echo "  [ok] $PROD_SVC inactive"; fi

  # 2. the cards must actually be free.
  local used gpu_bad=0
  while read -r used; do
    if [ "$used" -gt 2000 ]; then echo "  [x] a GPU still holds ${used} MiB"; gpu_bad=1; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ $gpu_bad -eq 0 ]; then echo "  [ok] both GPUs free"; else fail=1; fi

  # 3. the binary must be the CUDA one and newer than HEAD (debt 5, notte-15:
  #    `make check` overwrites c/qwen38 with the NOCUDA build -> rc=127).
  [ -x "$BIN" ] || { echo "  [x] $BIN missing"; fail=1; }
  if [ -x "$BIN" ]; then
    if ldd "$BIN" | grep -q libcudart; then echo "  [ok] binary is CUDA-linked"
    else echo "  [x] $BIN is NOT CUDA-linked -- run: make -C $REPO/c CUDA=1 qwen38"; fail=1; fi
    local head_t bin_t
    head_t=$(git -C "$REPO" log -1 --format=%ct)
    bin_t=$(stat -c %Y "$BIN")
    if [ "$bin_t" -ge "$head_t" ]; then echo "  [ok] binary newer than HEAD"
    else echo "  [x] $BIN predates HEAD -- rebuild before measuring"; fail=1; fi
  fi

  # 4. no concurrent writer on the same NVMe (rule 4 of HANDOFF_FINESTRA: a
  #    converter running alongside once polluted an entire measurement).
  local wr1 wr2
  wr1=$(awk '/^nvme/ {s+=$10} END{print s+0}' /proc/diskstats); sleep 2
  wr2=$(awk '/^nvme/ {s+=$10} END{print s+0}' /proc/diskstats)
  local wsect=$(( wr2 - wr1 ))
  if [ "$wsect" -lt 40000 ]; then echo "  [ok] disk quiet (${wsect} sectors written in 2 s)"
  else echo "  [x] disk BUSY (${wsect} sectors written in 2 s) -- find the writer first"; fail=1; fi

  # 5. the declared swap regime must be the real one.
  local swaptotal
  swaptotal=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  if [ "$SWAP_MODE" = off ] && [ "$swaptotal" -ne 0 ]; then
    echo "  [x] --swap off but SwapTotal=${swaptotal} kB -- run: sudo swapoff -a"; fail=1
  elif [ "$SWAP_MODE" = on ] && [ "$swaptotal" -eq 0 ]; then
    echo "  [x] --swap on but swap is off -- run: sudo swapon -a"; fail=1
  else echo "  [ok] swap regime '$SWAP_MODE' verified (SwapTotal=${swaptotal} kB)"; fi

  # 6. inputs present.
  local in_bad=0
  for f in "$SNAP_FP8/tokenizer.json" "$SNAP_INT4" "$HEAT" "$SCRATCH/$PROMPT" "$IOBENCH" "$PROBE_SHARD"; do
    [ -e "$f" ] || { echo "  [x] missing: $f"; in_bad=1; }
  done
  if [ $in_bad -eq 0 ]; then echo "  [ok] snapshot, heat file, prompt, iobench and probe shard present"
  else fail=1; fi

  return $fail
}

vmstat_field() { awk -v k="$1" '$1==k {print $2}' /proc/vmstat; }

storage_probe() {   # $1 = label
  say "storage probe ($1): iobench 4.7 MB x 64, 10 threads, O_DIRECT"
  [ "$DRY" = 1 ] && return 0
  "$IOBENCH" "$PROBE_SHARD" 4.7 64 10 1 2>&1 | tee "$OUT/iobench-$1.txt"
}

run_arm() {   # $1 = arm name
  local arm="$1" extra="${ARM_ENV[$1]:-}" log="$OUT/log-$1.txt" meta="$OUT/meta-$1.txt"

  say "arm '$arm': env [${extra:-<none>}]"
  if [ "$DRY" = 1 ]; then
    echo "    would run: $BIN $CACHE $BITS $SCRATCH/$PROMPT  (N_NEW=$N_NEW, COLI_TIMERS=1)"
    return 0
  fi

  sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3

  local swin0 swout0 t0
  swin0=$(vmstat_field pswpin); swout0=$(vmstat_field pswpout); t0=$(date +%s.%N)

  # COLI_TIMERS=1 is what publishes the `[qwen38 expert I/O]` line. Without it
  # there is no parallel-batches counter and the run cannot be validated.
  # shellcheck disable=SC2086
  setsid sudo -u "$RUN_USER" prlimit --memlock=unlimited:unlimited -- env \
      SNAP="$SNAP_FP8" Q38_INT4_SNAP="$SNAP_INT4" \
      COLI_CUDA=1 COLI_MAP_EXPERTS=1 COLI_TIMERS=1 \
      HEAT_FILE="$HEAT" OMP_NUM_THREADS="$OMP" \
      N_NEW="$N_NEW" NOSTREAM=1 Q38T_DEV_RESERVE_MB="$DEV_RESERVE_MB" \
      $extra \
      "$BIN" "$CACHE" "$BITS" "$SCRATCH/$PROMPT" > "$log" 2>&1 &
  local pid=$!

  # sample the process I/O counters while it lives; the last sample before exit
  # is the run total (the file disappears with the pid).
  local io_last=""
  while kill -0 "$pid" 2>/dev/null; do
    io_last=$(cat "/proc/$pid/io" 2>/dev/null) || true
    sleep 2
  done
  wait "$pid"; local rc=$?

  local t1 swin1 swout1
  t1=$(date +%s.%N); swin1=$(vmstat_field pswpin); swout1=$(vmstat_field pswpout)

  {
    echo "arm            : $arm"
    echo "extra env      : ${extra:-<none>}"
    echo "exit code      : $rc"
    echo "wall seconds   : $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"
    echo "swap regime    : $SWAP_MODE"
    echo "pswpin delta   : $(( swin1 - swin0 )) pages"
    echo "pswpout delta  : $(( swout1 - swout0 )) pages"
    echo "--- last /proc/<pid>/io sample ---"
    echo "$io_last"
    echo "--- expert I/O counters ---"
    grep -E '\[qwen38 expert I/O\]' "$log" || echo "(absent: COLI_TIMERS was not honoured)"
  } > "$meta"

  # the silent switch: batching off means the comparison is meaningless.
  local pb
  pb=$(grep -oE 'parallel-batches=[0-9]+' "$log" | tail -1 | cut -d= -f2)
  if [ -z "$pb" ]; then
    say "WARNING arm '$arm': no parallel-batches counter -- run NOT comparable"
  elif [ "$pb" = 0 ]; then
    say "INVALID arm '$arm': parallel-batches=0 (cap<topk cliff, #1594) -- do not report this number"
  else
    say "arm '$arm': parallel-batches=$pb, rc=$rc"
  fi
  [ "$rc" -eq 0 ] || say "WARNING arm '$arm' exited rc=$rc -- see $log"
}

# ---- go ---------------------------------------------------------------------
say "preflight"
if ! preflight; then die "preflight failed -- nothing was run, nothing was measured."; fi

if [ "$DRY" = 1 ]; then say "dry run: plan only"; else mkdir -p "$OUT"; fi
say "output: $OUT"

if [ "$DRY" != 1 ]; then
  { echo "date        : $(date -Is)"
    echo "HEAD        : $(git -C "$REPO" log -1 --format='%H %s')"
    echo "binary mtime: $(stat -c '%y' "$BIN")"
    echo "swap mode   : $SWAP_MODE"
    echo "prompt      : $PROMPT"
    echo "arms        : $ARMS"
    echo "shape       : cache=$CACHE bits=$BITS N_NEW=$N_NEW OMP=$OMP reserve=${DEV_RESERVE_MB}MB"
    echo "kernel      : $(uname -r)"
    echo "swappiness  : $(cat /proc/sys/vm/swappiness)"
  } > "$OUT/manifest.txt"
fi

storage_probe before
for arm in $ARMS; do
  [ -n "${ARM_ENV[$arm]+set}" ] || die "unknown arm '$arm' (known: ${!ARM_ENV[*]})"
  run_arm "$arm"
done
storage_probe after

say "done. Read $OUT/meta-*.txt before believing any tok/s in $OUT/log-*.txt."
