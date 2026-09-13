# Handoff — English pass and local-path removal on the `qwen38cuda` branch

**For:** qwen (follow-up agent)
**Branch:** `qwen38cuda` (`DebugSultan/Colibri`)
**Raised:** 2026-09-13, by the owner, after review of the branch so far.
**Scope:** language and portability only. **No behavioural change to any code.**

---

## 1. Why this exists

The `qwen38cuda` branch carries ten commits whose code comments, commit
messages and planning documents are written in **Italian**, while the rest of
the repository — `qwen36_tier.h`, `backend_cuda.cu`, every pre-existing engine
header, and every commit message on `main` — is written in **English**.

The same commits also hard-code machine-specific absolute paths
(`/home/sultano/...`, `/opt/zyonix/...`) in the planning documents and in the
benchmark script, which makes the script unrunnable anywhere else.

Neither issue affects behaviour. Both make the branch harder to merge and
harder for anyone but its author to read, so they should be cleared before the
branch goes any further.

---

## 2. Hard constraint: do not rewrite published history

All ten commits are **already pushed** to `origin/qwen38cuda` (local `HEAD` and
`origin/qwen38cuda` are identical — zero divergence). Rewriting their messages
would require a force-push, which is **forbidden on this repository**.

**Therefore:**

- **Do not** `rebase -i`, `commit --amend`, or force-push anything already on
  the remote.
- Italian **commit messages stay as they are.** They are history; they are not
  the deliverable.
- Fix the **file contents** with new, ordinary commits on top.
- Write every new commit message in English, so the drift stops here.

If the owner later decides the history itself must be clean, that is a separate
decision requiring their explicit instruction — not something to do on your own
initiative.

---

## 3. What to change

### 3.1 Source comments — Italian to English

These files were touched by this branch and contain Italian comments. Line
counts are approximate matches on Italian keywords, not exact comment counts —
read each file and translate every comment you find, not only the matched
lines.

| File | Status | Notes |
|---|---|---|
| `c/omp_tune.h` | committed | ~70 added lines, heavily commented. Contains the rationale block `PERCHE' UNA RISERVA, E PERCHE' IN CORE INTERI` — this reasoning is valuable, translate it faithfully rather than trimming it. |
| `c/qwen38.c` | committed | ~15 added lines, plus one Italian comment added later (uncommitted, at the `q38_tier_warmstart` call site). |
| `c/tests/test_omp_tune.c` | committed | ~55 added lines. |
| `c/coli` | committed | ~20 added lines (Python launcher). |
| `c/qwen38_tier.h` | **uncommitted** | ~26 Italian lines. The long file-header comment is the core design rationale of the whole CUDA tier — see §5 before touching it. |
| `c/qwen38_tier.c` | **uncommitted** | ~16 Italian lines plus inline comments throughout. |
| `c/qwen38_core.h` | **uncommitted** | ~23 Italian lines, all in blocks added by this branch. Comments already in the file before this branch are English — leave them alone. |

`c/tests/test_v4_cli.py` is **already English** and needs nothing.

### 3.2 Local absolute paths

| File | Line | Content | Suggested fix |
|---|---|---|---|
| `qwen38cuda/bench-thread.sh` | 8 | `SCRATCH=/opt/zyonix/backend/colibri-q38-scratch` | `SCRATCH="${SCRATCH:-$(dirname "$0")/../../colibri-q38-scratch}"` or an explicit required env var |
| `qwen38cuda/bench-thread.sh` | 9 | `ENG=/opt/zyonix/backend/colibri/c/qwen38` | derive from the script location: `ENG="${ENG:-$(dirname "$0")/../c/qwen38}"` |
| `qwen38cuda/bench-thread.sh` | 12 | `SNAP=/home/sultano/models/Qwen3.8-Flash-Next-FP8` | `SNAP="${SNAP:?set SNAP to the model snapshot directory}"` — no default; the model path is genuinely site-specific |
| `qwen38cuda/SPEC_COLIBRI_QWEN38.md` | 113 | `Clone locale: /home/sultano/colibri, branch main, HEAD fd93c41` | keep the branch and HEAD (they are real provenance), drop the local path |
| `qwen38cuda/PIANO_NEXT.md` | 287-288, 400 | `/opt/zyonix/backend/qwen38-tiny-venv` | replace with a relative or symbolic reference, e.g. "a torch+transformers venv, passed via `make PYTHON=…`" |
| `qwen38cuda/HANDOFF_THREADS.md` | 58, 62, 64 | working-directory and memory-directory rules | **judgement call, see below** |

The three `HANDOFF_THREADS.md` lines are different in kind from the others:
they are *operating rules for this specific machine* ("work only under
`/opt/zyonix/backend`, never in `/home/sultano`"), not path constants a program
depends on. A rule that names a machine path is doing its job. **Ask the owner
before changing these** — the right answer may be to leave them, or to move
them into a clearly-marked "site-specific setup" section. Do not silently
generalise a safety rule into vagueness.

### 3.3 Planning documents — language

`qwen38cuda/*.md` (`PIANO_NEXT.md` ~408 lines, `HANDOFF_THREADS.md`,
`SPEC_COLIBRI_QWEN38.md`, `QWEN38-GPU-RECON.md`) are Italian throughout.

These are the campaign's working log and are substantially longer than the code.
**Confirm with the owner whether they are in scope** before translating them.
Two defensible positions: they are internal working notes and can stay Italian,
or they ship with the branch and should match it. That is the owner's call, not
yours. If they say yes, translate `PIANO_NEXT.md` last — it is the largest and
the most load-bearing.

---

## 4. What **not** to change

- **Any code semantics.** This is a comment-and-string pass. If you believe you
  have found a bug while translating, write it down and report it — do not fix
  it in the same commit.
- **Identifiers.** Function, variable, macro and file names are already English
  and stay exactly as they are.
- **Measured numbers, dates, and telemetry strings** quoted in comments and
  documents. `-25,5 %`, `1,18 s/token`, `08/09` and so on are evidence. You may
  convert decimal commas to decimal points for consistency with English prose,
  but never round, restate, or "tidy" a measurement.
- **Commit messages already pushed** (§2).
- **`stderr` telemetry strings already in English** (the `[q38tier]` banners) —
  they are English already; check, don't assume.

---

## 5. Translation guidance for the tier header

`c/qwen38_tier.h` opens with a long comment that is the single most important
piece of prose on this branch: it explains why this tier is structured
differently from `qwen36_tier.h`. The argument, in brief, so you can check your
translation preserves it:

1. `qwen36_tier` requires every expert to be RAM-resident and keeps raw RAM
   pointers, so it can fetch weights by itself.
2. qwen38 cannot: the model is 185 GB on a 64 GB machine, its RAM slots are an
   **evicting LRU**, and under `COLI_MAP_EXPERTS=1` a slot does not even own its
   bytes — it points into a file mapping the kernel may drop.
3. Two consequences follow, and they are the whole design: the tier **owns its
   VRAM copy and never stores a RAM address**, and the tier **cannot promote on
   its own** — the engine *offers* it bytes at the one moment they exist.
4. In exchange the format costs nothing: with `native_fp8` the RAM slot already
   holds raw E4M3 plus 128×128 block scales, bit-identical to the backend's
   `fmt=8`, so staging is a `memcpy` and not a conversion.

Keep all four points. If your English version is shorter because one of them
went missing, it is wrong. Prefer a faithful, slightly long translation over a
crisp one that drops the reasoning — the reasoning is why the file exists.

Style note: match the surrounding codebase. `qwen36_tier.h` is the reference
for tone — direct, technical, explains *why* rather than restating *what*.

---

## 6. Suggested commit sequence

Small, reviewable, one concern each:

1. `qwen38: translate OpenMP tuner comments to English`
2. `qwen38: translate CUDA expert tier comments to English` (after the Phase 2
   code is committed — coordinate, see §7)
3. `qwen38cuda: make bench-thread.sh portable`
4. `qwen38cuda: drop local absolute paths from planning docs`
5. (only if the owner approves §3.3) `qwen38cuda: translate planning documents`

---

## 7. Coordination warning — uncommitted work in flight

At the time of writing, **Phase 2 (the CUDA VRAM expert tier) is code-complete
but not yet committed**: `c/qwen38_tier.c` and `c/qwen38_tier.h` are untracked,
and `c/Makefile`, `c/qwen38_core.h` and `c/qwen38.c` are modified in the
worktree.

**Do not start on those three files until that work has been committed**, or you
will conflict with it. `c/omp_tune.h`, `c/tests/test_omp_tune.c` and `c/coli`
are already committed and safe to begin on immediately.

---

## 8. Acceptance checks

Run these from the `c/` directory; all must pass.

1. **Nothing changed but comments and strings.** For each source file touched:
   ```
   gcc -fpreprocessed -dD -E -P <file>   # before and after, must be identical
   ```
   This strips comments and shows only real code. Any difference is a bug in
   your pass.
2. **Both builds stay clean:** `make qwen38` and `make CUDA=1 qwen38`.
3. **Tests stay green:** `make test-omp-tune` (or the equivalent target for
   `tests/test_omp_tune.c`) and `python3 -m unittest` for `tests/test_v4_cli.py`
   — note `pytest` is **not installed** on this machine.
4. **No Italian left in the files you touched.** Grep for the common markers:
   ```
   grep -inE "perche|cioe|quindi|invece|percio|il motore|deve |viene " <files>
   ```
   Expect zero matches. Be aware this catches only the frequent words — read the
   files too.
5. **No local paths left** in the files listed in §3.2:
   ```
   grep -nE "/home/[a-z]+|/opt/zyonix" <files>
   ```
   Expect zero matches, except any `HANDOFF_THREADS.md` lines the owner
   explicitly chose to keep.
6. **`bench-thread.sh` still runs** after the path change — it is a measurement
   script, and a broken one silently invalidates future comparisons.

---

## 9. Standing rules on this branch

Inherited from `HANDOFF_THREADS.md`; they apply to you as well.

- **An isolated green proves nothing**: repeat every measurement, count the
  payload.
- **KL + top-1, never PPL** for quality comparisons.
- Anything running on CPU stays **F32 or BF16**.
- **Never force-push, never push to `main`/`master`, never merge.**
- **Phase 1 (int4) must not be started** without the owner's explicit go-ahead.
