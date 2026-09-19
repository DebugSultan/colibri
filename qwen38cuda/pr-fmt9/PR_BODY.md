# cuda: decode fmt=9 (bf16) in the dense matmul

## Why

`bf16` is already a first-class dtype on the CPU side of this repo: `bf16_to_f32`
lives in `c/st.h:122` and is used by `qwen38_core.h`, `deepseek_v41.c`,
`inkling.c` and `st.h`'s own tensor readers. The CUDA backend does not know it.
A checkpoint whose dense side is bf16 end to end therefore has to keep every
dense projection on the CPU even when a device is present and the expert tier is
already running on it.

This adds the missing decode: `fmt=9` in `coli_cuda_matmul`.

## What it does

* `row_bytes(9, I) == I * 2` — two raw bytes per weight, no scale array.
* `bf16_at()`: `bf16 -> f32` is the identity on the top 16 bits, so the decoder
  is one shift. No rounding, no lookup table, and **no `<cuda_bf16.h>`
  dependency** — this compiles on the same toolchains everything else here does.
  It mirrors `st.h`'s `bf16_to_f32` bit for bit, NaN and Inf patterns included.
* A dedicated branch in `quant_matmul`, alongside the existing 4/6/7/8 branches.

## The shared scale-free predicate

`fmt=9` carries its magnitude in the weights, so like `fmt=0` (f32) and `fmt=6`
(E8, scales inside the block) it has no scale array to allocate, upload, charge
against the device budget or refresh. Upstream spells that rule out by hand, and
in more than one wording: `fmt && fmt != 6` at seven call sites, a longer variant
in the matmul epilogue, and `!fmt || fmt==6` — the same rule read backwards
— in `coli_cuda_tensor_update`. Adding a fourth scale-free format to nine
hand-written lists spelled two different ways is how such lists drift, so this PR
consolidates them into one predicate:

```c
__host__ __device__ static int fmt_scale_free(int fmt) {
    return fmt == 0 || fmt == 6 || fmt == 9;
}
```

Every converted site keeps the same truth table for formats 0..8, so no existing
container changes behaviour. Three of the conversions deserve a note:

* `coli_cuda_tensor_update()` is the one that mattered. It ended in
  `return !tensor->fmt || tensor->fmt==6 || cudaMemcpy(tensor->scales, scales, ...)`.
  That is the scale-free rule written in the opposite direction, so a sweep for
  `fmt && fmt != 6` never surfaces it, and `fmt=9` fell straight through to a
  memcpy from a NULL host pointer — `invalid argument` on the first in-place
  refresh. The test below caught it, and that is the concrete reason this PR
  consolidates the rule instead of adding a ninth hand-written list.
* `absorb_scale()` tested `!fmt` and would have read `wscale[row]` for a
  scale-free format. It is unreachable in practice — `absorb_fmt_ok()` gates the
  absorb kernels on `coli_cuda_weight_at_supported()`, which admits `{0,1,2,3,4}`
  only — so this one is hardening, not a fix for a live bug.
* the matmul epilogue already excluded `fmt != 6` explicitly; routing it through
  the predicate is what keeps `fmt=9` out of `partial[0] * scales[o]`, where
  `scales` is `NULL`.

## What it deliberately does not do

`coli_cuda_weight_at_supported()` keeps upstream's `{0,1,2,3,4}`. Callers ask
that predicate "can `weight_at` decode it", and bf16 rides `coli_cuda_matmul`'s
own branch instead — the same arrangement fmt=6, 7 and 8 already have. The
header comment now records that choice so a later reader does not "fix" it.

## Test

`c/tests/test_bf16_cuda.cu`, wired into `make cuda-test`. Four independently
falsifiable claims:

1. **DECODE** — `bf16_at()` reproduces `st.h`'s `bf16_to_f32` over all 65536 bit
   patterns, compared as raw `u32` so a NaN cannot hide a mismatch.
2. **PARITY** — `coli_cuda_matmul(fmt=9)` against a `double` CPU reference on a
   shape with an odd `I` (tail) and `S > 1`.
3. **SCALE-FREE** — upload with `scales == NULL`, `coli_cuda_tensor_bytes ==
   I*O*2`, and an in-place `coli_cuda_tensor_update()` with a NULL scale pointer.
   This claim is not decorative: it failed on the first run, and that failure is
   what exposed the `!fmt || fmt==6` site above.
4. **NEGATIVE CONTROL** — a scaled format (`fmt=1`) with `scales == NULL` must
   still be refused, so claim 3 cannot pass by the guard being gone.

It includes `backend_cuda.cu` directly (kernel-level oracle), the same pattern
`test_fp8_cuda.cu` and `test_cuda_fmt_trap_cuda.cu` use, and runs early in the
recipe: it is the cheapest kernel test there and has no host-dependent failure
mode.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
