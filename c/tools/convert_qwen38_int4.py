"""
Qwen3.8-Flash-Next -> colibri int4 group-scaled expert container (fmt=4, gs64).

DISK-SAFE, the strategy of convert_glm53.py: fetch the source shards ONE unit of
work needs, convert them, delete them, move on. Peak disk is the output plus the
shards of a single unit (~5 GB), never the 360 GB of the repo.

WHAT THIS TOOL EMITS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
Experts ONLY. The container is ~69,4 GB of routed experts; dense weights, PLE,
vision and MTP scaffolding keep being read straight from the official FP8
checkpoint, because st_init_multi() takes extra directories and the engine can
therefore mount "int4 experts + FP8 everything else" without a second copy of
the 3 % of parameters that are not experts. No tokenizer is copied either: the
FP8 checkpoint already carries it and duplicating it only creates a second
source of truth that can drift.

  per expert -> ONE `...merged_weight`  U8, gate||up||down nibbles, 2.457.600 B
                ONE `...qs`             F32, gate||up||down scales, 76.800 f32
                                                    = 2,6367 MiB per expert
  25.088 experts (48 MoE layers + 1 MTP layer)      = 69.363.302.400 B

Output shard naming follows the qwen36 container: one `model-{L:05d}.safetensors`
per MoE layer, read lazily by st_init(), which scans the directory for any
*.safetensors and needs no index.json.

THE NIBBLE CONVENTION, AND WHY IT IS NOT qwen36's
-------------------------------------------------
Two conventions exist in this tree and they differ by XOR 0x8, so getting it
wrong is silent garbage, not a crash:

  * OFFSET BINARY, stored u = q + 8, decoded (u - 8).  This is fmt=4, what
    quant.h's matmul_i4_grouped() decodes, and what convert_glm53.py writes.
  * TWO'S COMPLEMENT, stored u = q & 0xF, sign-extended on read.  This is what
    qwen36.c's own unpack_int4_to_int8() expects from convert_qwen36.py.

This container uses OFFSET BINARY / fmt=4, for three reasons that outlive taste:
matmul_i4_grouped() already implements the group-scaled AVX2 GEMV for exactly
this layout, so the expert path reuses a tested kernel instead of a new one; the
scale layout it wants, [O][I/gs] row-major, is the one specified here; and in a
decode-inside-GEMV kernel offset binary costs `and` + `sub` per vector where
two's complement costs `and` + `xor` + `sub`, AVX2 having no 8-bit shift. That
one instruction lands on the decode path, which is the measured risk in this
lever rather than an afterthought.

Packing per row and packing the flat gate||up||down blob produce IDENTICAL bytes
here, because every input dimension is even (2560 and 640), so `rb = (I+1)/2`
tiles the row exactly. The two descriptions of the container agree; they would
not for an odd I, which is asserted below.

TWO SOURCE LAYOUTS, BOTH MEASURED FROM THE REAL INDEXES
--------------------------------------------------------
  * BF16 repo (default, Qwen/Qwen3.8-Flash-Next @ de4b8e4d): experts are FUSED
    3D tensors, one pair per layer --
        `...mlp.experts.gate_up_proj`  BF16 [512, 1280, 2560]
        `...mlp.experts.down_proj`     BF16 [512, 2560,  640]
    Inside gate_up_proj rows 0:640 are gate and 640:1280 are up (verified by
    cosine against the FP8 checkpoint: 0,99964 for that pairing versus 0,0086
    and 0,0024 for the swapped and interleaved readings).
  * FP8 checkpoint (--indir, for testing): experts are SEPARATE 2D tensors per
    expert, `...experts.{e}.{gate,up,down}_proj.weight`, each with a 128x128
    block `_scale_inv`.

Both layouts put a layer's gate+up in one shard and its down in another, so the
unit of work is a LAYER, not a shard: a layer needs 2 source shards and yields
one 1,4156 GB output shard. The two are adjacent for the 48 language-model
layers and NOT adjacent for MTP (shards 60 and 62 in the BF16 repo), which is
why the plan resolves the shard set per unit instead of assuming a stride.

NUMPY ONLY, NO TORCH
--------------------
torch is not installed on the target machine, and pulling it in to read two
dtypes would be a 2 GB dependency for a lookup table. BF16 and FP8-e4m3 are
decoded here directly; the e4m3 table reproduces quant.h's E4M3_LUT, including
its NaN policy (only 0x7F/0xFF are NaN, max finite 448).

USAGE
  python3 tools/convert_qwen38_int4.py --outdir <dir>                  # BF16 repo
  python3 tools/convert_qwen38_int4.py --outdir <dir> --limit-shards 1 # layer 0 only
  python3 tools/convert_qwen38_int4.py --indir <fp8-dir> --outdir <dir>
  python3 tools/convert_qwen38_int4.py --selftest
"""
import argparse
import json
import mmap
import os
import re
import shutil
import sys
import time

import numpy as np

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d"

# Geometry the container is built for. These are not defaults to be overridden:
# they are what the checkpoint was measured to contain, and a mismatch means the
# source is not the model this tool knows how to convert.
EXPECT = {
    "num_hidden_layers": 48,
    "hidden_size": 2560,
    "num_experts": 512,
    "moe_intermediate_size": 640,
}
GROUP_SIZE = 64


# ------------------------------------------------------------------ dtypes
def _e4m3_lut():
    """256-entry FP8-e4m3fn decode table, the numpy twin of quant.h's E4M3_LUT.

    sign(1) exp(4, bias 7) mant(3); subnormal at exp==0; OCP E4M3-FN, so exp==0xF
    is NOT infinity -- only mant==0x7 there is NaN, and the largest finite value
    is exp=0xF/mant=0x6 -> 448. NaN is left as a real NaN and allowed to
    propagate, the policy quant.h documents for fmt=8."""
    b = np.arange(256, dtype=np.uint8)
    sign = np.where(b >> 7, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    mant = (b & 0x7).astype(np.float64)
    val = np.where(exp == 0, mant / 8.0 * 2.0 ** -6, (1.0 + mant / 8.0) * 2.0 ** (exp - 7))
    val = sign * val
    val[(b & 0x7F) == 0x7F] = np.nan
    return val.astype(np.float32)


E4M3_LUT = _e4m3_lut()


def _bf16_to_f32(raw):
    """bf16 is the top 16 bits of an f32: widen and shift, no table needed."""
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------- safetensors reader
class Shard:
    """Minimal read-only safetensors reader over mmap.

    Why not safetensors' own numpy backend: it refuses BF16 and F8_E4M3, the two
    dtypes this checkpoint is made of. Why mmap: a fused gate_up_proj is 3,36 GB
    and only one expert's 6,55 MB slice is needed at a time, so the pages that
    are touched are the pages that are read."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        nh = int.from_bytes(self._f.read(8), "little")
        self.header = json.loads(self._f.read(nh))
        self.header.pop("__metadata__", None)
        self._base = 8 + nh
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        self._mm.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def keys(self):
        return self.header.keys()

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def get(self, name, row=None):
        """Whole tensor as f32, or -- with `row` -- only slice [row] of a 3D one.

        The row form is what keeps peak RSS at one expert instead of one layer."""
        t = self.header[name]
        shape = tuple(t["shape"])
        start, stop = t["data_offsets"]
        dt = t["dtype"]
        width = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}.get(dt)
        if width is None:
            raise ValueError(f"{self.path}: {name}: unsupported dtype {dt}")
        if row is not None:
            if len(shape) != 3:
                raise ValueError(f"{name}: row slice of a {len(shape)}D tensor")
            per = shape[1] * shape[2] * width
            start += row * per
            stop = start + per
            shape = shape[1:]
        raw = self._mm[self._base + start:self._base + stop]
        if dt == "F8_E4M3":
            out = E4M3_LUT[np.frombuffer(raw, np.uint8)]
        elif dt == "BF16":
            out = _bf16_to_f32(raw)
        elif dt == "F16":
            out = np.frombuffer(raw, np.float16).astype(np.float32)
        else:
            out = np.frombuffer(raw, np.float32)
        return out.reshape(shape)


# ------------------------------------------------------------- quantizer
def quant_int4_grouped(w, gs=GROUP_SIZE):
    """int4 group-scaled: one f32 scale per `gs` elements along the input axis.

    Returns (packed uint8 [O*I/2], scales f32 [O*ngroups]) with the fmt=4
    convention: stored nibble u = q + 8, low nibble first, scales row-major
    [O][I/gs] -- byte-for-byte what quant.h's matmul_i4_grouped() decodes.

    The 1e-8 floor is applied to the SCALE, not to amax: a zero expert (the
    checkpoint ships 24 all-zero projections) then quantizes to q=0 with a
    harmless positive scale, instead of dividing by zero and writing NaN into a
    tensor that decodes to zero either way."""
    O, I = w.shape
    qmax = 7
    ngroups = (I + gs - 1) // gs
    Ipad = ngroups * gs
    if Ipad == I:
        wr = w.reshape(O, ngroups, gs)
    else:
        wpad = np.zeros((O, Ipad), np.float32)
        wpad[:, :I] = w
        wr = wpad.reshape(O, ngroups, gs)
    amax = np.abs(wr).max(axis=2, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)
    q = q.reshape(O, Ipad)[:, :I]
    if I & 1:
        # Not reachable for this model (I is 2560 or 640) and the flat-vs-per-row
        # packing equivalence the container relies on would break here, so refuse
        # rather than emit a blob whose two descriptions disagree.
        raise ValueError(f"odd input dim {I}: row packing and flat packing diverge")
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    packed = lo | (hi << 4)
    return packed.reshape(-1), s[:, :, 0].astype(np.float32).reshape(-1)


def merge_expert(gate, up, down, gs=GROUP_SIZE):
    """gate||up||down -> (merged_weight uint8, qs float32), the container pair."""
    parts = [quant_int4_grouped(t, gs) for t in (gate, up, down)]
    merged = np.concatenate([p[0] for p in parts])
    qs = np.concatenate([p[1] for p in parts])
    return merged, qs


def unpack_int4_grouped(packed, scales, O, I, gs=GROUP_SIZE):
    """Inverse of quant_int4_grouped, for the selftest and for verification."""
    b = packed.reshape(O, I // 2)
    q = np.empty((O, I), np.float32)
    q[:, 0::2] = (b & 0xF).astype(np.int32) - 8
    q[:, 1::2] = (b >> 4).astype(np.int32) - 8
    ng = (I + gs - 1) // gs
    s = scales.reshape(O, ng)
    out = np.empty((O, I), np.float32)
    for g in range(ng):
        lo, hi = g * gs, min((g + 1) * gs, I)
        out[:, lo:hi] = q[:, lo:hi] * s[:, g:g + 1]
    return out


# ---------------------------------------------------------- classification
# Every tensor name in both source layouts, by group. Entries are suffixes or
# fragments; an unrecognised name STOPS the conversion instead of being skipped
# in silence, which is the whole point of the guard: a renamed tensor in a future
# revision must be noticed here, not three hours into a run.
EXPERT_FUSED = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
EXPERT_SPLIT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
KNOWN_NON_EXPERT = (
    "lm_head.weight",
    "embed_tokens.weight",
    "hyper_connection_mixer.",
    "attn_hyper_connection.",
    "mlp_hyper_connection.",
    "linear_attn.",
    "mlp.gate.weight",
    "mlp.shared_expert.",
    "mlp.shared_expert_gate.weight",
    ".ple.",
    "self_attn.",
    "model.visual.",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "norm.weight",
)


def classify(name):
    """expert | consumed | skip. Raises KeyError on an unknown name."""
    if name.endswith("_scale_inv") or name.endswith(".weight_scale"):
        return "consumed"                  # FP8 block scale: consumed with its weight
    if any(name.endswith(k) for k in EXPERT_FUSED) or EXPERT_SPLIT.search(name):
        return "expert"
    for k in KNOWN_NON_EXPERT:
        if name.endswith(k) or k in name:
            return "skip"                  # stays in the FP8 checkpoint, by design
    raise KeyError(name)


# ------------------------------------------------------------------ plan
def _unit_key(name):
    """The MoE layer a fused/split expert tensor belongs to, or None."""
    m = re.match(r"^(model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+))\.mlp\.experts\.", name)
    if not m:
        return None
    if m.group(2) is not None:
        return ("layer", int(m.group(2)))
    return ("mtp", int(m.group(3)))


def build_plan(weight_map):
    """[(unit, prefix, {shards}, fused?)] in checkpoint order, one entry per layer.

    Also the param-guard on names: every tensor is classified, so a source whose
    naming drifted stops here rather than producing a silently short container."""
    units = {}
    for name, shard in weight_map.items():
        kind = classify(name)
        if kind != "expert":
            continue
        key = _unit_key(name)
        if key is None:
            raise KeyError(f"expert tensor outside a known layer namespace: {name}")
        prefix = name.split(".mlp.experts.")[0]
        entry = units.setdefault(key, {"prefix": prefix, "shards": set(), "fused": False})
        if entry["prefix"] != prefix:
            raise ValueError(f"unit {key}: two prefixes, {entry['prefix']} and {prefix}")
        entry["shards"].add(shard)
        if any(name.endswith(k) for k in EXPERT_FUSED):
            entry["fused"] = True
    plan = []
    for key in sorted(units, key=lambda k: (k[0] == "mtp", k[1])):
        e = units[key]
        plan.append((key, e["prefix"], e["shards"], e["fused"]))
    return plan


def check_geometry(config, where):
    """Refuse a source whose declared geometry is not the one measured here."""
    model_type = config.get("model_type")
    if model_type not in ("qwen4_exp", "qwen4_exp_text"):
        raise SystemExit(f"ERROR: {where}: model_type is {model_type!r}, not Qwen3.8-Flash-Next. "
                         "Other families have their own tools under c/tools/.")
    text = config.get("text_config", config)
    for key, want in EXPECT.items():
        got = text.get(key, config.get(key))
        if got != want:
            raise SystemExit(f"ERROR: {where}: {key} is {got}, expected {want}. "
                             "This container's geometry is compiled into the "
                             "engine-side loader; refusing to guess.")


# ------------------------------------------------------------------ fetch
def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def guard_space(outdir, min_free):
    """Stop BEFORE filling the disk rather than after."""
    while True:
        have = free_gb(outdir)
        if have >= min_free:
            return
        print(f"[WAIT] free space {have:.0f} GB below {min_free:.0f} GB on the output "
              f"volume - free some and this resumes by itself", flush=True)
        time.sleep(60)


def fetch_curl(repo, revision, filename, dest_dir, expected_size, token=None,
               chunk_bytes=256 << 20, idle_limit=40):
    """Download in explicit 256 MB ranged blocks, appending.

    Same reasoning as convert_glm53.py: this machine's link drops, and a single
    long connection loses the whole shard when it does. Each block is
    independent and verifiable, so a drop costs one block, never the hours
    already on disk. Gives up only after `idle_limit` consecutive attempts that
    add not one byte."""
    import subprocess
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    if not expected_size:
        raise RuntimeError(f"unknown expected size for {filename}")
    idle = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expected_size:
            return path
        if have > expected_size:                  # source changed, or a dirty file
            os.remove(path)
            continue
        stop = min(have + chunk_bytes, expected_size) - 1
        command = ["curl", "-sL", "-4", "--retry", "3", "--retry-delay", "5",
                   "--speed-limit", "50000", "--speed-time", "120",
                   "-r", f"{have}-{stop}", "--output", "-", url]
        if token:
            command[1:1] = ["-H", f"Authorization: Bearer {token}"]
        with open(path, "ab") as sink:
            subprocess.run(command, stdout=sink, check=False)
        grown = (os.path.getsize(path) if os.path.exists(path) else 0) - have
        if grown <= 0:
            idle += 1
            if idle >= idle_limit:
                raise RuntimeError(f"{filename}: no progress in {idle_limit} attempts "
                                   f"({have}/{expected_size} bytes)")
            time.sleep(min(5 * idle, 60))
        else:
            idle = 0


def hf_token():
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    path = os.path.expanduser("~/.hf_token")
    if os.path.isfile(path):
        return open(path).read().strip()
    return None


# ------------------------------------------------------------- conversion
def convert_unit(prefix, fused, shards, out_path, gs, expect_experts):
    """One MoE layer -> one output shard of 512 merged_weight/qs pairs."""
    from safetensors.numpy import save_file

    opened = {p: Shard(p) for p in shards}
    try:
        index = {}
        for path, sh in opened.items():
            for name in sh.keys():
                index[name] = sh

        def find(name):
            sh = index.get(name)
            if sh is None:
                raise KeyError(f"{name}: not in the shards of this unit")
            return sh

        out = {}
        zero_projections = 0
        if fused:
            gu_name, dn_name = prefix + ".mlp.experts.gate_up_proj", prefix + ".mlp.experts.down_proj"
            gu, dn = find(gu_name), find(dn_name)
            E, two_inter, hidden = gu.shape(gu_name)
            E2, hidden2, inter = dn.shape(dn_name)
            if (E, E2, hidden, hidden2, two_inter) != (expect_experts, expect_experts,
                                                       EXPECT["hidden_size"], EXPECT["hidden_size"],
                                                       2 * EXPECT["moe_intermediate_size"]):
                raise ValueError(f"{prefix}: unexpected fused shapes "
                                 f"{gu.shape(gu_name)} / {dn.shape(dn_name)}")
            for e in range(E):
                block = gu.get(gu_name, row=e)          # [2*inter, hidden]
                gate, up = block[:inter], block[inter:]
                down = dn.get(dn_name, row=e)           # [hidden, inter]
                merged, qs = merge_expert(gate, up, down, gs)
                out[f"{prefix}.mlp.experts.{e}.merged_weight"] = merged
                out[f"{prefix}.mlp.experts.{e}.qs"] = qs
                zero_projections += sum(1 for t in (gate, up, down) if not t.any())
        else:
            def deq(sh, name):
                w = sh.get(name)
                sc = sh.get(name + "_scale_inv")
                O, I = w.shape
                sc = np.repeat(np.repeat(sc, 128, 0), 128, 1)[:O, :I]
                return w * sc

            for e in range(expect_experts):
                base = f"{prefix}.mlp.experts.{e}."
                gate = deq(find(base + "gate_proj.weight"), base + "gate_proj.weight")
                up = deq(find(base + "up_proj.weight"), base + "up_proj.weight")
                down = deq(find(base + "down_proj.weight"), base + "down_proj.weight")
                if (gate.shape != (EXPECT["moe_intermediate_size"], EXPECT["hidden_size"])
                        or up.shape != gate.shape
                        or down.shape != (EXPECT["hidden_size"], EXPECT["moe_intermediate_size"])):
                    raise ValueError(f"{base}: unexpected shapes {gate.shape}/{up.shape}/{down.shape}")
                merged, qs = merge_expert(gate, up, down, gs)
                out[f"{prefix}.mlp.experts.{e}.merged_weight"] = merged
                out[f"{prefix}.mlp.experts.{e}.qs"] = qs
                zero_projections += sum(1 for t in (gate, up, down) if not t.any())
        save_file(out, out_path)
        return {"experts": len(out) // 2, "zero_projections": zero_projections}
    finally:
        for sh in opened.values():
            sh.close()


# ------------------------------------------------------------------ selftest
def selftest():
    """Round-trip against the decode formula the engine uses, plus the bit
    layout and the two failure modes that matter (zero tensor, odd I)."""
    import tempfile
    from safetensors.numpy import save_file, load_file

    rng = np.random.default_rng(1234)
    O, I = 64, 128
    w = rng.standard_normal((O, I), dtype=np.float32)
    packed, scales = quant_int4_grouped(w, 64)
    assert packed.shape == (O * I // 2,), packed.shape
    assert scales.shape == (O * (I // 64),), scales.shape

    # Bit layout: low nibble is element 2k, high nibble 2k+1, both offset by +8,
    # exactly how quant.h's matmul_i4_grouped() reads them.
    back = unpack_int4_grouped(packed, scales, O, I, 64)
    err = np.abs(back - w).max()
    step = scales.reshape(O, I // 64).repeat(64, axis=1)
    assert err <= step.max() * 0.5 + 1e-6, (err, step.max())
    # Expected relative error is not a free parameter: for gaussian weights the
    # amax of a 64-sample group sits near 2,7 sigma, so the step is amax/7 and
    # the uniform quantization error has rms step/sqrt(12) ~ 0,11 sigma. Bound it
    # on BOTH sides -- a suspiciously small error means the test is measuring
    # nothing (e.g. w read back through the same rounding).
    rel = np.linalg.norm(back - w) / np.linalg.norm(w)
    assert 0.08 < rel < 0.14, rel

    # Counterproof: reading the nibbles with the OTHER convention (two's
    # complement, qwen36's) must visibly break, or this test proves nothing.
    b = packed.reshape(O, I // 2)
    q_wrong = np.empty((O, I), np.float32)
    lo = (b & 0xF).astype(np.int32)
    hi = (b >> 4).astype(np.int32)
    q_wrong[:, 0::2] = np.where(lo & 8, lo - 16, lo)
    q_wrong[:, 1::2] = np.where(hi & 8, hi - 16, hi)
    wrong = q_wrong * scales.reshape(O, I // 64).repeat(64, axis=1)
    rel_wrong = np.linalg.norm(wrong - w) / np.linalg.norm(w)
    assert rel_wrong > 0.5, f"two's-complement misread went unnoticed: rel={rel_wrong}"

    # Zero projection: no NaN, no division by zero, scale at the floor.
    zp, zs = quant_int4_grouped(np.zeros((8, 64), np.float32), 64)
    assert np.all(zp == 0x88), "zero must encode as q=0, i.e. nibble 8"
    assert np.all(zs == 1e-8), zs[:4]
    assert np.isfinite(zs).all()

    # Odd input dim must be refused, not silently mis-packed.
    try:
        quant_int4_grouped(np.zeros((2, 65), np.float32), 64)
        raise AssertionError("odd I was accepted")
    except ValueError:
        pass

    # Container sizes, the numbers the plan is costed on.
    gate = rng.standard_normal((640, 2560), dtype=np.float32)
    up = rng.standard_normal((640, 2560), dtype=np.float32)
    down = rng.standard_normal((2560, 640), dtype=np.float32)
    merged, qs = merge_expert(gate, up, down, 64)
    assert merged.nbytes == 2_457_600, merged.nbytes
    assert qs.size == 76_800 and qs.nbytes == 307_200, (qs.size, qs.nbytes)
    assert merged.dtype == np.uint8 and qs.dtype == np.float32

    # The pair survives a safetensors round-trip with its dtypes intact.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "e0.safetensors")
        save_file({"merged_weight": merged, "qs": qs}, p)
        got = load_file(p)
        assert np.array_equal(got["merged_weight"], merged)
        assert np.array_equal(got["qs"], qs)

    # Name guard: known names classify, unknown names stop the run.
    assert classify("model.language_model.layers.3.mlp.experts.gate_up_proj") == "expert"
    assert classify("mtp.layers.0.mlp.experts.7.down_proj.weight") == "expert"
    assert classify("model.language_model.layers.3.mlp.gate.weight") == "skip"
    assert classify("model.language_model.layers.3.mlp.experts.4.up_proj.weight_scale_inv") == "consumed"
    try:
        classify("model.language_model.layers.3.mlp.experts.brand_new_thing")
        raise AssertionError("unknown name was accepted")
    except KeyError:
        pass

    print(f"selftest OK  (int4 rel err {rel:.4f}, two's-complement misread rel {rel_wrong:.3f}, "
          f"expert {merged.nbytes + qs.nbytes} B = {(merged.nbytes + qs.nbytes)/2**20:.4f} MiB)")


# ------------------------------------------------------------------ fixture
FIXTURE_MAGIC = 0x51333849          # "Q38I"

def emit_fixture(path, O=32, I=128, S=2, gs=GROUP_SIZE, seed=20260914):
    """Write the cross-language conformance fixture.

    The selftest above only proves this file agrees with ITSELF: it quantizes
    and dequantizes through two functions written side by side, so a shared
    misreading of the container spec would pass. What has to hold is that the
    bytes this tool writes are the bytes the ENGINE decodes, and the engine's
    decoder is quant.h's matmul_i4_grouped(), in C.

    So the contract is pinned as data: this emits the packed nibbles, the group
    scales, an input activation and the reference y computed here in float, and
    tests/test_qwen38_int4_container.c replays it through the real kernel. If
    either side ever changes its nibble convention, its scale layout or its
    group stride, the fixture stops matching and the test says so.

    Layout (little-endian): magic,O,I,S,gs as int32, then q4[O*I/2] u8,
    scale[O*I/gs] f32, x[S*I] f32, y[S*O] f32."""
    import struct
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((O, I), dtype=np.float32)
    x = rng.standard_normal((S, I), dtype=np.float32)
    q4, scale = quant_int4_grouped(w, gs)
    # Reference y from the DEQUANTIZED weights, in float64 to keep the tolerance
    # about the kernel's arithmetic rather than about this line's rounding.
    wq = unpack_int4_grouped(q4, scale, O, I, gs)
    y = (x.astype(np.float64) @ wq.astype(np.float64).T).astype(np.float32)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<5i", FIXTURE_MAGIC, O, I, S, gs))
        fh.write(q4.tobytes())
        fh.write(scale.astype(np.float32).tobytes())
        fh.write(x.tobytes())
        fh.write(y.tobytes())
    print(f"fixture -> {path}  O={O} I={I} S={S} gs={gs}  "
          f"{os.path.getsize(path)} B")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--indir", default=None,
                    help="already-downloaded shards (the FP8 checkpoint, for testing)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="convert only the first N units, one unit = one MoE layer "
                         "= one output shard (--limit-shards 1 gives the layer-0 subset)")
    ap.add_argument("--keep-source", action="store_true",
                    help="do not delete a downloaded shard once no unit still needs it")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-fixture", default=None,
                    help="write the C conformance fixture read by "
                         "tests/test_qwen38_int4_container.c, then exit")
    a = ap.parse_args()

    if a.emit_fixture:
        emit_fixture(a.emit_fixture)
        return
    if a.selftest:
        selftest()
        return
    if not a.outdir:
        ap.error("--outdir is required (or --selftest)")

    os.makedirs(a.outdir, exist_ok=True)
    src_dir = os.path.join(a.outdir, "_src")
    done_path = os.path.join(a.outdir, ".converted.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

    sizes = {}
    if a.indir:
        index_path = os.path.join(a.indir, "model.safetensors.index.json")
        config_path = os.path.join(a.indir, "config.json")
        weight_map = json.load(open(index_path))["weight_map"]
        config = json.load(open(config_path))
        fetch = lambda fn: os.path.join(a.indir, fn)
        where = index_path
    else:
        from huggingface_hub import hf_hub_download
        token = hf_token()
        print(f"HF token: {'present' if token else 'absent (throttled)'}", flush=True)

        def pull(fn):
            return hf_hub_download(a.repo, fn, revision=a.revision, token=token or False,
                                   local_dir=os.path.join(a.outdir, "_meta"))
        weight_map = json.load(open(pull("model.safetensors.index.json")))["weight_map"]
        config = json.load(open(pull("config.json")))
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(a.repo, revision=a.revision, files_metadata=True,
                                      token=token or False)
            sizes = {s.rfilename: s.size for s in info.siblings if s.size}
        except Exception as exc:                             # noqa: BLE001
            raise SystemExit(f"ERROR: file sizes unavailable ({exc}); ranged download "
                             "cannot tell a finished shard from a truncated one")
        fetch = lambda fn: fetch_curl(a.repo, a.revision, fn, src_dir, sizes.get(fn, 0), token)
        where = f"{a.repo}@{a.revision}"

    check_geometry(config, where)
    plan = build_plan(weight_map)                 # also the unknown-name guard
    expect_units = EXPECT["num_hidden_layers"] + sum(1 for u, _, _, _ in plan if u[0] == "mtp")
    if len(plan) != expect_units:
        raise SystemExit(f"ERROR: {where}: found {len(plan)} MoE units, expected {expect_units}")
    total_experts = len(plan) * EXPECT["num_experts"]
    print(f"{where}: {len(plan)} MoE units x {EXPECT['num_experts']} experts "
          f"= {total_experts} experts, gs={a.group_size}", flush=True)

    if a.limit_shards:
        plan = plan[:a.limit_shards]

    # Reference count per source shard, so a shard is deleted the moment no
    # remaining unit needs it -- and never when it came from --indir.
    need = {}
    for _unit, _prefix, shards, _fused in plan:
        for s in shards:
            need[s] = need.get(s, 0) + 1

    t_start = time.time()
    converted = 0
    for i, (unit, prefix, shards, fused) in enumerate(plan, 1):
        out_name = (f"model-{unit[1]:05d}.safetensors" if unit[0] == "layer"
                    else f"model-mtp{unit[1]:02d}.safetensors")
        out_path = os.path.join(a.outdir, out_name)
        if out_name in done and os.path.exists(out_path):
            for s in shards:
                need[s] -= 1
            continue
        guard_space(a.outdir, a.min_free_gb)
        t0 = time.time()
        paths = [fetch(s) for s in sorted(shards)]
        t_dl = time.time() - t0
        t1 = time.time()
        stats = convert_unit(prefix, fused, paths, out_path, a.group_size,
                             EXPECT["num_experts"])
        t_conv = time.time() - t1
        for s in shards:
            need[s] -= 1
            if need[s] == 0 and not a.indir and not a.keep_source:
                p = os.path.join(src_dir, s)
                if os.path.exists(p):
                    os.remove(p)
        done.add(out_name)
        json.dump(sorted(done), open(done_path, "w"))
        converted += 1
        print(f"[{i}/{len(plan)}] {prefix}  dl {t_dl:5.0f}s  conv {t_conv:5.0f}s  "
              f"out {os.path.getsize(out_path)/1e9:5.3f} GB  experts {stats['experts']}  "
              f"zero-proj {stats['zero_projections']}  | free {free_gb(a.outdir):.0f} GB",
              flush=True)

    meta = {
        "family": "qwen38",
        "source_repo": a.repo,
        "source_revision": a.revision if not a.indir else "local",
        "expert_gs": a.group_size,
        "expert_fmt": 4,
        "expert_bits": 4,
        "nibble_encoding": "offset-binary (stored u = q + 8, decoded u - 8)",
        "scale_layout": "[O][I/gs] row-major, gate||up||down concatenated per expert",
        "scale_floor": 1e-8,
        "hidden_size": EXPECT["hidden_size"],
        "moe_intermediate_size": EXPECT["moe_intermediate_size"],
        "num_experts": EXPECT["num_experts"],
        "num_hidden_layers": EXPECT["num_hidden_layers"],
        "moe_units": len(plan),
        "experts_total": len(plan) * EXPECT["num_experts"],
        "bytes_per_expert": 2_764_800,
        "contains": "routed experts only; dense, PLE, vision and tokenizer stay in "
                    "the FP8 checkpoint, mounted alongside via st_init_multi",
    }
    with open(os.path.join(a.outdir, "qwen38_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    total = sum(os.path.getsize(os.path.join(a.outdir, s)) / 1e9
                for s in done if os.path.exists(os.path.join(a.outdir, s)))
    print(f"\ndone: {len(done)} units ({converted} this run), {total:.1f} GB in "
          f"{(time.time()-t_start)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
