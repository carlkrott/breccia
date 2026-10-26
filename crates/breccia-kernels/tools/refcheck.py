#!/usr/bin/env python3
"""Cross-check this crate's CPU decode against the verbatim ggml reference.

The oracle is the *actual* `ggml/src/ggml-quants.c` of the reference clone, not a
re-typed copy: `tools/refcheck_drv.c` includes `ggml-quants.h` and calls
`dequantize_row_iq2_xxs` / `..._iq2_xs` / `..._iq3_xxs` / `..._iq3_s` /
`..._iq4_nl` / `..._q5_K` / `..._q2_0` directly. Both sides read the same random
block bytes and must produce the same fp32 stream.

    python3 tools/refcheck.py [--blocks 2000] [--c-driver PATH]

Requires the reference clone (docs/reference-engines.md). Prints one line per
type: `OK <type> <n> blocks, max |diff| = 0` or the first mismatch.

This is the evidence behind "the reference port is faithful" in
docs/phase-3/01-dequant-gemv.md §3. It is a tool, not a test: CI does not have
`/srv/strata-data` or the reference clone.
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRATE))
REF = os.path.join(REPO, "Reference Projects", "llama.cpp", "ggml", "src")

# (our type name, ggml id, block bytes, block elements, scale field offsets)
TYPES = [
    ("IQ2_XXS", 16, 66, 256, [0]),
    ("IQ2_XS", 17, 74, 256, [0]),
    ("IQ3_XXS", 18, 98, 256, [0]),
    ("IQ3_S", 21, 110, 256, [0]),
    ("IQ4_NL", 20, 18, 32, [0]),
    ("Q5_K", 13, 176, 256, [0, 2]),
    ("Q2_0", 42, 18, 64, [0]),
]


def make_block(rng, bb, scale_offsets, kind):
    blk = bytearray(rng.getrandbits(8) for _ in range(bb))
    for off in scale_offsets:
        r = rng.random()
        if kind == "finite":
            # the range real artefacts use: 2^-11 .. 2^0
            v = rng.uniform(5e-4, 1.0)
        elif kind == "zero":
            v = 0.0
        elif kind == "subnormal":
            v = 2.0 ** -24 * rng.randint(1, 1023)
        elif kind == "huge":
            v = 65504.0 if r % 2 else -65504.0
        else:  # inf/nan
            v = [float("inf"), float("-inf"), float("nan"), 1.0][rng.randint(0, 3)]
        blk[off : off + 2] = struct.pack("<e", v)
    return bytes(blk)


def load_f32(path):
    data = open(path, "rb").read()
    return struct.unpack("<%df" % (len(data) // 4), data)


def same(a, b):
    """Bit-identical, with any-NaN == any-NaN (payload/quiet bit may differ)."""
    if a != a and b != b:
        return True
    if a != a or b != b:
        return False
    if a in (float("inf"), float("-inf")):
        return a == b
    return struct.pack("<f", a) == struct.pack("<f", b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--c-driver", default=None, help="prebuilt refcheck driver")
    args = ap.parse_args()

    drv = args.c_driver
    tmp = tempfile.mkdtemp(prefix="breccia-refcheck-")
    if not drv:
        drv = os.path.join(tmp, "refcheck_drv")
        src_drv = os.path.join(HERE, "refcheck_drv.c")
        cmd = [
            "gcc", "-O2", "-ffp-contract=off",
            "-I" + REF, "-I" + os.path.join(REF, "..", "include"),
            src_drv, os.path.join(REF, "ggml-quants.c"), "-o", drv, "-lm",
        ]
        print("building the reference driver:", " ".join(cmd[:3]), "...")
        subprocess.run(cmd, check=True)

    # our binary (release: the reference is not the hot path but 2000 blocks x 7 types is slow in debug)
    subprocess.run(
        ["cargo", "build", "--release", "--example", "dequant_dump"],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rust_bin = os.path.join(REPO, "target", "release", "examples", "dequant_dump")
    if not os.path.exists(rust_bin):
        sys.exit(f"missing {rust_bin}")

    rng = random.Random(args.seed)
    kinds = ["finite"] * 8 + ["zero", "subnormal", "huge", "nonfinite"]
    failures = 0
    for name, gid, bb, ne, soff in TYPES:
        kinds_for_this = kinds[:]
        blocks = b"".join(
            make_block(rng, bb, soff, kinds_for_this[i % len(kinds_for_this)])
            for i in range(args.blocks)
        )
        blk_path = os.path.join(tmp, f"{name}.blocks")
        open(blk_path, "wb").write(blocks)

        with open(blk_path, "rb") as f:
            c_out = subprocess.run(
                [drv, str(gid), str(bb), str(args.blocks)],
                stdin=f, capture_output=True, check=True,
            ).stdout
        with open(blk_path, "rb") as f:
            r_out = subprocess.run(
                [rust_bin, name, str(args.blocks)],
                stdin=f, capture_output=True, check=True,
            ).stdout
        if len(c_out) != len(r_out):
            print(f"MISMATCH {name}: length {len(c_out)} vs {len(r_out)}")
            failures += 1
            continue
        open(os.path.join(tmp, f"{name}.c.bin"), "wb").write(c_out)
        open(os.path.join(tmp, f"{name}.rust.bin"), "wb").write(r_out)
        a = load_f32(os.path.join(tmp, f"{name}.c.bin"))
        b = load_f32(os.path.join(tmp, f"{name}.rust.bin"))
        bad = 0
        first = None
        for i, (x, y) in enumerate(zip(a, b)):
            if not same(x, y):
                bad += 1
                if first is None:
                    first = (i, x, y)
        if bad:
            failures += 1
            i, x, y = first
            print(f"MISMATCH {name}: {bad}/{len(a)} values differ, first at {i}: c={x!r} rust={y!r}")
        else:
            print(f"OK {name}: {args.blocks} blocks / {len(a)} values bit-identical to ggml-quants.c")
    if failures:
        sys.exit(f"{failures} type(s) disagreed")
    print(f"all {len(TYPES)} types bit-identical (artifacts in {tmp})")


if __name__ == "__main__":
    main()
