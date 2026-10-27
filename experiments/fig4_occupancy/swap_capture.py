#!/usr/bin/env python
"""Figure 4, stage 5b: capture the matched-context set at the stage 4 layer and settings (GPU).

    PYTHONPATH=src:experiments/common:experiments/fig4_occupancy \\
        python experiments/fig4_occupancy/swap_capture.py   # ~20 min on one GPU
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np

from weekday_manifold.corpus.capture import capture_final_token, load_model_for_capture
from weekday_manifold.utils import data_dir


def check_self_variants(acts, b, raw_path, n_check=5000, seed=0):
    """Compare each family's unchanged variant with its raw row and print the relative difference."""
    if not os.path.exists(raw_path):
        print(f"[check] no {raw_path}; self-variant check skipped")
        return
    R = np.load(raw_path, allow_pickle=True)
    pos = {int(r): i for i, r in enumerate(np.asarray(R["row"]))}
    idx = np.where(b["is_self"].astype(bool))[0]
    fam_row = np.repeat(b["fam_row"], 7)[idx]
    if len(idx) > n_check:
        idx_pick = np.sort(np.random.default_rng(seed).choice(len(idx), n_check, replace=False))
        idx, fam_row = idx[idx_pick], fam_row[idx_pick]
    a = acts[idx].astype(np.float32)
    r = R["acts"][[pos[int(x)] for x in fam_row]].astype(np.float32)
    rel = np.linalg.norm(a - r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-6)
    ident = float((np.abs(a - r).max(1) == 0).mean())
    # bf16 batch noise is about 1e-2 relative; a misaligned gather differs by about 1
    ok = float(rel.max()) < 0.1
    print(f"[check] self-variant vs raw on {len(idx)} families: identical {100 * ident:.1f}%, "
          f"worst relative difference {rel.max():.2e}" + ("" if ok else "  <-- MISALIGNED"))
    if not ok:
        raise SystemExit("swap set and raw capture do not line up row for row")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=data_dir("fig4_occupancy"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit-families", type=int, default=None,
                    help="smoke test on the first N families (whole families only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b = np.load(os.path.join(args.inp, f"swap_set_n{args.n}.npz"), allow_pickle=True)
    ids, fam = b["ids"], b["fam"]
    if args.limit_families:
        keep = fam < args.limit_families
        ids, fam = ids[keep], fam[keep]
    N, V = len(ids), 7
    if N % V:
        raise SystemExit(f"{N} rows is not a whole number of {V}-variant families")
    print(f"[set] {N} prompts = {N // V} families x {V}; classes "
          f"{dict(collections.Counter(np.asarray(b['fam_cls']).astype(str).tolist()))}", flush=True)

    print(f"[model] loading {args.model} ({args.dtype}) ...", flush=True)
    model, cfg = load_model_for_capture(args.model, args.dtype)
    t0 = time.time()
    acts = capture_final_token(model, ids, args.layer, args.batch_size, cfg.prepend_bos)

    carry = {k: b[k] for k in b.files if k != "ids"}
    if args.limit_families:
        F = args.limit_families
        for k in list(carry):
            if k.startswith("fam_") and np.ndim(carry[k]) and np.shape(carry[k])[0] == int(b["fam"].max()) + 1:
                carry[k] = carry[k][:F]
        carry["fam"], carry["swap"], carry["is_self"] = fam, b["swap"][:N], b["is_self"][:N]
    out = args.out or os.path.join(args.inp, f"swap_L{args.layer}_n{args.n}.npz")
    np.savez(out, acts=acts, layer=args.layer, site="final_token", dtype=args.dtype,
             prepend_bos=cfg.prepend_bos, d_model=acts.shape[1], **carry)
    print(f"\n[saved] {out}  acts={acts.shape} ({os.path.getsize(out) / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s")
    check_self_variants(acts, carry, os.path.join(args.inp, f"raw_L{args.layer}_n{args.n}.npz"))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
