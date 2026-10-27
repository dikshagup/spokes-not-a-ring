#!/usr/bin/env python
"""Figure 4, stage 4: capture the layer-28 residual of every window at its capture token (GPU).

Reads the final 16 tokens (BOS prepended); stores the residual after layer 28 at the last token.

    PYTHONPATH=src:experiments/common:experiments/fig4_occupancy \\
        python experiments/fig4_occupancy/capture_layer.py   # ~10 min on one GPU
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np

from weekday_manifold.corpus.capture import capture_final_token, load_model_for_capture
from weekday_manifold.corpus.split import fit_mask
from weekday_manifold.utils import data_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=data_dir("fig4_occupancy"))
    ap.add_argument("--n-max", type=int, default=256, help="length the list was built at")
    ap.add_argument("--n", type=int, default=16, help="tokens the model reads: ids[:, -n:]")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0, help="seed of the fit/held-out split")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N sampled windows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"), allow_pickle=True)
    ids = blob["ids"][:, -args.n:]
    cls = np.asarray(blob["cls"]).astype(str)
    day = np.asarray(blob["day"]).astype(str)
    doc = np.asarray(blob["doc"]).astype(str)
    is_fit = fit_mask(cls, doc, seed=args.seed)
    sel = np.arange(len(ids))
    if args.limit and args.limit < len(ids):
        # random sample, since the list is grouped by class
        sel = np.sort(np.random.default_rng(args.seed).choice(len(ids), args.limit, replace=False))
    ids, cls, day, doc, is_fit = (a[sel] for a in (ids, cls, day, doc, is_fit))
    print(f"[list] {len(ids)} windows at n={args.n}: {dict(collections.Counter(cls.tolist()))}; "
          f"fit split {int(is_fit.sum())} windows", flush=True)

    print(f"[model] loading {args.model} ({args.dtype}) ...", flush=True)
    model, cfg = load_model_for_capture(args.model, args.dtype)
    t0 = time.time()
    acts = capture_final_token(model, ids, args.layer, args.batch_size, cfg.prepend_bos)

    out = args.out or os.path.join(args.inp, f"raw_L{args.layer}_n{args.n}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, acts=acts, cls=cls, day=day, doc=doc, is_fit=is_fit, row=sel,
             layer=args.layer, n=args.n, site="final_token", model=args.model,
             dtype=args.dtype, prepend_bos=cfg.prepend_bos, d_model=acts.shape[1],
             split_seed=args.seed)
    print(f"\n[saved] {out}  acts={acts.shape} {acts.dtype} "
          f"({os.path.getsize(out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
