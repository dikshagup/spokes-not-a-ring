#!/usr/bin/env python
"""Fit and save the reference ring (ruler) used by the Figure 3 plates.

    PYTHONPATH=src:experiments/common:experiments/fig3_uncertainty \\
        python experiments/fig3_uncertainty/fit_ruler.py              # GPU, ~5 min
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from weekday_manifold.manifold.days import N_DAYS
from ring_coords import (LAYER, RULER_K, RULER_Q, corpus_design, corpus_sha1, fit_ruler,
                         ruler_path)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = [os.path.join(HERE, "corpus", f) for f in
           ("register_balanced.jsonl.gz", "mixed_register.jsonl.gz")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    bases, texts = corpus_design(CORPORA)
    print(f"[ruler] {len(bases)} templates x {N_DAYS} days = {len(texts)} prompts", flush=True)

    from model_capture import MODEL, capture_final_token, load_llama
    model, cfg = load_llama(args.layer)
    acts = capture_final_token(model, texts, args.layer, cfg.prepend_bos)
    X = acts.astype(np.float64).reshape(len(bases), N_DAYS, -1)

    fit = fit_ruler(X, RULER_K, RULER_Q)
    out = ruler_path(args.layer)
    np.savez(out, mu=fit["mu"], W=fit["W"], M=fit["M"], layer=args.layer, model=MODEL,
             k=RULER_K, q=RULER_Q, n_bases=len(bases), corpus_sha1=corpus_sha1(texts),
             in_sample_acc=fit["in_sample_acc"])
    print(f"[ruler] nearest-centroid day accuracy on the fit set {fit['in_sample_acc']:.3f}")
    print(f"[ruler] -> {out}")


if __name__ == "__main__":
    main()
