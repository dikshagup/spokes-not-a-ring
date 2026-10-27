#!/usr/bin/env python
"""Capture the two-day uncertainty prompts at the final period token (Figure 3 and appendix).

    PYTHONPATH=src:experiments/common:experiments/fig3_uncertainty \\
        python experiments/fig3_uncertainty/capture_prompts.py --family frequency   # GPU, ~3 min
        python experiments/fig3_uncertainty/capture_prompts.py --family hedges      # GPU, ~6 min
"""
from __future__ import annotations

import argparse
import collections

from weekday_manifold.manifold.probes import build_frequency_pairs, build_hedge_families
from ring_coords import LAYER, capture_path, save_capture


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("frequency", "hedges"), required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    specs = build_frequency_pairs() if args.family == "frequency" else build_hedge_families()
    by_fam = collections.Counter(s.meta["family_name"] for s in specs)
    print(f"[{args.family}] {len(specs)} prompts: " +
          ", ".join(f"{k} {n}" for k, n in sorted(by_fam.items())), flush=True)

    from model_capture import MODEL, capture_final_token, load_llama
    model, cfg = load_llama(args.layer)
    acts = capture_final_token(model, [s.text for s in specs], args.layer, cfg.prepend_bos)
    out = capture_path(args.family, args.layer)
    save_capture(out, acts, specs, args.layer, MODEL)
    print(f"[{args.family}] -> {out}")


if __name__ == "__main__":
    main()
