#!/usr/bin/env python
"""Figure 2 stage 1: capture the residual stream at the weekday token for the clock, phrase and placebo prompts.

    python experiments/fig2_timeofday/capture_prompts.py          # ~1 min on a GPU
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from weekday_manifold.manifold.days import DAYS  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402

# midnight is a separate family and is not plotted
CLOCK_HOURS = tuple(range(1, 24))
MIDNIGHT_HOUR = 0
WORDS = [("early morning", 7.0), ("late morning", 10.0), ("mid afternoon", 14.0),
         ("early evening", 18.0), ("late evening", 21.0), ("late night", 23.0)]
# each modifier is two Llama tokens, so the weekday is at the same position in every prompt
PLACEBOS_VERY = ["very rainy", "very quiet", "very sunny", "very windy", "very crowded",
                 "very noisy", "very muddy", "very chilly", "very humid", "very cloudy"]
PLACEBOS_VARIED = ["quite rainy", "rather quiet", "fairly sunny", "pretty windy",
                   "somewhat crowded", "oddly noisy", "unusually muddy", "mildly chilly",
                   "slightly humid", "terribly cloudy", "strangely damp", "notably dusty"]


def clock_word(h: int) -> str:
    """Spoken 12-hour phrase for a 24-hour index, such as "seven pm"."""
    names = ["twelve", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven"]
    return f"{names[h % 12]} {'am' if h < 12 else 'pm'}"


def build_prompts():
    """Return (text, day index, family, label, nominal hour) for every modifier and weekday."""
    mods = [(clock_word(h), "clock", f"{h:02d}:00", float(h)) for h in CLOCK_HOURS]
    mods += [(clock_word(MIDNIGHT_HOUR), "clock_midnight", f"{MIDNIGHT_HOUR:02d}:00",
              float(MIDNIGHT_HOUR))]
    mods += [(w, "word", w, hr) for w, hr in WORDS]
    mods += [(p, "placebo_very", p, np.nan) for p in PLACEBOS_VERY]
    mods += [(p, "placebo_varied", p, np.nan) for p in PLACEBOS_VARIED]
    rows = []
    for phrase, fam, label, hour in mods:
        for d, day in enumerate(DAYS):
            rows.append((f"It was {phrase} on {day}.", d, fam, label, hour))
    return rows


def capture(model_name, layer, device, chunk):
    """Residual stream after `layer` at the weekday token for every prompt. Returns (acts, rows, position, n_tokens)."""
    import torch
    from weekday_manifold.model import load_model

    rows = build_prompts()
    texts = [r[0] for r in rows]
    model = load_model(model_name, device=device, fold_ln=True, center_writing_weights=True,
                       center_unembed=True, dtype="bfloat16" if device == "cuda" else "float32")
    toks = [model.to_tokens(t, prepend_bos=True) for t in texts]
    # the weekday token must be at the same position in every prompt
    day_id = [model.to_tokens(" " + DAYS[r[1]], prepend_bos=False)[0, 0].item() for r in rows]
    pos = [int((toks[i][0] == day_id[i]).nonzero()[0, 0]) for i in range(len(rows))]
    if len(set(pos)) != 1:
        by_pos = {}
        for i, p in enumerate(pos):
            by_pos.setdefault(p, set()).add(rows[i][3])
        raise SystemExit("weekday token is not at one position: "
                         + "; ".join(f"pos {p}: {sorted(v)[:6]}" for p, v in by_pos.items()))
    P = pos[0]
    n_tok = sorted({int(t.shape[1]) for t in toks})[0]
    print(f"[capture] {len(rows)} prompts, weekday token at position {P}, layer {layer}",
          flush=True)
    out = []
    with torch.no_grad():
        for c0 in range(0, len(toks), chunk):
            batch = torch.cat(toks[c0:c0 + chunk], 0)
            store = {}
            model.run_with_hooks(
                batch, return_type=None,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post",
                            lambda a, hook: store.setdefault("x", a.detach().float().cpu()))])
            out.append(store["x"][:, P, :].numpy())
    return np.concatenate(out, 0), rows, P, n_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--data-dir", default=data_dir("fig2_timeofday"))
    args = ap.parse_args()

    A, rows, P, n_tok = capture(args.model, args.layer, args.device, args.chunk)
    out = os.path.join(args.data_dir, f"timeofday_L{args.layer}.npz")
    np.savez(out, A=A.astype(np.float32),
             day=np.array([r[1] for r in rows]), fam=np.array([r[2] for r in rows]),
             lab=np.array([r[3] for r in rows]),
             hour=np.array([r[4] for r in rows], dtype=float),
             layer=args.layer, pos=P, n_tokens=n_tok, texts=np.array([r[0] for r in rows]),
             model=args.model)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
