#!/usr/bin/env python
"""Figure 2C stage 2: choose the twelve carrier sentences for the steering ladder and write <data>/meta.json.

    python experiments/fig2_timeofday/carrier_screen.py          # ~1 min on a GPU
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from weekday_manifold.model import load_model  # noqa: E402
from weekday_manifold.timeofday import CARRIERS, validate_carriers  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402
from hour_readout import circ_diff  # noqa: E402

TAIL = "at around"


def prompt(carrier, modifier, day, tail=TAIL):
    m = f" {modifier}" if modifier else ""
    return f"{carrier}{m} on {day} {tail}"


def hour_forms(h):
    hh = 12 if h % 12 == 0 else h % 12
    ap = "am" if h < 12 else "pm"
    dotted = "a.m." if h < 12 else "p.m."
    return [f" {hh}{ap}", f" {hh} {ap}", f" {hh} {dotted}"]


def _lse(a):
    m = np.max(a)
    return float(m + np.log(np.exp(a - m).sum()))


def summarise(logp_hour):
    """Circular mean hour and concentration of a 24-bin hour log-probability vector."""
    p = np.exp(logp_hour - _lse(logp_hour))
    ang = 2 * np.pi * np.arange(24) / 24.0
    R = complex((p * np.cos(ang)).sum(), (p * np.sin(ang)).sum())
    cm = (math.atan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi)
    return dict(circ_mean=cm, conc=abs(R), p=p)


class HourScorer:
    """Teacher-forced log-probability of each hour's surface forms after a prompt."""

    def __init__(self, model, prepend_bos=True):
        self.m = model
        self.bos = prepend_bos
        self.dev = model.cfg.device
        forms, owner = [], []
        for h in range(24):
            for f in hour_forms(h):
                forms.append(model.to_tokens(f, prepend_bos=False)[0].tolist())
                owner.append(h)
        self.forms, self.owner = forms, np.array(owner)

    @torch.no_grad()
    def score(self, text):
        """Return (log-prob per hour [24], log-prob per form [n_forms, 24])."""
        pre = self.m.to_tokens(text, prepend_bos=self.bos)[0].tolist()
        seqs = [pre + f for f in self.forms]
        mx = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), mx), dtype=torch.long, device=self.dev)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s, device=self.dev)
        lg = torch.log_softmax(self.m(ids, return_type="logits").float(), -1)
        n_pre = len(pre)
        out = np.full(len(seqs), -np.inf)
        for j, s in enumerate(seqs):
            idx = torch.arange(n_pre, len(s), device=self.dev)
            out[j] = float(lg[j, idx - 1, torch.tensor(s[n_pre:], device=self.dev)].sum())
        per_hour = np.array([_lse(out[self.owner == h]) for h in range(24)])
        return per_hour, out.reshape(24, -1).T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--max-conc", type=float, default=0.55,
                    help="drop carriers whose clean hour distribution is more concentrated "
                         "than this: no headroom to move")
    ap.add_argument("--data-dir", default=data_dir("fig2_timeofday"))
    args = ap.parse_args()

    validate_carriers()
    model = load_model(args.model, device="cuda", fold_ln=True, center_writing_weights=True,
                       center_unembed=True, dtype="bfloat16")
    sc = HourScorer(model)
    print("[screen] cross-form agreement and concentration on the clean Wednesday prompt",
          flush=True)
    screen = []
    for c in CARRIERS:
        lp_all, per_form = sc.score(prompt(c, "", "Wednesday"))
        cms = [summarise(f)["circ_mean"] for f in per_form]
        spread = max(abs(circ_diff(a, b)) for a in cms for b in cms)
        s = summarise(lp_all)
        screen.append(dict(carrier=c, form_spread=float(spread), conc=float(s["conc"]),
                           circ_mean=float(s["circ_mean"])))
        print(f"   {c:<24} form-spread={spread:5.2f}h  conc={s['conc']:.3f}  "
              f"hour={s['circ_mean']:5.2f}", flush=True)
    eligible = sorted((r for r in screen if r["conc"] <= args.max_conc),
                      key=lambda r: r["form_spread"])
    keep = [r["carrier"] for r in eligible[:args.n_carriers]]
    print(f"[screen] {len(eligible)}/{len(screen)} carriers under the concentration cap "
          f"{args.max_conc}; kept {len(keep)}: {keep}")

    meta = dict(model=args.model, carriers_kept=keep, screen=screen, max_conc=args.max_conc,
                tail=TAIL)
    out = os.path.join(args.data_dir, "meta.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
