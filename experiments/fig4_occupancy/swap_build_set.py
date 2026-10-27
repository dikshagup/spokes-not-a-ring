#!/usr/bin/env python
"""Figure 4, stage 5a: build the matched-context set, seven weekday variants of every window.

Writes swap_set_n16.npz (family-major). Tokeniser only.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

from weekday_manifold.corpus.classes import ALL_WEEKDAY_CLASSES
from weekday_manifold.corpus.split import fit_mask
from weekday_manifold.manifold.days import DAYS
from weekday_manifold.utils import data_dir

META = ("pos", "url", "pn_side", "title_case", "left_month", "left_coref",
        "left_year", "right_month", "right_year", "would_demote_on_year")


def weekday_token_map(model_name):
    """{token_id: (day, form)} and its inverse, for the 14 single-token weekday spellings."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    fwd, back = {}, {}
    for d in DAYS:
        for form, s in (("spaced", " " + d), ("bare", d)):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) != 1:
                raise SystemExit(f"{s!r} is not a single token under {model_name}: {t}")
            fwd[t[0]] = (d, form)
            back[(d, form)] = t[0]
    return fwd, back


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=data_dir("fig4_occupancy"))
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--seed", type=int, default=0, help="seed of the fit/held-out split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"), allow_pickle=True)
    ids_all = blob["ids"][:, -args.n:]
    cls = np.asarray(blob["cls"]).astype(str)
    day = np.asarray(blob["day"]).astype(str)
    doc = np.asarray(blob["doc"]).astype(str)
    is_fit = fit_mask(cls, doc, seed=args.seed)

    rows = np.where(np.isin(cls, ALL_WEEKDAY_CLASSES))[0]
    print(f"[list] {len(rows)} weekday-class windows: "
          f"{dict(collections.Counter(cls[rows].tolist()))}", flush=True)

    fwd, back = weekday_token_map(args.model)
    last = ids_all[rows, -1]
    unknown = sorted({int(t) for t in last if int(t) not in fwd})
    if unknown:
        raise SystemExit(f"{len(unknown)} final tokens are not weekday forms: {unknown[:8]}")
    fam_day = np.array([fwd[int(t)][0] for t in last])
    fam_form = np.array([fwd[int(t)][1] for t in last])
    if int((fam_day != day[rows]).sum()):
        raise SystemExit("some windows' final token disagrees with their day label")
    print(f"[forms] {dict(collections.Counter(fam_form.tolist()))}", flush=True)

    F, V = len(rows), 7
    ids = np.repeat(ids_all[rows], V, axis=0)              # [7F, n], family-major
    swap_ix = np.tile(np.arange(V, dtype=np.int8), F)
    ids[:, -1] = [back[(DAYS[v], f)] for f, v in zip(np.repeat(fam_form, V), swap_ix)]
    orig = np.repeat(ids_all[rows], V, axis=0)
    if (ids[:, :-1] != orig[:, :-1]).any():
        raise SystemExit("context columns were modified: the swap is not a last-token edit")
    same = np.repeat(fam_day, V) == np.array(DAYS)[swap_ix]
    if not (ids[same, -1] == orig[same, -1]).all():
        raise SystemExit("self-swap did not reproduce the original token")
    print(f"[swap] {F} families x {V} variants = {len(ids)} prompts", flush=True)

    out = {"ids": ids.astype(np.int32),
           "fam": np.repeat(np.arange(F, dtype=np.int32), V),
           "swap": swap_ix,                                  # index into DAYS
           "is_self": same,
           "days": np.array(DAYS),
           # per family (length F)
           "fam_row": rows.astype(np.int32),                 # row in the capture list / raw file
           "fam_cls": cls[rows], "fam_day": fam_day, "fam_form": fam_form,
           "fam_doc": doc[rows], "fam_is_fit": is_fit[rows],
           "n": args.n, "n_max": args.n_max, "model": args.model, "split_seed": args.seed}
    for k in META:
        out["fam_" + k] = np.asarray(blob[k])[rows]
    path = args.out or os.path.join(args.inp, f"swap_set_n{args.n}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **out)
    print(f"[saved] {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
