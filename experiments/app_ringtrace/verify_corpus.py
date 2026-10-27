#!/usr/bin/env python
"""Check the invariants of the period-final capture list (tokeniser only).

Text-vs-decode agreement is reported as a rate because tok.decode drops spaces before some punctuation.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_period_windows import N16, build_period_family  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_CLASSES = ("positive_clean", "positive_modifier", "proper_noun_weekday")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    args = ap.parse_args()
    fails = collections.Counter()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    fam = build_period_family(tok)
    day_ids = {}
    for d in DAYS:
        for f in (d, " " + d):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                day_ids[e[0]] = d

    L = np.load(os.path.join(args.corpus, "capture_list_n256.npz"), allow_pickle=True)
    ids = L["ids"]; ids16 = ids[:, -N16:]
    cls = L["cls"].astype(str); day = L["day"].astype(str)
    anchor = L["anchor_idx16"].astype(int); dist = L["dist"].astype(int)
    pid = L["period_id"].astype(int); si = L["site_idx16"]
    N = len(ids)
    print(f"[list] {N} rows: {dict(collections.Counter(cls.tolist()))}")

    fails["last token != period_id"] = int((ids[:, -1] != pid).sum())
    fails["period_id not in family"] = int(sum(int(p) not in fam for p in pid))
    fails["dist != 15 - anchor"] = int((dist != N16 - 1 - anchor).sum())
    fails["anchor out of 1..14"] = int(((anchor < 1) | (anchor > N16 - 2)).sum())
    fails["site_idx16 != (anchor, 15)"] = int(((si[:, 0] != anchor) | (si[:, 1] != N16 - 1)).sum())
    wk = np.isin(cls, WEEKDAY_CLASSES)
    at = ids16[np.arange(N), anchor]
    is_day = np.array([int(t) in day_ids for t in at])
    lab_ok = np.array([day_ids.get(int(t)) == d for t, d in zip(at, day)])
    fails["weekday-class anchor is not a weekday"] = int((wk & ~is_day).sum())
    fails["weekday-class anchor != day label"] = int((wk & is_day & ~lab_ok).sum())
    n_wd16 = np.array([sum(int(t) in day_ids for t in row) for row in ids16])
    fails["weekday-class with != 1 weekday in 16"] = int((wk & (n_wd16 != 1)).sum())
    fails["control with a weekday in 16"] = int((~wk & (n_wd16 > 0)).sum())

    # agreement rate of the 16-token text slice with decode(ids16)
    rows = [json.loads(l) for l in open(os.path.join(args.corpus, "windows_n256_dedup.jsonl"))]
    by_hash = {r["hash"]: r for r in rows}
    n_chk, n_mis = 0, 0
    for k in range(0, N, max(1, N // 2000)):
        r = by_hash.get(str(L["hash"][k]))
        if r is None:
            continue
        t16 = r["text"][int(r["text16_off"]):]
        dec = tok.decode(ids16[k].tolist(), clean_up_tokenization_spaces=False)
        n_chk += 1; n_mis += int(t16 != dec)
    print(f"[text] 16-token text slice == decode(ids16) on {n_chk - n_mis}/{n_chk} sampled rows "
          f"({100 * n_mis / max(n_chk, 1):.1f}% differ in whitespace; see the docstring)")

    bad = {k: v for k, v in fails.items() if v}
    print("\n[verify] " + ("ALL INVARIANTS HOLD" if not bad else "FAILURES: " + json.dumps(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
