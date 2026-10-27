#!/usr/bin/env python
"""Build capture_list_n256.npz from the deduplicated windows, dropping rows that fail the token-id checks."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_period_windows import N16, MAX_DIST, build_period_family  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_CLASSES = ("positive_clean", "positive_modifier", "proper_noun_weekday")

CAPS = {"positive_clean": None, "positive_modifier": None, "proper_noun_weekday": None,
        "near_miss": 18000, "matched_negative": 68000, "matched_negative_far": 20000,
        "floor": 20000}

META_KEYS = ["cls", "day", "doc", "pos", "url", "dump", "hash", "pn_side", "title_case",
             "period_id", "period_form", "anchor_pos", "anchor_idx16", "dist",
             "role_proxy", "prev_token", "sent_len", "sentence_complete_in_16",
             "text16_off",
             "mod_left_weekday", "mod_left_month", "mod_left_shifting", "mod_left_coref",
             "mod_left_clock", "mod_left_ordinal", "mod_left_year", "mod_left_timeofday",
             "mod_right_weekday", "mod_right_month", "mod_right_shifting",
             "mod_right_coref", "mod_right_clock", "mod_right_ordinal",
             "mod_right_year", "mod_right_timeofday",
             "l255_weekday", "l255_month", "l255_shifting", "l255_coref", "l255_clock",
             "l255_ordinal", "l255_year"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    day_ids = {}
    for d in DAYS:
        for f in (d, " " + d):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                day_ids[e[0]] = d
    fam = build_period_family(tok)

    path = os.path.join(args.inp, f"windows_n{args.n_max}_dedup.jsonl")
    rows = [json.loads(l) for l in open(path)]
    idpath = os.path.join(args.inp, f"windows_n{args.n_max}_dedup_ids.npy")
    if not os.path.exists(idpath):
        raise SystemExit(f"missing {idpath}. Token ids are the authoritative window "
                         f"definition and must come from selection, not from "
                         f"re-tokenising text. Re-run select_period_windows.py.")
    all_ids = np.load(idpath)
    if len(all_ids) != len(rows):
        raise SystemExit(f"id/row mismatch: {len(all_ids)} vs {len(rows)}")
    for r, i in zip(rows, all_ids):
        r["_ids"] = i.tolist()
    print(f"[load] {len(rows)} deduplicated windows + ids {all_ids.shape}")

    rng = random.Random(args.seed)
    by_cls = collections.defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    chosen = []
    for cls, rs in sorted(by_cls.items()):
        cap = CAPS.get(cls)
        if cap is not None and len(rs) > cap:
            rs = rng.sample(rs, cap)
        chosen.extend(rs)
        print(f"  {cls:<24}{len(by_cls[cls]):>8} -> {len(rs):<8}"
              f"{'(capped)' if cap and len(by_cls[cls]) > cap else ''}")

    ids = np.zeros((len(chosen), args.n_max), dtype=np.int32)
    bad = collections.Counter()
    keep = []
    off16 = args.n_max - N16
    for r in chosen:
        enc = r["_ids"]
        if len(enc) != args.n_max:
            bad[f"length {len(enc)} != {args.n_max}"] += 1
            continue
        if enc[-1] != r["period_id"] or enc[-1] not in fam:
            bad["final token is not the recorded period"] += 1
            continue
        a = r["anchor_idx16"]
        if not (1 <= a <= N16 - 2) or r["dist"] != N16 - 1 - a:
            bad["anchor index / dist inconsistent"] += 1
            continue
        if r["cls"] in WEEKDAY_CLASSES:
            t = enc[off16 + a]
            if t not in day_ids:
                bad["anchor token is not a weekday"] += 1
                continue
            if day_ids[t] != r["day"]:
                bad["anchor weekday != stored day label"] += 1
                continue
            if sum(1 for t_ in enc[off16:] if t_ in day_ids) != 1:
                bad["more than one weekday in the 16-token input"] += 1
                continue
        else:
            if any(t_ in day_ids for t_ in enc[off16:]):
                bad["control has a weekday in the 16-token input"] += 1
                continue
        ids[len(keep)] = enc
        keep.append(r)
    ids = ids[:len(keep)]

    print(f"\n[verify] {len(keep)}/{len(chosen)} windows passed the invariants")
    for k, v in bad.items():
        print(f"    dropped {v:>6}  {k}")

    out = args.out or os.path.join(args.inp, f"capture_list_n{args.n_max}.npz")
    cols = {k: np.array([r.get(k) if r.get(k) is not None else "" for r in keep],
                        dtype=object) for k in META_KEYS}
    # the two read sites as indices into the 16-token model input, BOS not counted
    site_idx16 = np.stack([np.array([int(r["anchor_idx16"]) for r in keep]),
                           np.full(len(keep), N16 - 1)], 1).astype(np.int16)
    np.savez_compressed(out, ids=ids, n_max=args.n_max, n_model=N16, model=args.model,
                        seed=args.seed, site_idx16=site_idx16,
                        site_names=np.array(["anchor_token", "period_token"]),
                        read_site="period", **cols)
    counts = collections.Counter(r["cls"] for r in keep)
    print(f"\n[saved] {out}  ids={ids.shape} ({os.path.getsize(out)/1e6:.0f} MB)")
    for c, n in sorted(counts.items()):
        print(f"    {c:<24}{n:>8}")
    with open(os.path.join(args.inp, f"capture_meta_n{args.n_max}.json"), "w") as f:
        json.dump({"counts": dict(counts), "dropped": dict(bad), "n_max": args.n_max,
                   "n_model": N16, "max_dist": MAX_DIST, "model": args.model,
                   "total": len(keep), "read_site": "period"}, f, indent=2)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
