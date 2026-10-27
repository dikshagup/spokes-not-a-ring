#!/usr/bin/env python
"""Figure 4, stage 1: stream FineWeb and select the windows (CPU, tokeniser only).

Writes windows_n256.jsonl, windows_n256_ids.npy and selection_meta_n256.json.

    HF_HUB_OFFLINE=0 python select_windows.py        # this stage streams the dataset
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

from corpus_filters import (DAYS, build_token_sets, is_capitalised_word,  # noqa: E402
                            proper_noun_trigger, regex_temporal_token_mask, title_case_run,
                            RE_SENT_END)

def scan_document(doc_idx, rec, tok, sets, n_max, rng, per_doc_cap):
    """Yield window dicts of all classes, with all flags, for one document."""
    day_capture = sets["day_capture"]
    day_exclude, month_ids = sets["day_exclude"], sets["month"]
    shifting_ids, coref_ids = sets["shifting"], sets["coref"]
    text = rec["text"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) < n_max:
        return
    pieces = tok.convert_ids_to_tokens(ids)
    pieces = [p.replace("Ġ", " ") for p in pieces]
    rx_marks = regex_temporal_token_mask(text, offsets)

    day_of = {}
    for d, idset in day_capture.items():
        for t in idset:
            day_of[t] = d

    def left_flags(i):
        """Flags over the causal context; ``left_coref`` is recorded but does not demote."""
        f = {"left_weekday": False, "left_month": False, "left_shifting": False,
             "left_coref": False, "left_clock": False, "left_ordinal": False,
             "left_year": False, "left_month_dist": None,
             "left_temporal_dist": None}
        for j in range(i - n_max + 1, i):
            t = ids[j]
            if t in day_exclude:
                f["left_weekday"] = True
            if t in month_ids:
                f["left_month"] = True
                f["left_month_dist"] = min(f["left_month_dist"] or 10**9, i - j)
            if t in shifting_ids:
                f["left_shifting"] = True
            if t in coref_ids:
                f["left_coref"] = True
            for kind in ("clock", "ordinal", "year"):
                if j in rx_marks[kind]:
                    f[f"left_{kind}"] = True
            if (t in day_exclude or t in month_ids or t in shifting_ids
                    or j in rx_marks["clock"] or j in rx_marks["ordinal"]):
                f["left_temporal_dist"] = min(f["left_temporal_dist"] or 10**9, i - j)
        return f

    def right_flags(i):
        hi = min(len(ids), i + 9)
        f = {"right_month": False, "right_weekday": False, "right_shifting": False,
             "right_coref": False, "right_clock": False, "right_ordinal": False,
             "right_year": False}
        for j in range(i + 1, hi):
            if ids[j] in month_ids:
                f["right_month"] = True
            if ids[j] in day_exclude:
                f["right_weekday"] = True
            if ids[j] in shifting_ids:
                f["right_shifting"] = True
            if ids[j] in coref_ids:
                f["right_coref"] = True
            for kind in ("clock", "ordinal", "year"):
                if j in rx_marks[kind]:
                    f[f"right_{kind}"] = True
        return f

    def emit(i, cls, extra=None):
        lo = i - n_max + 1
        w = {"doc": doc_idx, "id": rec.get("id"), "url": rec.get("url"),
             "dump": rec.get("dump"), "pos": i, "cls": cls,
             "capture_piece": pieces[i],
             "window_ids": ids[lo:i + 1],
             "text": text[offsets[lo][0]:offsets[i][1]]}
        w.update(left_flags(i))
        w.update(right_flags(i))
        if extra:
            w.update(extra)
        w["hash"] = hashlib.blake2b(w["text"].encode(), digest_size=12).hexdigest()
        return w

    out = []
    for i in range(n_max - 1, len(ids)):
        if ids[i] not in day_of:
            continue
        side, trig = proper_noun_trigger(pieces, i)
        lf = left_flags(i)
        tc = title_case_run(pieces, i)
        extra = {"day": day_of[ids[i]], "pn_side": side, "pn_trigger": trig,
                 "title_case": tc}
        # a bare year does not demote; it is recorded as would_demote_on_year
        demote = (lf["left_weekday"] or lf["left_month"] or lf["left_shifting"]
                  or lf["left_clock"] or lf["left_ordinal"])
        if side is not None:
            cls = "proper_noun_weekday"
        elif demote:
            cls = "dirty_positive"
        else:
            cls = "positive"                # left_coref, left_year allowed here
        extra["would_demote_on_year"] = bool(not demote and lf["left_year"])
        out.append(emit(i, cls, extra))

    for i in range(n_max - 1, len(ids)):
        if ids[i] not in month_ids and ids[i] not in shifting_ids:
            continue
        if ids[i] in day_exclude or left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "near_miss"))

    cands = [i for i in range(n_max - 1, len(ids))
             if is_capitalised_word(pieces[i]) and ids[i] not in day_exclude
             and ids[i] not in month_ids and ids[i] not in shifting_ids
             and ids[i] not in coref_ids]
    rng.shuffle(cands)
    for i in cands[:per_doc_cap]:
        if left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "matched_negative"))

    cands = list(range(n_max - 1, len(ids)))
    rng.shuffle(cands)
    for i in cands[:per_doc_cap]:
        if left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "floor"))

    by_cls = collections.defaultdict(list)
    for w in out:
        by_cls[w["cls"]].append(w)
    for cls, ws in by_cls.items():
        rng.shuffle(ws)
        for w in ws[:per_doc_cap]:
            yield w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--max-docs", type=int, default=700000)
    ap.add_argument("--n-max", type=int, default=256,
                    help="window length; class membership is decided at this length")
    ap.add_argument("--per-doc-cap", type=int, default=2,
                    help="max windows per class per document")
    ap.add_argument("--target-clean", type=int, default=24000,
                    help="clean weekday windows to collect; the scan stops when this AND "
                         "--target-controls are both met (24,000 took 574,217 documents)")
    ap.add_argument("--target-controls", type=int, default=60000,
                    help="capitalised-word controls to collect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the figure's data directory")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    sets = build_token_sets(tok)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    rng = random.Random(args.seed)

    counts = collections.Counter()
    flagc = collections.Counter()
    seen_hash = set()
    samples = collections.defaultdict(list)
    days = collections.Counter()
    dup = 0
    kept = []
    kept_ids = []

    if args.out is None:
        from weekday_manifold.utils import data_dir
        args.out = data_dir("fig4_occupancy")
    for doc_idx, rec in enumerate(ds):
        if doc_idx >= args.max_docs:
            break
        if (counts["positive"] >= args.target_clean
                and counts["matched_negative"] >= args.target_controls):
            break
        for w in scan_document(doc_idx, rec, tok, sets, args.n_max, rng,
                               args.per_doc_cap):
            if w["hash"] in seen_hash:
                dup += 1
                continue
            seen_hash.add(w["hash"])
            counts[w["cls"]] += 1
            if w["cls"] in ("positive", "dirty_positive"):
                days[w["day"]] += 1
            for k in ("left_month", "left_weekday", "left_shifting", "left_coref",
                      "left_clock", "left_ordinal", "left_year", "right_month",
                      "right_weekday", "right_shifting", "right_coref",
                      "right_clock", "right_ordinal", "right_year",
                      "title_case", "would_demote_on_year"):
                if w.get(k):
                    flagc[f"{w['cls']}:{k}"] += 1
            if w.get("pn_side"):
                flagc[f"{w['cls']}:pn_{w['pn_side']}"] += 1
            if len(samples[w["cls"]]) < 12:
                samples[w["cls"]].append(w)
            # keep the ids; re-tokenising the stored text may not reproduce them
            kept_ids.append(w["window_ids"])
            kept.append({k: v for k, v in w.items() if k != "window_ids"})
        if doc_idx % 2000 == 0 and doc_idx:
            print(f"[scan] {doc_idx} docs | " +
                  " ".join(f"{c}={counts[c]}" for c in sorted(counts)), flush=True)

    print("\n" + "=" * 78)
    print(f"SELECTION SUMMARY  ({doc_idx + 1} docs scanned, N_max={args.n_max})")
    print("=" * 78)
    for c in ["positive", "dirty_positive", "proper_noun_weekday", "near_miss",
              "matched_negative", "floor"]:
        print(f"  {c:<24}{counts[c]:>8}")
    print(f"  {'exact-dup dropped':<24}{dup:>8}")
    print(f"\n  positive day balance: " +
          ", ".join(f"{d[:3]}={days[d]}" for d in DAYS))
    tot_wd = counts["positive"] + counts["dirty_positive"] + counts["proper_noun_weekday"]
    if tot_wd:
        print(f"  clean-positive yield: {counts['positive']}/{tot_wd} = "
              f"{100*counts['positive']/tot_wd:.1f}% of all weekday-final windows")
    print("\n  flags:")
    for k in sorted(flagc):
        print(f"    {k:<44}{flagc[k]:>7}")

    for c, ws in samples.items():
        print(f"\n--- {c} samples (last 60 chars of window, capture token in <>) ---")
        for w in ws[:8]:
            tail = w["text"][-60:].replace("\n", " ")
            print(f"    ...{tail!r}  <{w['capture_piece'].strip()}>")

    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    idpath = os.path.join(args.out, f"windows_n{args.n_max}_ids.npy")
    np.save(idpath, np.asarray(kept_ids, dtype=np.int32))
    print(f"[saved] {idpath}  {np.asarray(kept_ids).shape}")
    path = os.path.join(args.out, f"windows_n{args.n_max}.jsonl")
    with open(path, "w") as f:
        for w in kept:
            f.write(json.dumps(w) + "\n")
    meta = {"dataset": args.dataset, "config": args.config, "model": args.model,
            "n_max": args.n_max, "docs_scanned": doc_idx + 1, "seed": args.seed,
            "per_doc_cap": args.per_doc_cap, "counts": dict(counts),
            "day_balance": dict(days), "dups_dropped": dup, "flags": dict(flagc)}
    with open(os.path.join(args.out, f"selection_meta_n{args.n_max}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] {path}")
    return 0


if __name__ == "__main__":
    rc = main()
    # skip finalization, which the streaming threads otherwise abort with exit 134
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
