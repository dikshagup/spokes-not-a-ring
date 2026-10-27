#!/usr/bin/env python
"""Stream FineWeb and select 256-token windows that end on a sentence-final period, with controls (CPU).

Weekday windows have the weekday 1 to 14 tokens before the period; the model reads the last 16 tokens.

  python experiments/app_ringtrace/select_period_windows.py --out <corpus dir>        # streams FineWeb: run online
Writes windows_n256.jsonl, windows_n256_ids.npy and selection_meta_n256.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from corpus_filters import (  # noqa: E402
    DAYS, build_token_sets, regex_temporal_token_mask, is_capitalised_word,
    proper_noun_trigger, title_case_run)

N16 = 16                                   # tokens the model reads
MAX_DIST = N16 - 2                         # largest weekday-to-period distance

# used for role_proxy only, not for filtering
ADVERBIAL_PREV = {"on", "by", "until", "till", "from", "every", "this", "next", "last",
                  "each", "since", "before", "after", "through", "for"}

TIME_OF_DAY = ["morning", "afternoon", "evening", "night", "noon", "midnight", "midday"]

# a period after one of these words is not a sentence end
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "ltd", "co", "corp",
          "vs", "etc", "no", "vol", "fig", "approx", "dept", "est", "gov", "sen", "rep",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
          "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}

RE_PERIOD_FORM = re.compile(r'^\.["\'”’)\]]*\s*$')


def build_period_family(tok):
    """Return {id: form} for every token that is a period plus optional closing quotes, brackets or whitespace."""
    fam = {}
    for i in range(len(tok)):
        s = tok.decode([i])
        if RE_PERIOD_FORM.match(s):
            fam[i] = s
    return fam


def build_extra_sets(tok):
    def ids_for(forms):
        out = set()
        for f in forms:
            enc = tok.encode(f, add_special_tokens=False)
            if len(enc) == 1:
                out.add(enc[0])
        return out
    words = TIME_OF_DAY + [w.capitalize() for w in TIME_OF_DAY]
    return {"timeofday": ids_for(words + [" " + w for w in words])}


def is_sentence_end(i, ids, pieces, text, offsets, fam):
    if ids[i] not in fam:
        return False
    start, end = offsets[i]
    # character right after the period, which may be inside the same token
    after = text[start + 1:start + 2]
    if after.isalnum():
        return False
    prev = pieces[i - 1].strip().strip(".") if i > 0 else ""
    if prev.lower() in ABBREV or (len(prev) == 1 and prev.isalpha()):
        return False
    nxt = text[end:end + 4].lstrip()
    if nxt and not (nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in "\"'“‘(["):
        return False
    return True


def role_proxy(prev_tok):
    if not prev_tok:
        return "initial"
    return "adverbial" if prev_tok.lower() in ADVERBIAL_PREV else "incidental"


def sentence_start(i, ids, pieces, fam, cap=80):
    """Index of the token that ends the previous sentence, or i-cap-1; the sentence is pieces[start+1 : i]."""
    lo = max(-1, i - cap - 1)
    for j in range(i - 1, lo, -1):
        p = pieces[j]
        if ids[j] in fam or p.strip() in ("!", "?") or "\n" in p or "Ċ" in p:
            return j
    return lo


def scan_document(doc_idx, rec, tok, sets, xsets, fam, n_max, rng, per_doc_cap,
                  terminators="."):
    """Yield the period-final window dicts of every class for one document, then a rejection count."""
    day_capture = sets["day_capture"]
    day_exclude, month_ids = sets["day_exclude"], sets["month"]
    shifting_ids, coref_ids = sets["shifting"], sets["coref"]
    tod_ids = xsets["timeofday"]
    text = rec["text"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) < n_max:
        return
    pieces = [p.replace("Ġ", " ") for p in tok.convert_ids_to_tokens(ids)]
    rx_marks = regex_temporal_token_mask(text, offsets)
    day_of = {t: d for d, idset in day_capture.items() for t in idset}

    def flags_over(rng_, prefix):
        f = {f"{prefix}_{k}": False for k in ("weekday", "month", "shifting", "coref",
                                              "clock", "ordinal", "year", "timeofday")}
        f[f"{prefix}_temporal_dist"] = None
        for j in rng_:
            t = ids[j]
            if t in day_exclude: f[f"{prefix}_weekday"] = True
            if t in month_ids: f[f"{prefix}_month"] = True
            if t in shifting_ids: f[f"{prefix}_shifting"] = True
            if t in coref_ids: f[f"{prefix}_coref"] = True
            if t in tod_ids: f[f"{prefix}_timeofday"] = True
            for kind in ("clock", "ordinal", "year"):
                if j in rx_marks[kind]:
                    f[f"{prefix}_{kind}"] = True
        return f

    def emit(i, w, cls, extra=None):
        """Window dict ending at period i, with the anchor (weekday or control word) at w."""
        lo = i - n_max + 1
        lo16 = i - N16 + 1
        dist = i - w
        rec_ = {"doc": doc_idx, "id": rec.get("id"), "url": rec.get("url"),
                "dump": rec.get("dump"), "pos": i, "cls": cls,
                "capture_piece": pieces[i], "period_id": ids[i], "period_form": fam[ids[i]],
                "anchor_pos": w, "anchor_piece": pieces[w], "dist": dist,
                "anchor_idx16": N16 - 1 - dist,
                "window_ids": ids[lo:i + 1],
                "text": text[offsets[lo][0]:offsets[i][1]],
                "text16_off": offsets[lo16][0] - offsets[lo][0]}
        rec_.update(flags_over(range(lo16, w), "mod_left"))
        rec_.update(flags_over(range(w + 1, i), "mod_right"))
        l255 = flags_over(range(w - n_max + 1, w), "l255")
        rec_.update({k: v for k, v in l255.items() if not k.endswith("timeofday")})
        if extra:
            rec_.update(extra)
        rec_["hash"] = hashlib.blake2b(rec_["text"].encode(), digest_size=12).hexdigest()
        return rec_

    out = []
    rejects = collections.Counter()
    for i in range(n_max - 1, len(ids)):
        if not is_sentence_end(i, ids, pieces, text, offsets, fam):
            continue
        start = sentence_start(i, ids, pieces, fam)
        span = range(start + 1, i)
        sent_len = i - start                          # tokens, including the period
        wd = [j for j in span if ids[j] in day_exclude]

        if wd:
            # ---- sentence with a weekday ------------------------------------------
            if len(wd) > 1:
                rejects["multi_weekday_sentence"] += 1
                continue
            w = wd[0]
            dist = i - w
            if dist > MAX_DIST:
                rejects[f"dist_gt_{MAX_DIST}"] += 1
                rejects[f"dist_hist:{min(dist, 40)}"] += 1
                continue
            if ids[w] not in day_of:                  # a weekday form not in the capture set
                rejects["weekday_not_capture_form"] += 1
                continue
            prev_tok = pieces[w - 1].strip() if w - 1 > start else ""
            role = role_proxy(prev_tok)
            side, trig = proper_noun_trigger(pieces, w)
            extra = {"day": day_of[ids[w]], "pn_side": side, "pn_trigger": trig,
                     "title_case": title_case_run(pieces, w), "role_proxy": role,
                     "prev_token": prev_tok, "sent_len": sent_len,
                     "sentence_complete_in_16": bool(sent_len <= N16)}
            rec_ = emit(i, w, "positive_clean", extra)
            demote = any(rec_[f"mod_{s}_{k}"] for s in ("left", "right")
                         for k in ("weekday", "month", "shifting", "clock", "ordinal",
                                   "timeofday"))
            if side is not None:
                rec_["cls"] = "proper_noun_weekday"
            elif demote:
                rec_["cls"] = "positive_modifier"
            out.append(rec_)
            continue

        # ---- sentence with no weekday: control windows ------------------------
        if any(ids[j] in day_exclude for j in range(i - n_max + 1, i)):
            continue
        base = {"sent_len": sent_len, "sentence_complete_in_16": bool(sent_len <= N16)}
        # one control class per sentence end, since identical windows would be removed in dedup
        if ids[i - 1] in month_ids or ids[i - 1] in shifting_ids:
            out.append(emit(i, i - 1, "near_miss", base))
            continue
        elig = [("floor", i - 1)]
        if is_capitalised_word(pieces[i - 1]) and ids[i - 1] not in coref_ids:
            elig.append(("matched_negative", i - 1))
        far = [d for d in range(2, MAX_DIST + 1) if i - d > start
               and is_capitalised_word(pieces[i - d]) and ids[i - d] not in month_ids
               and ids[i - d] not in shifting_ids and ids[i - d] not in coref_ids]
        if far:
            elig.append(("matched_negative_far", i - rng.choice(far)))
        cls, w = rng.choice(elig)
        out.append(emit(i, w, cls, base))

    by_cls = collections.defaultdict(list)
    for w in out:
        by_cls[w["cls"]].append(w)
    for cls, ws in by_cls.items():
        rng.shuffle(ws)
        for w in ws[:per_doc_cap]:
            w["_rejects"] = None
            yield w
    yield {"_rejects": rejects}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--max-docs", type=int, default=400000)
    ap.add_argument("--n-max", type=int, default=256,
                    help="window length the ids are stored at; the model reads the last 16")
    ap.add_argument("--per-doc-cap", type=int, default=2)
    ap.add_argument("--target-positives", type=int, default=5000,
                    help="stop when this many positive_clean AND --target-negatives are met")
    ap.add_argument("--target-negatives", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="corpus directory")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    sets = build_token_sets(tok)
    xsets = build_extra_sets(tok)
    fam = build_period_family(tok)
    print(f"[period] {len(fam)} period-family tokens: "
          + " ".join(repr(v) for _, v in sorted(fam.items())[:12]), flush=True)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    rng = random.Random(args.seed)

    counts, flagc, rejects = collections.Counter(), collections.Counter(), collections.Counter()
    hist = collections.defaultdict(collections.Counter)
    seen_hash, samples, days = set(), collections.defaultdict(list), collections.Counter()
    dup, kept, kept_ids = 0, [], []

    for doc_idx, rec in enumerate(ds):
        if doc_idx >= args.max_docs:
            break
        if (counts["positive_clean"] >= args.target_positives
                and counts["matched_negative"] >= args.target_negatives):
            break
        for w in scan_document(doc_idx, rec, tok, sets, xsets, fam, args.n_max, rng,
                               args.per_doc_cap):
            if w.get("_rejects") is not None and len(w) == 1:
                rejects.update(w["_rejects"])
                continue
            w.pop("_rejects", None)
            if w["hash"] in seen_hash:
                dup += 1
                continue
            seen_hash.add(w["hash"])
            c = w["cls"]
            counts[c] += 1
            if "day" in w:
                days[w["day"]] += 1
            hist["dist"][f"{c}:{w['dist']}"] += 1
            hist["period_form"][f"{c}:{w['period_form']!r}"] += 1
            if "role_proxy" in w:
                hist["role"][f"{c}:{w['role_proxy']}"] += 1
            hist["sentence_complete"][f"{c}:{w['sentence_complete_in_16']}"] += 1
            for k, v in w.items():
                if (k.startswith("mod_") or k.startswith("l255_")) and v is True:
                    flagc[f"{c}:{k}"] += 1
            if w.get("pn_side"):
                flagc[f"{c}:pn_{w['pn_side']}"] += 1
            if len(samples[c]) < 12:
                samples[c].append(w)
            kept_ids.append(w["window_ids"])
            kept.append({k: v for k, v in w.items() if k != "window_ids"})
        if doc_idx % 2000 == 0 and doc_idx:
            print(f"[scan] {doc_idx} docs | " +
                  " ".join(f"{c}={counts[c]}" for c in sorted(counts)), flush=True)

    print("\n" + "=" * 78)
    print(f"SELECTION SUMMARY  ({doc_idx + 1} docs scanned, N_max={args.n_max}, "
          f"read at the period, weekday within {MAX_DIST} tokens)")
    print("=" * 78)
    for c in sorted(counts):
        print(f"  {c:<26}{counts[c]:>8}")
    print(f"  {'exact-dup dropped':<26}{dup:>8}")
    print("\n  rejects: " + ", ".join(f"{k}={v}" for k, v in sorted(rejects.items())
                                     if not k.startswith("dist_hist")))
    dh = sorted((int(k.split(':')[1]), v) for k, v in rejects.items() if k.startswith("dist_hist"))
    if dh:
        print("  rejected-distance histogram (weekday too far from period): "
              + " ".join(f"{d}:{v}" for d, v in dh))
    print("\n  positive day balance: " + ", ".join(f"{d[:3]}={days[d]}" for d in DAYS))
    for name in ("dist", "role", "period_form", "sentence_complete"):
        print(f"\n  {name}:")
        for k, v in sorted(hist[name].items()):
            print(f"    {k:<44}{v:>7}")
    print("\n  flags:")
    for k, v in sorted(flagc.items()):
        print(f"    {k:<44}{v:>7}")
    for c, ws in sorted(samples.items()):
        print(f"\n--- {c} samples (the 16-token model input) ---")
        for w in ws[:8]:
            t16 = w["text"][w["text16_off"]:].replace("\n", "\\n")
            print(f"    {t16!r}   dist={w['dist']} role={w.get('role_proxy', '-')}")

    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    idpath = os.path.join(args.out, f"windows_n{args.n_max}_ids.npy")
    np.save(idpath, np.asarray(kept_ids, dtype=np.int32))
    print(f"\n[saved] {idpath}  {np.asarray(kept_ids).shape}")
    path = os.path.join(args.out, f"windows_n{args.n_max}.jsonl")
    with open(path, "w") as f:
        for w in kept:
            f.write(json.dumps(w) + "\n")
    meta = {"dataset": args.dataset, "config": args.config, "model": args.model,
            "n_max": args.n_max, "n_model": N16, "max_dist": MAX_DIST,
            "docs_scanned": doc_idx + 1, "seed": args.seed, "per_doc_cap": args.per_doc_cap,
            "read_site": "period", "period_family": {int(k): v for k, v in fam.items()},
            "adverbial_prev": sorted(ADVERBIAL_PREV), "time_of_day": TIME_OF_DAY,
            "counts": dict(counts), "day_balance": dict(days), "dups_dropped": dup,
            "rejects": dict(rejects), "hist": {k: dict(v) for k, v in hist.items()},
            "flags": dict(flagc)}
    with open(os.path.join(args.out, f"selection_meta_n{args.n_max}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
