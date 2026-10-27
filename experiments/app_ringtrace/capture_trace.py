#!/usr/bin/env python
"""Project the period-final corpus onto the weekday-ring plane at every layer and read slot (GPU).

Slot k = 0..14 is the token k places after the weekday (k = 0 is the weekday); slot 15 is the period.

  python experiments/app_ringtrace/capture_trace.py --in <corpus dir>          # writes <corpus dir>/trace_n16.npz
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, _SRC)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FIT_CLASS = "positive_clean"
N_SLOTS = 16
WEEKDAY_CLASSES = ("positive_clean", "positive_modifier", "proper_noun_weekday")
DEFAULT_CAPS = "matched_negative_far=8000,matched_negative=4000,floor=4000,near_miss=2000"


def gather_sites(act, pos):
    """Gather act [B, P, d] at pos [B, S] (long, already shifted for BOS) into [B, S, d]."""
    B = act.shape[0]
    return act[torch.arange(B, device=act.device)[:, None], pos, :]


def split_by_document(meta, fit_frac, seed):
    """Split the positive_clean rows into fit and score sets with no document in both."""
    cls = meta["cls"].astype(str)
    docs = meta["doc"].astype(str)
    pos_mask = cls == FIT_CLASS
    rs = np.random.default_rng(seed)
    pos_docs = np.unique(docs[pos_mask])
    rs.shuffle(pos_docs)
    fit_docs = set(pos_docs[:int(round(fit_frac * len(pos_docs)))].tolist())
    fit_doc = np.array([d in fit_docs for d in docs])
    is_fit = pos_mask & fit_doc
    print(f"[split] {FIT_CLASS} {int(pos_mask.sum())} from {len(pos_docs)} docs -> "
          f"fit {int(is_fit.sum())} / score {int((pos_mask & ~is_fit).sum())} "
          f"(document-disjoint; {int(fit_doc.sum())} rows of any class are in fit docs)",
          flush=True)
    return is_fit, fit_doc


def slot_positions(anchor, n):
    """Return [N, 16] read positions min(a+k, n-1) for k < 15, then n-1, and a validity mask."""
    k = np.arange(N_SLOTS - 1)[None, :]
    pos = np.minimum(anchor[:, None] + k, n - 1)
    valid = (anchor[:, None] + k) <= n - 1
    pos = np.concatenate([pos, np.full((len(anchor), 1), n - 1)], 1)
    valid = np.concatenate([valid, np.ones((len(anchor), 1), bool)], 1)
    return pos.astype(np.int64), valid


def select_rows(blob, caps, seed):
    cls = np.asarray(blob["cls"]).astype(str)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(cls):
        ix = np.where(cls == c)[0]
        cap = caps.get(c)
        if cap is not None and cap < len(ix):
            ix = np.sort(rng.choice(ix, cap, replace=False))
        keep.append(ix)
    return np.sort(np.concatenate(keep))


def forward(model, ids16, pos, layers, batch_size, prepend_bos, dev, on_batch, tag):
    grab = {}

    def make_hook(l):
        def hook(act, hook):
            grab[l] = act.detach()
        return hook

    fwd = [(f"blocks.{l}.hook_resid_post", make_hook(l)) for l in layers]
    bos = (torch.tensor([[model.tokenizer.bos_token_id]], device=dev) if prepend_bos
           else None)
    N, t0 = len(ids16), time.time()
    for s in range(0, N, batch_size):
        chunk = torch.tensor(ids16[s:s + batch_size].astype(np.int64), device=dev)
        p = torch.tensor(pos[s:s + batch_size], device=dev)
        if bos is not None:
            chunk = torch.cat([bos.expand(chunk.shape[0], 1), chunk], dim=1)
            p = p + 1
        with torch.no_grad():
            model.run_with_hooks(chunk, return_type=None, fwd_hooks=fwd)
        # [B, 16, d] per layer, stacked to [B, L, 16, d]
        on_batch(s, torch.stack([gather_sites(grab[l], p).float() for l in layers], 1))
        grab.clear()
        if s and (s // batch_size) % 50 == 0:
            el = time.time() - t0
            print(f"[{tag}] {s}/{N}  {el:.0f}s  eta {el / s * (N - s):.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dims", type=int, default=6)
    ap.add_argument("--fit-frac", type=float, default=0.34)
    ap.add_argument("--layers", default="all", help="'all' or comma-separated")
    ap.add_argument("--caps", default=DEFAULT_CAPS,
                    help="per-class row caps, cls=N,...; weekday classes are never capped")
    ap.add_argument("--min-per-day", type=int, default=10,
                    help="a cell whose fit split has fewer rows than this for some day "
                         "is written but flagged in cell_ok")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from weekday_manifold.manifold.config import ManifoldConfig
    from weekday_manifold.plateau.model import load_plateau_model

    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.caps.split(",") if kv}
    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"),
                   allow_pickle=True)
    sel = select_rows(blob, caps, args.seed)
    ids16 = blob["ids"][sel][:, -args.n:]
    anchor = np.asarray(blob["anchor_idx16"])[sel].astype(int)
    skip = {"ids", "n_max", "n_model", "model", "seed", "site_idx16", "site_names",
            "read_site"}
    meta = {k: np.asarray(blob[k])[sel] for k in blob.files if k not in skip}
    cls = meta["cls"].astype(str)
    dist = meta["dist"].astype(int)
    assert (dist == args.n - 1 - anchor).all()
    assert (ids16[:, -1] == meta["period_id"].astype(int)).all()
    N = len(sel)
    print(f"[list] {N} rows: {dict(collections.Counter(cls.tolist()))}", flush=True)
    pos, valid = slot_positions(anchor, args.n)

    cfg = ManifoldConfig(model_name=args.model, dtype=args.dtype)
    print(f"[model] loading {args.model} ({args.dtype}) ...", flush=True)
    model = load_plateau_model(cfg)
    cfg = cfg.resolve(model.cfg.n_layers)
    dev = next(model.parameters()).device
    layers = (list(range(model.cfg.n_layers)) if args.layers == "all"
              else sorted({int(x) for x in args.layers.split(",")}))
    L, S, K, d = len(layers), N_SLOTS, args.n_dims, model.cfg.d_model

    # ---- pass 1: day centroids per (layer, slot) from the fit rows ----------------
    is_fit, fit_doc = split_by_document(meta, args.fit_frac, args.seed)
    day_ix = {dd: i for i, dd in enumerate(DAYS)}
    fdays = np.array([day_ix[dd] for dd in meta["day"].astype(str)[is_fit]])
    fvalid = valid[is_fit]
    sums = torch.zeros(7, L, S, d, dtype=torch.float64, device=dev)
    counts = np.zeros((7, S), dtype=np.int64)

    def on_fit(s, h):                                    # h [B, L, S, d]
        B = h.shape[0]
        dd, vv = fdays[s:s + B], torch.tensor(fvalid[s:s + B], device=dev)
        hm = h.double() * vv[:, None, :, None]
        for day in np.unique(dd):
            m = torch.tensor(dd == day, device=dev)
            sums[day] += hm[m].sum(0)
            counts[day] += fvalid[s:s + B][dd == day].sum(0)

    t0 = time.time()
    forward(model, ids16[is_fit], pos[is_fit], layers, args.batch_size, cfg.prepend_bos,
            dev, on_fit, "fit")
    print(f"[fit] {time.time()-t0:.0f}s; fit rows per slot: {counts.sum(0).tolist()}",
          flush=True)
    cell_ok = (counts >= args.min_per_day).all(0)           # [S]
    cnt = torch.tensor(np.maximum(counts, 1), dtype=torch.float64, device=dev)
    day_means = sums / cnt[:, None, :, None]                 # [7, L, S, d]
    mu = sums.sum(0) / cnt.sum(0)[None, :, None]             # [L, S, d]
    basis = torch.zeros(L, S, K, d, dtype=torch.float32, device=dev)
    cents_own = np.zeros((L, S, 7, K), np.float32)
    cents_tr = np.zeros((L, S, 7, K), np.float32)
    for li in range(L):
        for si in range(S):
            M = (day_means[:, li, si] - mu[li, si]).cpu().numpy()
            Vt = np.linalg.svd(M, full_matrices=False)[2][:K]
            basis[li, si] = torch.tensor(Vt, dtype=torch.float32, device=dev)
            cents_own[li, si] = M @ Vt.T
        for si in range(S):
            M0 = (day_means[:, li, si] - mu[li, 0]).cpu().numpy()
            cents_tr[li, si] = M0 @ basis[li, 0].cpu().numpy().T
    mu_f = mu.float()
    del sums, day_means

    # ---- pass 2: project every selected row --------------------------------------
    z_own = np.zeros((N, L, S, K), np.float16); z_tr = np.zeros((N, L, S, K), np.float16)
    n_own = np.zeros((N, L, S), np.float16); n_tr = np.zeros((N, L, S), np.float16)

    def on_all(s, h):                                    # h [B, L, S, d]
        B = h.shape[0]
        c = h - mu_f[None]                               # centred on this (layer, slot) mean
        z_own[s:s + B] = torch.einsum("blsd,lskd->blsk", c, basis).cpu().numpy()
        n_own[s:s + B] = c.norm(dim=-1).cpu().numpy()
        c0 = h - mu_f[None, :, :1]                       # centred on the weekday-slot mean
        z_tr[s:s + B] = torch.einsum("blsd,lkd->blsk", c0, basis[:, 0]).cpu().numpy()
        n_tr[s:s + B] = c0.norm(dim=-1).cpu().numpy()

    t0 = time.time()
    forward(model, ids16, pos, layers, args.batch_size, cfg.prepend_bos, dev, on_all,
            "trace")
    out = args.out or os.path.join(args.inp, f"trace_n{args.n}.npz")
    np.savez_compressed(out, z_own=z_own, z_tr=z_tr, norm_own=n_own, norm_tr=n_tr,
                        valid=valid, slot_pos=pos.astype(np.int8), cell_ok=cell_ok,
                        fit_counts=counts, cents_own=cents_own, cents_tr=cents_tr,
                        is_fit=is_fit, fit_doc=fit_doc, layers=np.array(layers),
                        n=args.n, n_dims=K, model=args.model, prepend_bos=cfg.prepend_bos,
                        row=sel, caps=args.caps, **meta)
    print(f"\n[saved] {out}  z_own={z_own.shape} ({os.path.getsize(out)/1e6:.0f} MB) "
          f"in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
