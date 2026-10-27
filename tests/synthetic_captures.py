"""Synthetic capture files with the same keys as the real captures, for the figure tests.

Only the file layout matches the real captures; the values say nothing about the model.
"""
import json

import numpy as np

from weekday_manifold.manifold.days import DAYS, N_DAYS

MODEL = "meta-llama/Llama-3.1-8B"
FORMULATION = "mention_early"
PATCH_LAYER, READOUT_LAYER = 2, 28
LAYERS = [2, 6, 10, 14, 18, 22, 26, 28, 30, 31]
WORDS = ["one", "two", "three", "four", "five", "six", "seven"]

# day_pos: the weekday-mention token; ro_pos: the readout token (mention: full stop at 8)
FAMILIES = {
    "mention_early": dict(
        ro_pos=8, day_pos=2, sites=["weekday"], readout_group="input",
        tokens=["<|begin_of_text|>", "On", " Monday", " the", " market", " opened",
                " at", " nine", ".", " Traders", " were", " busy"]),
    "interrogative": dict(
        ro_pos=12, day_pos=9, sites=["weekday", "answer"], readout_group="answer",
        tokens=["<|begin_of_text|>", "Q", ":", " What", " day", " is", " one", " days",
                " after", " Monday", "?\n", "A", ":"]),
}
RO_POS, DAY_POS = FAMILIES[FORMULATION]["ro_pos"], FAMILIES[FORMULATION]["day_pos"]


def _ring_centroids(rng, d, radius=1.0, noise=0.05):
    """Seven mean-centred centroids on an irregular loop in the first two dims."""
    ang = np.arange(N_DAYS) / N_DAYS * 2 * np.pi
    C = np.zeros((N_DAYS, d))
    C[:, 0] = np.cos(ang) * (1 + 0.2 * np.sin(3 * ang))
    C[:, 1] = np.sin(ang)
    C *= radius
    C += noise * radius * rng.standard_normal((N_DAYS, d))
    return C - C.mean(0)


def _prompts(formulation, n_prompts):
    """Prompt texts, mentioned days and answer days; figure 6 regex-parses day and offset from the text."""
    src = np.arange(n_prompts) % N_DAYS
    if formulation == "interrogative":
        # mostly offset 1, a few offset 2, which the plate's offset <= 1 filter drops
        off = np.where(np.arange(n_prompts) % 5 == 4, 2, 1)
        texts = [f"Q: What day is {WORDS[o - 1]} days after {DAYS[s]}?\nA:"
                 for s, o in zip(src, off)]
    else:
        off = np.zeros(n_prompts, int)
        texts = [f"On {DAYS[s]} the market opened at nine. Traders were busy" for s in src]
    return texts, src, (src + off) % N_DAYS


def _behaviour(rng, us, shift, kappa=6.0, eps=0.02):
    """Output distribution [n_u, N_DAYS + 1] peaked at the steered day plus ``shift``."""
    th = 2 * np.pi * (np.arange(N_DAYS)[None, :] / N_DAYS - (us[:, None] + shift / N_DAYS))
    w = np.exp(kappa * np.cos(th)) * (1 + 0.05 * rng.random((len(us), N_DAYS)))
    w = w / w.sum(1, keepdims=True) * (1 - eps)
    return np.concatenate([w, np.full((len(us), 1), eps)], 1)


def polar_field(path, n_u=56, n_r=17, n_bg=6, d=64, with_off=False, seed=0,
                readout_layer=READOUT_LAYER, model=MODEL, formulation=FORMULATION):
    """jacobian_field.py's npz; centroids mean-centred as in the real capture."""
    rng = np.random.default_rng(seed)
    fam = FAMILIES[formulation]
    kw = dict(
        centroids=_ring_centroids(rng, d).astype(np.float32), mu=np.zeros(d, np.float32),
        us=np.linspace(0.0, 1.0, n_u, endpoint=False),
        rs=np.linspace(0.0, 1.6, n_r),
        fro=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        gain_t=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        gain_r=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        tokens=np.array(fam["tokens"], dtype=object),
        meta=json.dumps(dict(model=model, formulation=formulation, ro_pos=fam["ro_pos"],
                             day_pos=fam["day_pos"], patch_layer=PATCH_LAYER,
                             readout_layer=readout_layer, n_hutch=6)),
    )
    if with_off:
        kw["gain_off"] = (rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32)
    np.savez(path, **kw)
    return str(path)


def ring_walk(path, n_prompts=14, n_u=29, n_layers=32, seed=0, formulation=FORMULATION,
              model=MODEL, readout_layer=READOUT_LAYER, layers=LAYERS):
    """ring_walk.py's npz; n_u = 29 puts the seven knots on grid points 0, 4, ..., 24.

    The leading array axis is the site; the last steer layer is random, so panel D excludes it.
    """
    rng = np.random.default_rng(seed)
    fam = FAMILIES[formulation]
    sites, n_s, n_l = fam["sites"], len(fam["sites"]), len(layers)
    us = np.linspace(0.0, 1.0, n_u)
    texts, src, ans = _prompts(formulation, n_prompts)
    off = (ans - src) % N_DAYS

    dists = np.zeros((n_s, n_l, n_prompts, n_u, N_DAYS + 1), np.float32)
    for si in range(n_s):
        for li in range(n_l):
            for pi in range(n_prompts):
                if li == n_l - 1:
                    w = rng.random((n_u, N_DAYS + 1)) + 0.5
                    dists[si, li, pi] = w / w.sum(1, keepdims=True)
                else:
                    dists[si, li, pi] = _behaviour(rng, us, off[pi])
    # unpatched distributions answer every prompt correctly, at u = src/7
    d0 = np.stack([_behaviour(rng, np.array([s / N_DAYS]), o)[0] for s, o in zip(src, off)])

    np.savez(
        path,
        us=us, layers=np.array(layers), sites=np.array(sites),
        dists=dists,
        stride_in=rng.random((n_s, n_l, n_u - 1)) + 1.0,
        d_layer=rng.random((n_s, n_l, n_prompts, n_u, n_layers, 2)).astype(np.float32),
        step_layer=rng.random((n_s, n_l, n_prompts, n_u - 1, n_layers, 2)).astype(np.float32),
        resid_scale=(rng.random((n_layers, 2)) + 1.0).astype(np.float32),
        knot_index=np.tile(np.arange(N_DAYS) * ((n_u - 1) // N_DAYS), (n_l, 1)),
        h_identity=np.zeros((n_s, n_l, n_prompts)), w_identity=np.zeros((n_s, n_l, n_prompts)),
        prompts=np.arange(n_prompts), prompt_days=src,
        prompt_texts=np.array(texts),
        day_pos=np.full(n_l, fam["day_pos"]), ro_pos=fam["ro_pos"],
        seq_len=len(fam["tokens"]), readout_layer=readout_layer, d_model=64,
        n_prompts=n_prompts, concept_ids=np.arange(N_DAYS),
        clean_acc=1.0, dist_unpatched=d0.astype(np.float32),
        formulation=formulation, readout_group=fam["readout_group"], model=model,
    )
    return str(path)


def rings(path, d=64, seed=0, formulation="interrogative", model=MODEL,
          readout_layer=READOUT_LAYER, layers=LAYERS):
    """ring_walk.py --dump-rings: the day centroids at every (steer layer, site), keyed "L<layer>@<pos>"."""
    rng = np.random.default_rng(seed)
    fam = FAMILIES[formulation]
    keys = [(L, p) for L in layers for p in sorted({fam["day_pos"], fam["ro_pos"]})]
    depth = np.array([L / max(layers) for L, _ in keys])
    C = np.stack([_ring_centroids(rng, d, radius=1.0 + 3.0 * t) for t in depth])
    radius = np.linalg.norm(C, axis=2)                       # [K, N_DAYS]
    np.savez(
        path,
        layers=np.array(layers), readout_layer=readout_layer, ro_pos=fam["ro_pos"],
        keys=np.array([f"L{L}@{p}" for L, p in keys]),
        which=np.array(["input" if p == fam["day_pos"] else "answer" for _, p in keys]),
        centroids=C.astype(np.float32),
        recovery=0.8 + 0.2 * rng.random(len(keys)), radius=radius,
        centroids_ro=C[-1].astype(np.float32), recovery_ro=0.9, radius_ro=radius[-1],
        model=model,
        resid_norm=20.0 * (1.0 + 4.0 * depth),
    )
    return str(path)
