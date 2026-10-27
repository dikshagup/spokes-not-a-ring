"""Draw the confound appendix figure, figures/fig_confound.png (+ .pdf), and print its caption.

  PYTHONPATH=src:experiments/common:experiments/app_confound \\
      python experiments/app_confound/figure_confound.py       # CPU, about 1 min
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from masked_kl import MaskedKL
from paper_style import helvetica
from periodicity import seven_fold
from weekday_manifold.utils import data_dir

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figures", "fig_confound.png")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RED, BLUE = "#CC3311", "#0077BB"
N_DRAWS = 1000          # random token sets for the null band in (A)
TITLES = {"A": "(A)  Weekday tokens track the whole vocabulary",
          "B": "(B)  Masking them leaves the peaks",
          "C": "(C)  Periodicity is spread across the vocabulary",
          "D": "(D)  Day-name tokens peak at the crossing into their own day",
          "E": "(E)  The most periodic tokens peak at every crossing"}


def glyphs(s: str) -> str:
    """Replace each character above U+024F with its code point."""
    return "".join(c if ord(c) <= 0x024F else f"U+{ord(c):04X}" for c in s)


def despine(ax, drop=("top", "right"), keep_colour="0.6"):
    for side in drop:
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom", "right", "top"):
        if side not in drop:
            ax.spines[side].set_color(keep_colour)
    ax.tick_params(colors="0.35", length=3)
    ax.yaxis.label.set_color("0.2"); ax.xaxis.label.set_color("0.2")


def single_token(tok, s: str):
    t = tok.encode(s, add_special_tokens=False)
    return int(t[0]) if len(t) == 1 else None


def analyse(z, tok, n_draws: int = N_DRAWS, seed: int = 0) -> dict:
    u = np.asarray(z["u"], float) * 7
    gain = np.asarray(z["gain"], float)
    L = np.asarray(z["out_logits"], np.float64)
    weekday_ids = set(np.asarray(z["day_ids"], int).tolist())      # every weekday spelling
    Lc = L - L.mean(1, keepdims=True)
    d = Lc - np.roll(Lc, 1, axis=0)                                  # backward difference, closed ring
    V = L.shape[1]
    names = [single_token(tok, " " + n) for n in DAYS]               # ids of ' Monday' to ' Sunday'
    # category ids: capitalised day names, with and without a leading space
    cid = np.array(sorted({i for n in DAYS for i in (single_token(tok, " " + n), single_token(tok, n))
                           if i is not None and i in weekday_ids}), int)
    rng = np.random.default_rng(seed)

    # (A) per-step L2 norm of the logit change, and a null band from random token sets of the same size
    full = np.linalg.norm(d, axis=1); cat = np.linalg.norm(d[:, cid], axis=1)
    keep = np.ones(V, bool); keep[cid] = False; pool = np.where(keep)[0]
    draws = np.stack([np.linalg.norm(d[:, rng.choice(pool, len(cid), replace=False)], axis=1)
                      for _ in range(n_draws)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    # (B) KL per step, unmasked and with the category masked
    mk = MaskedKL(L); kl_all = mk(np.array([], int)); kl_day = mk(cid)
    # (C) total movement and seven-fold share per token; (E) the most periodic non-weekday tokens
    tot = np.abs(d).sum(0); share = seven_fold(np.abs(d)); mov = tot > 0
    is_weekday = np.zeros(V, bool); is_weekday[list(weekday_ids)] = True
    cand = np.argsort(-tot)[:5000]; cand = cand[~is_weekday[cand]]
    # ties are broken by token id
    per7 = cand[np.lexsort((cand, -share[cand]))[:7]]
    # (D) whether each day name takes its largest step at the crossing into its own day (0.6 mod 1)
    own_pos = [(k - 0.4) % 7 for k in range(7)]
    peak_pos = [float(u[int(np.argmax(np.abs(d[:, t])))]) for t in names]
    own_hit = [abs(((p - o + 3.5) % 7) - 3.5) < 0.6 for p, o in zip(peak_pos, own_pos)]
    return dict(u=u, gain=gain, d=d, names=names, cid=cid, full=full, cat=cat, draws=draws, lo=lo, hi=hi,
                kl_all=kl_all, kl_day=kl_day, tot=tot, share=share, mov=mov, per7=per7,
                own_pos=own_pos, peak_pos=peak_pos, own_hit=own_hit)


def render(r: dict, tok):
    """Draw the figure; call it under helvetica.context(style=None), since sizes assume matplotlib defaults."""
    u, d, cid, per7, names, share, tot, mov = (r[k] for k in ("u", "d", "cid", "per7", "names", "share", "tot", "mov"))
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(3, 7, height_ratios=[1.25, 0.72, 0.72], hspace=0.78, wspace=1.8)
    # (A)
    ax = fig.add_subplot(gs[0, :3]); ax_a = ax; ax2 = ax.twinx()
    ax.plot(u, r["full"], color="k", lw=2, label="whole vocabulary")
    ax2.fill_between(u, r["lo"], r["hi"], color=RED, alpha=0.18, lw=0, label=f"random {len(cid)}-token sets, 95%")
    ax2.plot(u, r["cat"], color=RED, lw=2, label=f"weekday tokens ({len(cid)})")
    ax.set_xlim(0, 7); ax.set_xticks(range(8)); ax.set_xlabel("Steered position (days)")
    ax.set_ylabel("|Δ logit| per step"); ax2.set_ylabel("")
    despine(ax, drop=("top", "right")); despine(ax2, drop=("top", "left"), keep_colour=RED)
    ax2.tick_params(axis="y", colors=RED, labelsize=8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", frameon=False)

    # (B)
    ax = fig.add_subplot(gs[0, 3:5]); ax_b = ax
    ax.plot(u, r["kl_all"], color="k", lw=2, label="whole vocabulary")
    ax.plot(u, r["kl_day"], color=RED, lw=2, label=f"weekday tokens masked ({len(cid)})")
    ax.set_xlim(0, 7); ax.set_xticks(range(8)); ax.set_xlabel("Steered position (days)"); ax.set_ylabel("KL per step (nats)")
    ax.legend(fontsize=8, loc="upper left")
    despine(ax)

    # (C)
    ax = fig.add_subplot(gs[0, 5:7]); ax_c = ax
    ax.hexbin(np.log10(tot[mov]), share[mov], gridsize=70, bins="log", cmap="Greys", mincnt=1)
    ax.scatter(np.log10(tot[cid]), share[cid], s=30, color=RED, edgecolor="k", lw=0.3, label=f"weekday tokens ({len(cid)})", zorder=3)
    ax.scatter(np.log10(tot[per7]), share[per7], s=42, color=BLUE, edgecolor="k", lw=0.3, marker="D", label="most periodic non-day (7)", zorder=4)
    ax.axhline(np.median(share[mov]), color="0.45", ls="--", lw=1)
    ax.text(ax.get_xlim()[1] - 0.03, np.median(share[mov]) + 0.015, "vocabulary median", fontsize=7.5, color="0.35", ha="right")
    ax.set_xlabel("Total movement, log$_{10}$ Σ|Δ logit|"); ax.set_ylabel("Seven-fold share"); ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    despine(ax)

    # (D) and (E): one panel per token, each with its own y scale
    row_labels = []
    inner = [gs[1, :].subgridspec(1, 7, wspace=0.18), gs[2, :].subgridspec(1, 7, wspace=0.18)]
    for row, (ids, col) in enumerate([(names, RED), (list(per7), BLUE)]):
        first_ax = None
        for k, tid in enumerate(ids):
            ax = fig.add_subplot(inner[row][0, k])
            if k == 0:
                first_ax = ax
            for dpos in np.arange(7) + 0.6:
                ax.axvline(dpos, color="0.88", lw=0.8, zorder=0)
            ax.plot(u, np.abs(d[:, tid]), color=col, lw=1.6, zorder=3)
            ax.set_xlim(0, 7); ax.set_xticks(range(0, 8, 2)); ax.set_yticks([])
            despine(ax, drop=("top", "right", "left")); ax.tick_params(axis="x", labelsize=8)
            ax.set_title(glyphs(f"{tok.decode([tid])!r}"), fontsize=9)
            if row == 0:                                             # crossing into this token's own day
                ax.axvline(r["own_pos"][k], color="k", lw=1, ls="--", alpha=0.6)
            else:
                ax.text(0.5, -0.19, f"seven-fold {100 * share[tid]:.0f}%", transform=ax.transAxes, ha="center", fontsize=7.5, color="0.4")
            if k == 0:
                ax.set_ylabel("|Δ logit| per step", fontsize=9)
            if k == 3:
                ax.set_xlabel("Steered position (days)", fontsize=9, labelpad=14)
        row_labels.append((first_ax, first_ax.get_position().y1 + 0.035, TITLES["D" if row == 0 else "E"]))
    fig.canvas.draw(); rend = fig.canvas.get_renderer()

    def left_edge(a):
        return a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).x0
    labels = [(ax_a, ax_a.get_position().y1 + 0.014, TITLES["A"]),
              (ax_b, ax_b.get_position().y1 + 0.014, TITLES["B"]),
              (ax_c, ax_c.get_position().y1 + 0.014, TITLES["C"])] + row_labels
    for a, y, t in labels:
        fig.text(left_edge(a), y, t, fontsize=11, ha="left")
    return fig


def caption(r: dict, tok, n_draws: int) -> str:
    cid, full, cat, draws, kl_all, kl_day, gain = (r[k] for k in ("cid", "full", "cat", "draws", "kl_all", "kl_day", "gain"))
    share, tot, mov, per7, names = (r[k] for k in ("share", "tot", "mov", "per7", "names"))
    r_cat = np.corrcoef(cat, full)[0, 1]
    r_rand = np.mean([np.corrcoef(dr, full)[0, 1] for dr in draws])
    pct = min(100 * (tot[cid].min() > tot[mov]).mean(), 99.9)
    n_hit = sum(r["own_hit"])
    missed = [DAYS[k] for k, h in enumerate(r["own_hit"]) if not h]
    shares_e = 100 * share[per7]
    def crossing_name(p):
        into = int(np.round(p + 0.4)) % 7
        return f"{DAYS[(into - 1) % 7]}\u2013{DAYS[into]}"
    missed_txt = "" if not missed else (
        " " + " and ".join(missed) + " instead take their largest step at the "
        + " and ".join(sorted({crossing_name(r["peak_pos"][DAYS.index(m)]) for m in missed})) + " crossing.")

    return (
        "The sevenfold sensitivity pattern is not driven by weekday-related logits. Llama-3.1-8B, mention family, "
        "the day token steered at layer 2 and read at the model output, on the frames “On {day}, …”. Positions are in day units, "
        "so the seven day centroids sit at the integers and the crossings between neighbouring days fall near 0.6 modulo 1. "
        f"(A) Norm of the per-step change in mean-centred output logits over the whole vocabulary (black, left axis) and over the "
        f"{len(cid)} capitalised day-name tokens (red, right axis), with the 2.5–97.5 percentile band of {n_draws} random "
        f"{len(cid)}-token sets (shaded). The weekday subset tracks the whole vocabulary (r = {r_cat:+.2f}), but so do random sets of "
        f"the same size (r = {r_rand:+.2f}), so tracking on its own does not implicate the day logits. "
        f"(B) Kullback–Leibler divergence between successive steered positions, before and after setting the {len(cid)} day-name "
        f"logits to −∞. Masking keeps {100 * kl_day.mean() / kl_all.mean():.0f}% of the mean divergence and leaves every peak in "
        f"place, although the correlation with the layer-28 tangential gain falls from {np.corrcoef(kl_all, gain)[0, 1]:+.2f} to "
        f"{np.corrcoef(kl_day, gain)[0, 1]:+.2f}. "
        f"(C) Every moving token placed by how far it moves and by the share of its motion explained by a one-day periodic template. "
        f"The day-name tokens (red) move further than {pct:.1f}% of moving tokens, yet their periodicity is ordinary (median "
        f"{np.median(share[cid]):.2f} against {np.median(share[mov]):.2f} for the vocabulary), while the most periodic tokens (blue) "
        f"are unrelated words. "
        f"(D) The seven day-name tokens. The dashed line marks the crossing that arrives at each token’s own day; {n_hit} of the "
        f"seven take their largest step there.{missed_txt} "
        f"(E) The seven most periodic tokens that are not day names. Each steps at all seven crossings, with seven-fold shares of "
        f"{shares_e.min():.0f}–{shares_e.max():.0f}%, far above the day names. Panels (D) and (E) are scaled to their own token; "
        "a token written as U+XXXX is a character for which the figure font has no glyph."
    )


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=os.path.join(data_dir("app_confound"), "steer_ring_output_logits.npz"))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    tok = AutoTokenizer.from_pretrained(meta.get("model", "meta-llama/Llama-3.1-8B"))
    r = analyse(z, tok, n_draws=args.draws)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with helvetica.context(style=None):
        fig = render(r, tok)
        for out in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
            fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
    print("\n--- caption ---\n" + caption(r, tok, args.draws))
    print("\n(A) corr with whole vocabulary: weekday "
          f"{np.corrcoef(r['cat'], r['full'])[0, 1]:+.2f}, random draws "
          f"{np.mean([np.corrcoef(dr, r['full'])[0, 1] for dr in r['draws']]):+.2f}")
    print(f"(B) masking the {len(r['cid'])} weekday tokens keeps {100 * r['kl_day'].mean() / r['kl_all'].mean():.0f}% of the mean KL; "
          f"corr with the gain {np.corrcoef(r['kl_day'], r['gain'])[0, 1]:+.2f} (unmasked {np.corrcoef(r['kl_all'], r['gain'])[0, 1]:+.2f})")
    print(f"(C) seven-fold share: vocabulary median {np.median(r['share'][r['mov']]):.2f}, the {len(r['cid'])} weekday tokens "
          f"{np.median(r['share'][r['cid']]):.2f}, the seven most periodic tokens {np.median(r['share'][r['per7']]):.2f}")
    print("(D) day tokens: " + ", ".join(f"{tok.decode([t])!r} peak {p:.1f}, own crossing {o:.1f}"
                                         for t, p, o in zip(r["names"], r["peak_pos"], r["own_pos"])))
    print("(E) most periodic non-day tokens: " + ", ".join(f"{glyphs(tok.decode([t]))!r} {100 * r['share'][t]:.0f}%" for t in r["per7"]))


if __name__ == "__main__":
    main()
