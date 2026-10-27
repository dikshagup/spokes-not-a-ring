"""Figure 1, panels A and B, drawn from a ring_walk.py npz."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
GRID = "#e8eaef"
LETTER_X = -0.24
TITLE_X = -0.135
TITLE_Y = 1.13
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def xlab():
    return "Steered position on the ring\n(arc length, fraction of loop)"


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=8.3, length=2.5, width=0.7, colors=MID, pad=2)


def day_marks(ax, xs, day_ids):
    """Draw dotted vertical lines at the day knots, with day names above the axes."""
    for j in range(0, 7):
        if j:
            ax.axvline(xs[j], color=PALE, ls=(0, (1, 2)), lw=0.7, zorder=1)
        ax.text(xs[j], 1.012, DAYS[day_ids[j]], transform=ax.get_xaxis_transform(),
                rotation=0, fontsize=7.0, color=MID, ha="center", va="bottom", zorder=7)


def knot_order(kidx, ref, n):
    """Positions of the day knots after a curve is rolled to start at `ref`, and the day at each."""
    kidx = np.asarray(kidx, int)
    order = [(int(ref) + j) % 7 for j in range(7)]
    pos = np.array([(kidx[d] - kidx[int(ref)]) % n for d in order])
    return pos, order


def by_day(arr, days, kidx, closed=True, roll=True):
    """Roll each prompt to start at its own day, then average over the prompts of each day."""
    a = arr[:, :-1] if closed else arr
    out = []
    for k in range(7):
        rows = [(np.roll(a[i], -int(kidx[k]), axis=0) if roll else a[i])
                for i in range(len(days)) if days[i] == k]
        out.append(np.mean(rows, axis=0) if rows else np.full_like(a[0], np.nan))
    return np.stack(out)


def arc_x(stride, day, kidx):
    """Cumulative arc length from this day's knot, as a fraction of the loop."""
    rolled = np.roll(np.asarray(stride), -int(kidx[int(day)]))
    cum = np.concatenate([[0.0], np.cumsum(rolled)])
    return cum / cum[-1]


def draw_row(fig, axA, axC, npz, fix_in=2, compare=28, readout_depth=None, ref_in=0,
             depth_every=2):
    """Draw panels A and B for steer layer ``fix_in`` (A also shows ``compare``), starting at day ``ref_in``."""
    z = np.load(npz, allow_pickle=True)
    layers = [int(v) for v in z["layers"]]
    us = np.asarray(z["us"])
    q = 1                                                    # readout position
    d_layer = np.asarray(z["d_layer"], np.float32)[..., q]
    step_layer = np.asarray(z["step_layer"], np.float32)[..., q]
    nL = d_layer.shape[-1]
    RO = int(readout_depth if readout_depth is not None else z["readout_layer"])
    assert 0 <= RO < nL, f"readout depth {RO} outside the {nL} recorded depths"
    days = np.asarray(z["prompt_days"], int)
    n_step = len(us) - 1
    KID = np.asarray(z["knot_index"], int)
    ref, fix = int(ref_in), int(fix_in)
    fi = layers.index(fix)

    # ---- A: Hellinger distance on the 8-class simplex vs activation arc length ----------
    kid = KID[fi]
    cum_in = arc_x(np.asarray(z["stride_in"])[0, fi], ref, kid)
    ki, kday = knot_order(kid, ref, n_step)

    def behaviour_curve(layer_idx, kidx):
        """Cumulative Hellinger distance along the walk per prompt, as a fraction of its total."""
        roll = -int(kidx[ref])
        dd = np.sqrt(np.clip(np.asarray(z["dists"])[0, layer_idx], 0, None))
        out = []
        for pi in np.where(days == ref)[0]:
            stepb = np.linalg.norm(np.diff(dd[pi], axis=0), axis=1) / np.sqrt(2.0)
            cb = np.concatenate([[0.0], np.cumsum(np.roll(stepb, roll))])
            if cb[-1] > 0:
                out.append(cb / cb[-1])
        return np.array(out)

    axA.plot([0, 1], [0, 1], color=MID, ls=(0, (3, 3)), lw=0.9, zorder=2)
    for L, col, dark in ((fix, "#2b5d9e", True), (int(compare), "#b3322c", False)):
        if L not in layers:
            continue
        lj = layers.index(L)
        kj = KID[lj]
        cur = behaviour_curve(lj, kj)
        if not len(cur):
            continue
        xj = arc_x(np.asarray(z["stride_in"])[0, lj], ref, kj)
        kij, kdj = knot_order(kj, ref, n_step)
        m = cur.mean(0)
        lo, hi = np.percentile(cur, [10, 90], axis=0)
        axA.fill_between(xj[:len(m)], lo, hi, color=col, alpha=0.16, lw=0, zorder=3)
        axA.plot(xj[:len(m)], m, color=col, lw=1.9, zorder=5, label=f"steered at L{L}")
        if dark:
            axA.scatter(xj[kij], m[kij], s=20, c=[DAY_C[d] for d in kdj], zorder=6,
                        edgecolors="white", linewidths=0.7)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)
    axA.set_xlabel(xlab(), fontsize=8.9, color=INK, linespacing=1.4)
    axA.set_ylabel("Behaviour-space distance travelled\n(fraction of total)",
                   fontsize=8.9, color=INK, linespacing=1.4)
    axA.legend(fontsize=7.4, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
               handlelength=1.4, labelspacing=0.25)
    axA.text(TITLE_X, TITLE_Y, "Non-uniform downstream behaviour", transform=axA.transAxes,
             fontsize=9.5, color=INK, ha="left", va="baseline")
    tidy(axA)
    day_marks(axA, cum_in[ki], kday)

    # ---- B: readout speed around the ring at every depth ----------------------------------
    steps = by_day(step_layer[0, fi], days, kid, closed=False)[ref]     # [U-1, nL]
    st = np.roll(np.asarray(z["stride_in"], float)[0, fi], -int(kid[ref]))

    # a steer at block L reaches the readout position only from block L+1 on
    cum = np.concatenate([[0.0], np.cumsum(steps[:, RO])])
    assert cum[-1] > 0, (
        f"nothing moves at depth {RO} for a steer at L{fix}: a write at block {fix} reaches "
        f"the readout position only from block {fix + 1} on, so depths <= {fix} are "
        f"untouched by construction. Pass --readout-depth greater than {fix}.")

    deep = list(range(fix + 1, nL, depth_every))
    norm = plt.Normalize(vmin=deep[0], vmax=deep[-1])
    cm = plt.get_cmap("viridis")
    spd = steps / np.maximum(st, 1e-12)[:, None]
    for r in deep:
        axC.plot(cum_in[:len(spd)], spd[:, r], lw=1.0, zorder=4,
                 color=cm(0.10 + 0.72 * norm(r)))
    axC.set_xlim(0, 1)
    axC.set_ylim(top=float(np.nanmax(spd[:, deep])) * 1.45)
    axC.set_xlabel(xlab(), fontsize=8.9, color=INK, linespacing=1.4)
    axC.set_ylabel("$\\|\\Delta$ readout resid$\\|$ / $\\|\\Delta$ steer$\\|$\n(raw)",
                   fontsize=8.9, color=INK, linespacing=1.4)
    axC.text(TITLE_X, TITLE_Y, "Speed peaks between the weekdays", transform=axC.transAxes,
             fontsize=9.5, color=INK, ha="left", va="baseline")
    tidy(axC)
    day_marks(axC, cum_in[ki], kday)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=LinearSegmentedColormap.from_list(
        "deep", cm(0.10 + 0.72 * np.linspace(0, 1, 256))))
    cax = axC.inset_axes([0.31, 0.790, 0.36, 0.026])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Readout depth", fontsize=6.8, color=MID, labelpad=2)
    cb.set_ticks([deep[0], deep[-1]])
    cb.ax.set_xticklabels([f"L{deep[0]}", f"L{deep[-1]}"], fontsize=6.8, color=MID)
    cb.ax.tick_params(length=1.5, width=0.5, colors=MID, pad=1)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(PALE)
