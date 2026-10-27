"""Predicted clock hour for figure 2C, as the circular mean of teacher-forced hour probabilities."""
from __future__ import annotations

import numpy as np
import torch

MINUTES = (0, 15, 30, 45)


def _hh_ap(h):
    return (12 if h % 12 == 0 else h % 12), ("am" if h < 12 else "pm")


def unambiguous_forms():
    """List of (string, hour value) for time strings that fix the hour, including am or pm."""
    out = []
    for h in range(24):
        hh, ap = _hh_ap(h)
        AP, DOT, DOTU = ap.upper(), f"{ap[0]}.{ap[1]}.", f"{ap[0].upper()}.{ap[1].upper()}."
        for m in MINUTES:
            t = "" if m == 0 else f":{m:02d}"
            v = h + m / 60.0
            out += [(f" {hh}{t}{ap}", v), (f" {hh}{t} {ap}", v), (f" {hh}{t} {DOT}", v),
                    (f" {hh}{t}{AP}", v), (f" {hh}{t} {AP}", v), (f" {hh}{t} {DOTU}", v)]
    for h in range(13, 24):
        for m in MINUTES:
            out.append((f" {h}:{m:02d}", h + m / 60.0))
    out += [(" noon", 12.0), (" midday", 12.0), (" 12 noon", 12.0),
            (" midnight", 0.0), (" 12 midnight", 0.0)]
    seen, uniq = set(), []
    for s, v in out:
        if s not in seen:
            seen.add(s); uniq.append((s, v))
    return uniq


def ambiguous_prefixes():
    """List of (string, clock-face value) for time strings that do not specify am or pm."""
    out = [(f" {hh}", float(hh)) for hh in range(1, 13)]
    out += [(f" {hh}:{m:02d}", hh + m / 60.0) for hh in range(1, 13) for m in MINUTES if m]
    out += [(f" {hh} o'clock", float(hh)) for hh in range(1, 13)]
    return out


class TimeReadout:

    def __init__(self, model, prepend_bos=True, chunk=320):
        self.m = model
        self.bos = prepend_bos
        self.chunk = chunk
        self.forms = unambiguous_forms()
        self.values = np.array([v for _, v in self.forms])
        amb = ambiguous_prefixes()
        self.amb = [a for a, _ in amb]
        self.amb_val = np.array([v for _, v in amb])
        tok = lambda s: model.to_tokens(s, prepend_bos=False)[0].tolist()  # noqa: E731
        self._ids = [tok(s) for s, _ in self.forms]
        self._amb_ids = [tok(s) for s in self.amb]
        # for each ambiguous prefix, the scored strings that start with it
        self._ext = [[j for j, (s, _) in enumerate(self.forms) if s.startswith(a)]
                     for a in self.amb]

    @torch.no_grad()
    def _logp(self, pre, cont_ids, fwd_hooks=None):
        out = np.full(len(cont_ids), -np.inf)
        dev, n = self.m.cfg.device, len(pre)
        for lo in range(0, len(cont_ids), self.chunk):
            part = cont_ids[lo:lo + self.chunk]
            seqs = [pre + c for c in part]
            mx = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), mx), dtype=torch.long, device=dev)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s, device=dev)
            logits = (self.m.run_with_hooks(ids, return_type="logits", fwd_hooks=fwd_hooks)
                      if fwd_hooks else self.m(ids, return_type="logits"))
            # positions n-1 .. mx-2 predict continuation tokens n .. mx-1
            lg = torch.log_softmax(logits[:, n - 1:mx - 1, :].float(), -1)
            for j, s in enumerate(seqs):
                k = len(s) - n
                tgt = torch.tensor(s[n:], device=dev).unsqueeze(-1)
                out[lo + j] = float(lg[j, :k].gather(-1, tgt).sum())
        return out

    def score(self, text, fwd_hooks=None):
        pre = self.m.to_tokens(text, prepend_bos=self.bos)[0].tolist()
        lp = self._logp(pre, self._ids, fwd_hooks)
        lpa = self._logp(pre, self._amb_ids, fwd_hooks)
        p, pa = np.exp(lp), np.exp(lpa)

        captured = float(p.sum())
        # residual per ambiguous prefix: its probability minus that of the scored strings extending it
        resid = np.array([max(float(pa[i] - p[self._ext[i]].sum()), 0.0)
                          for i in range(len(self.amb))])
        # remove the ":mm" and "o'clock" residuals from the bare hour's residual, which includes them
        bare = {f" {h}": i for i, h in zip(range(len(self.amb)), range(1, 13))}
        for i, a in enumerate(self.amb):
            for b, bi in bare.items():
                if a != b and a.startswith(b):
                    resid[bi] = max(resid[bi] - resid[i], 0.0)
        ambiguous = float(resid.sum())

        w = p / max(captured, 1e-30)
        ang = 2 * np.pi * self.values / 24.0
        R = complex(float((w * np.cos(ang)).sum()), float((w * np.sin(ang)).sum()))
        cm = (np.arctan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi)

        bins = np.zeros(24)
        for k, v in enumerate(self.values):
            bins[int(round(v)) % 24] += w[k]
        am = float(w[(self.values >= 0) & (self.values < 12)].sum())
        return dict(p=bins, circ_mean=float(cm), conc=abs(R),
                    logodds=float(np.log(max(am, 1e-12)) - np.log(max(1 - am, 1e-12))),
                    captured=captured, ambiguous=ambiguous, resid=resid, logp=lp, weights=w)

    def bounds(self, res):
        """Circular mean if all the ambiguous mass were am, and if it were all pm."""
        mass = np.concatenate([res["weights"] * res["captured"], res["resid"]])
        tot = mass.sum()
        base = self.amb_val % 12.0
        out = {}
        for side, off in (("am", 0.0), ("pm", 12.0)):
            vv = np.concatenate([self.values, base + off])
            ang = 2 * np.pi * vv / 24.0
            ww = mass / max(tot, 1e-30)
            R = complex(float((ww * np.cos(ang)).sum()), float((ww * np.sin(ang)).sum()))
            out[side] = float((np.arctan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi))
        return out


def circ_diff(a, b):
    """Signed circular difference a - b in hours, wrapped to (-12, 12]."""
    return (a - b + 12.0) % 24.0 - 12.0
