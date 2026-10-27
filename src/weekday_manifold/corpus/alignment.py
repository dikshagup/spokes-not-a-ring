"""Nested translation, scale and rotation fits of each family's weekday points onto a shared ring,
with a noise null for the rotation step."""
from __future__ import annotations

import numpy as np

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fit_plane(raw, fit_mask):
    """Compute the day centroids, their 6-D bases, the 2-D ring plane and the ring radius from the fit windows."""
    A16 = raw["acts"]
    day = np.asarray(raw["day"]).astype(str)
    fitm = np.asarray(fit_mask, bool)
    Xf = A16[fitm].astype(np.float64)
    mu = Xf.mean(0)
    cents = np.stack([Xf[day[fitm] == d].mean(0) for d in DAYS]); del Xf
    basis6 = np.linalg.svd(cents - mu, full_matrices=False)[2][:6]
    mean_c = cents.mean(0)
    sv, Vt = np.linalg.svd(cents - mean_c, full_matrices=False)[1:]
    plane, pc_var = Vt[:2], (sv ** 2 / (sv ** 2).sum())[:2]
    hull6 = Vt[:6]
    ring_R = float(np.linalg.norm(cents - mean_c, axis=1).mean())
    return dict(mu=mu, cents=cents, basis6=basis6, mean_c=mean_c, plane=plane,
                pc_var=pc_var, hull6=hull6, ring_R=ring_R, n_fit=int(fitm.sum()))


def family_configs(swap, basis, center, chunk=20000):
    """Return each family's seven variants in `basis` coordinates as [F, 7, k], in DAYS order."""
    o = np.lexsort((swap["swap"], swap["fam"]))
    A = swap["acts"]
    Z = np.empty((len(o), basis.shape[0]), np.float64)
    B = basis.T.astype(np.float32)
    c32 = center.astype(np.float32)
    for a in range(0, len(o), chunk):
        idx = o[a:a + chunk]
        Z[a:a + chunk] = (A[idx].astype(np.float32) - c32) @ B
    F = len(o) // 7
    return Z.reshape(F, 7, -1)


def procrustes_proper(k, xt):
    """Per family, fit the rotation R (det +1) and scale s minimising |xt - s k R|, and flag fits that needed a reflection fix."""
    M = np.einsum("jd,fje->fde", k, xt)                     # k^T xt, [F, d, d]
    U, S, Vt = np.linalg.svd(M)
    R0 = U @ Vt
    reflect = np.linalg.det(R0) < 0
    sign = np.ones_like(S); sign[reflect, -1] = -1.0
    R = np.einsum("fij,fj,fjk->fik", U, sign, Vt)
    s = (S * sign).sum(1) / (k ** 2).sum()
    return R, s, reflect


def rotation_angle_deg(R):
    """Signed angle in degrees of each [2, 2] rotation R acting on row vectors (y = x R)."""
    return np.degrees(np.arctan2(R[:, 0, 1], R[:, 0, 0]))


def ladder(x, k):
    """Fit every family x [F, 7, d] onto k [7, d] by translation, then scale, then rotation, and return each stage."""
    x = np.asarray(x, np.float64); k = np.asarray(k, np.float64)
    kk = (k ** 2).sum()
    sse = lambda r: (r ** 2).sum((1, 2))
    sse0 = sse(x - k[None])
    t = x.mean(1)
    xt = x - t[:, None, :]
    sseT = sse(xt - k[None])
    s = (xt * k[None]).sum((1, 2)) / kk
    sseTS = sse(xt - s[:, None, None] * k[None])
    R, s2, reflect = procrustes_proper(k, xt)
    fitR = s2[:, None, None] * np.einsum("jd,fde->fje", k, R)
    sseTSR = sse(xt - fitR)
    out = dict(sse0=sse0, sseT=sseT, sseTS=sseTS, sseTSR=sseTSR, t=t, s=s, s_rot=s2, R=R,
               reflect=reflect, xt=xt, fit_TS=s[:, None, None] * k[None], fit_TSR=fitR)
    if k.shape[1] == 2:
        out["theta_deg"] = rotation_angle_deg(R)
    return out


def noise_null(k, s, sseTS, n_draws, rng, per_point=False):
    """Rotation-step SSE gain on simulated data y = s k + e, with sigma^2 = SSE_TS / (7d - d - 1)."""
    k = np.asarray(k, np.float64); d = k.shape[1]; F = len(s)
    sigma = np.sqrt(np.asarray(sseTS) / (7 * d - d - 1))
    gain = np.empty((n_draws, F, 7) if per_point else (n_draws, F))
    theta = np.empty((n_draws, F)) if d == 2 else None
    for i in range(n_draws):
        y = s[:, None, None] * k[None] + sigma[:, None, None] * rng.standard_normal((F, 7, d))
        L = ladder(y, k)
        if per_point:
            gain[i] = (((L["xt"] - L["fit_TS"]) ** 2).sum(2)
                       - ((L["xt"] - L["fit_TSR"]) ** 2).sum(2))
        else:
            gain[i] = L["sseTS"] - L["sseTSR"]
        if d == 2:
            theta[i] = L["theta_deg"]
    return gain, theta


def stage_distances(L, k):
    """Per point [F, 7]: distance to its own day's reference point at each stage, and whether that point is the nearest."""
    x = L["xt"] + L["t"][:, None, :]
    refs = {"raw": (x, np.broadcast_to(k, x.shape)),
            "T": (L["xt"], np.broadcast_to(k, x.shape)),
            "T+S": (L["xt"], L["fit_TS"]),
            "T+S+R": (L["xt"], L["fit_TSR"])}
    dist, hit = {}, {}
    for name, (pts, ref) in refs.items():
        dist[name] = np.linalg.norm(pts - ref, axis=2)
        d2 = ((pts[:, :, None, :] - ref[:, None, :, :]) ** 2).sum(-1)      # [F, 7, 7]
        hit[name] = d2.argmin(2) == np.arange(7)[None, :]
    return dist, hit


def spoke_distances(L, k):
    """Per point [F, 7]: distance to its own day's spoke (ray from the origin) at each stage, and whether that spoke is the nearest."""
    x = L["xt"] + L["t"][:, None, :]
    F = x.shape[0]
    k = np.asarray(k, np.float64)
    U0 = np.broadcast_to(k / np.linalg.norm(k, axis=1, keepdims=True), x.shape)   # [F, 7, d]
    kR = np.einsum("jd,fde->fje", k, L["R"])
    UR = kR / np.linalg.norm(kR, axis=2, keepdims=True)
    stages = {"raw": (x, U0), "T": (L["xt"], U0), "T+S": (L["xt"], U0), "T+S+R": (L["xt"], UR)}
    dist, hit = {}, {}
    for name, (pts, U) in stages.items():
        proj = np.einsum("fjd,fkd->fjk", pts, U)                        # p_j . u_k, [F, 7, 7]
        perp2 = (pts ** 2).sum(2)[:, :, None] - np.clip(proj, 0, None) ** 2
        d_all = np.sqrt(np.maximum(perp2, 0.0))                        # distance to every spoke
        dist[name] = d_all[:, np.arange(7), np.arange(7)]
        hit[name] = d_all.argmin(2) == np.arange(7)[None, :]
    return dist, hit
