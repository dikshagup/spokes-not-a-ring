"""Split of the weekday windows into fit and held-out sets by document."""
from __future__ import annotations

import numpy as np

from weekday_manifold.corpus.classes import WEEKDAY_CLASSES

FIT_FRACTION = 0.34


def fit_documents(cls, doc, classes=WEEKDAY_CLASSES, frac=FIT_FRACTION, seed=0):
    """Return a random ``round(frac * n)`` of the ``n`` unique documents as the fit set."""
    cls = np.asarray(cls).astype(str); doc = np.asarray(doc).astype(str)
    docs = np.unique(doc[np.isin(cls, list(classes))])
    np.random.default_rng(seed).shuffle(docs)
    return set(docs[:int(round(frac * len(docs)))].tolist())


def fit_mask(cls, doc, classes=WEEKDAY_CLASSES, frac=FIT_FRACTION, seed=0):
    """True for analysed-class windows in a fit document."""
    cls = np.asarray(cls).astype(str); doc = np.asarray(doc).astype(str)
    fd = fit_documents(cls, doc, classes, frac, seed)
    return np.isin(cls, list(classes)) & np.isin(doc, list(fd))


def held_out_mask(cls, doc, classes=WEEKDAY_CLASSES, frac=FIT_FRACTION, seed=0):
    """True for analysed-class windows not in a fit document."""
    cls = np.asarray(cls).astype(str); doc = np.asarray(doc).astype(str)
    fd = fit_documents(cls, doc, classes, frac, seed)
    return np.isin(cls, list(classes)) & ~np.isin(doc, list(fd))
