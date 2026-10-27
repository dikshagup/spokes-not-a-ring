"""The six window classes of the natural-text corpus; each window is 256 tokens ending at the capture token."""
from __future__ import annotations

# on-disk name -> description used in the paper
CLASS_LABEL = {
    "positive":            "weekday, no other temporal cue in context",
    "dirty_positive":      "weekday, with a temporal cue in context",
    "proper_noun_weekday": "weekday inside a proper noun",
    "near_miss":           "near miss: month, yesterday or tomorrow",
    "matched_negative":    "capitalised word",
    "floor":               "any token",
}

# Command-line short names.
CLASS_ALIAS = {
    "clean": "positive", "temporal": "dirty_positive", "proper-noun": "proper_noun_weekday",
    "near-miss": "near_miss", "capitalised": "matched_negative", "any": "floor",
}

# Analysed classes; proper-noun weekdays are captured but not analysed.
WEEKDAY_CLASSES = ("positive", "dirty_positive")
ALL_WEEKDAY_CLASSES = ("positive", "dirty_positive", "proper_noun_weekday")
CONTROL_CLASS = "matched_negative"


def resolve_classes(spec: str) -> list[str]:
    """Map comma-separated short or on-disk class names to on-disk names."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name = CLASS_ALIAS.get(tok, tok)
        if name not in CLASS_LABEL:
            raise ValueError(f"unknown class {tok!r}; known: {sorted(CLASS_ALIAS)}")
        out.append(name)
    return out
