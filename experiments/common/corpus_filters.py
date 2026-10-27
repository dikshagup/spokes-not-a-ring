"""Token-level filters shared by the two FineWeb window selections (tokeniser only)."""
from __future__ import annotations

import re

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
# Day ids use capitalised full names only.
# Shifting deixis demotes a weekday window; co-referential deixis is only flagged.
SHIFTING_DEIXIS = ["yesterday", "tomorrow"]
COREF_DEIXIS = ["today", "tonight"]

# matched in the raw text and mapped to tokens through the offset mapping
RE_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm|AM|PM|a\.m\.|p\.m\.)\b")
RE_ORDINAL_DATE = re.compile(r"\b(?:[12]?\d|3[01])(?:st|nd|rd|th)\b")
RE_YEAR = re.compile(r"\b(?:1[0-9]{3}|2[0-9]{3})\b")
RE_SENT_END = re.compile(r"[.!?]")


def build_token_sets(tok):
    """Return a dict of single-token id sets for day names, month names and deictic words."""
    def ids_for(forms):
        out = set()
        for f in forms:
            enc = tok.encode(f, add_special_tokens=False)
            if len(enc) == 1:
                out.add(enc[0])
        return out

    def both(words):                       # bare and space-prefixed
        return ids_for(list(words) + [" " + w for w in words])

    day_capture = {d: both([d]) for d in DAYS}
    day_all = set().union(*day_capture.values())
    return {
        "day_capture": day_capture,        # per-day capture ids
        "day_exclude": day_all,            # union of the per-day capture ids
        "month": both(MONTHS),
        "shifting": both(SHIFTING_DEIXIS) | both([w.capitalize()
                                                  for w in SHIFTING_DEIXIS]),
        "coref": both(COREF_DEIXIS) | both([w.capitalize() for w in COREF_DEIXIS]),
    }


def regex_temporal_token_mask(text, offsets):
    """For each regex form (clock, ordinal, year), the set of token indices overlapping a match."""
    out = {}
    for name, rx in (("clock", RE_CLOCK), ("ordinal", RE_ORDINAL_DATE),
                     ("year", RE_YEAR)):
        spans = [(m.start(), m.end()) for m in rx.finditer(text)]
        marks = set()
        if spans:
            for i, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                for (s, e) in spans:
                    if a < e and s < b:
                        marks.add(i)
                        break
        out[name] = marks
    return out


def title_case_run(pieces, i, look=5, need=3):
    """True if at least `need` of the `look` tokens before i are capitalised words."""
    lo = max(0, i - look)
    caps = sum(1 for p in pieces[lo:i] if is_capitalised_word(p))
    return caps >= need


def is_capitalised_word(piece):
    p = piece.lstrip()
    return bool(p) and p[0].isupper() and p.isalpha() and len(p) >= 3


def proper_noun_trigger(pieces, i):
    """(side, trigger word) if the weekday at i looks like part of a proper name, else (None, None)."""
    if i + 1 < len(pieces) and is_capitalised_word(pieces[i + 1]):
        return "right", pieces[i + 1].strip()
    if i - 1 >= 0 and is_capitalised_word(pieces[i - 1]):
        prev2 = pieces[i - 2].strip() if i - 2 >= 0 else ""
        if not RE_SENT_END.search(prev2):
            return "left", pieces[i - 1].strip()
    return None, None
