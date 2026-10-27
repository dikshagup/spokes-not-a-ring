"""Prompt builders for the probe families."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from weekday_manifold.manifold.concept import PromptSpec
from weekday_manifold.manifold.days import DAYS, N_DAYS
from weekday_manifold.manifold.prompt_library import _sites

# Same as days.INTERROGATIVE_TEMPLATES[0].
GOODFIRE_TEMPLATE = "Q: What day is {k} days after {z}?\nA:"
# Probes on this template are captured at the final token, where the next token is the day.
TYPO_OFFSET_K = 1
TIMES_OF_DAY: List[str] = ("morning", "afternoon", "evening")

DEICTIC_HOURS: List[int] = (12, 18, 24, 36)

HOURS_PER_DAY = 24

_WORD_HOURS: Dict[int, str] = {
    6: "six", 12: "twelve", 18: "eighteen", 24: "twenty-four", 30: "thirty",
    36: "thirty-six", 42: "forty-two", 48: "forty-eight", 54: "fifty-four",
    60: "sixty", 72: "seventy-two", 84: "eighty-four",
}

_WORD_HALVES: Dict[float, str] = {
    0.5: "half a",
    1.5: "one and a half",
    2.5: "two and a half",
    3.5: "three and a half",
    4.5: "four and a half",
    5.5: "five and a half",
    6.5: "six and a half",
}
_WORD_INTS: Dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven",
}

# Misspellings that mix two weekdays; blends (d, d) marks a near-typo of one day.
TYPO_BLENDS: List[Dict[str, object]] = [
    {"typo": "Tuurday", "blends": (1, 3)},      # Tuesday / Thursday
    {"typo": "Thuesday", "blends": (1, 3)},     # Tuesday / Thursday
    {"typo": "Tuednesday", "blends": (1, 2)},   # Tuesday / Wednesday
    {"typo": "Thursnesday", "blends": (2, 3)},  # Wednesday / Thursday
    {"typo": "Fridnesday", "blends": (2, 4)},   # Wednesday / Friday
    {"typo": "Frisday", "blends": (4, 5)},      # Friday / Saturday
    {"typo": "Satursday", "blends": (5, 6)},    # Saturday / Sunday
    {"typo": "Satunday", "blends": (5, 6)},     # Saturday / Sunday
    {"typo": "Sunsday", "blends": (5, 6)},      # Saturday / Sunday
    {"typo": "Sundnesday", "blends": (2, 6)},   # Wednesday / Sunday
    {"typo": "Monsday", "blends": (0, 6)},      # Monday / Sunday
    {"typo": "Fronday", "blends": (0, 4)},      # Monday / Friday
    {"typo": "Mridayy", "blends": (0, 4)},      # Monday / Friday
    # near-typos of a single day
    {"typo": "Wednesay", "blends": (2, 2)},
    {"typo": "Thurday", "blends": (3, 3)},
    {"typo": "Tusday", "blends": (1, 1)},
    {"typo": "Mondey", "blends": (0, 0)},
]

# answer_day for probes whose answer is not a whole day.
NO_ANSWER_DAY = -1


def _spec(text: str, family: str, meta: Dict[str, object],
          stated_day: Optional[str], answer_day: int) -> PromptSpec:
    """Build one probe ``PromptSpec`` with ``meta["sites"]`` set as in the prompt library."""
    full = dict(meta)
    full["sites"] = _sites(text, stated_day)
    full["family"] = family
    full["template_id"] = f"{family}:{full.get('content_id', 0)}"
    full["probe"] = True
    return PromptSpec(text=text, answer_day=answer_day, capture_text=text,
                      formulation=family, meta=full)


def _u_expected(z: int, k_days: float) -> float:
    """Expected ring coordinate for an offset of ``k_days`` from day ``z``."""
    return ((z + k_days) / float(N_DAYS)) % 1.0


def knot_distance(u, n_labels: int = N_DAYS):
    """Distance from ``u`` to the nearest day knot, in days (0 at a day, 0.5 halfway)."""
    import numpy as np

    f = np.mod(np.asarray(u, dtype=float) * n_labels, 1.0)
    return np.minimum(f, 1.0 - f)


def build_fractional_probes(
    ks: Sequence[float] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    renderings: Sequence[str] = ("digit", "word"),
) -> List[PromptSpec]:
    """P1: fractional day offsets ("2.5 days after Monday")."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for k in ks:
                if rendering == "digit":
                    k_str = f"{k:g}"
                else:
                    if k not in _WORD_HALVES:
                        raise ValueError(f"no word rendering for k={k}.")
                    k_str = _WORD_HALVES[k]
                if k < 1.0:
                    # "half a day", not "half a days"
                    text = (GOODFIRE_TEMPLATE
                            .replace("{k} days", f"{k_str} day" if rendering == "word"
                                     else f"{k_str} days")
                            .format(z=DAYS[z]))
                else:
                    text = GOODFIRE_TEMPLATE.format(k=k_str, z=DAYS[z])
                out.append(_spec(
                    text, "P1",
                    {"z": z, "k_days": float(k), "rendering": rendering,
                     "content_id": ri, "u_expected": _u_expected(z, k),
                     "arith_load": "high", "role": "compute"},
                    DAYS[z], NO_ANSWER_DAY))
    return out


def build_hours_probes(
    hours: Sequence[int] = (12, 18, 24, 30, 36, 42, 48, 60, 84),
    renderings: Sequence[str] = ("word", "digit"),
) -> List[PromptSpec]:
    """P2: offsets in hours ("thirty-six hours after Monday")."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for h in hours:
                k_days = h / float(HOURS_PER_DAY)
                if rendering == "word":
                    if h not in _WORD_HOURS:
                        raise ValueError(f"no word rendering for h={h}.")
                    h_str = _WORD_HOURS[h]
                else:
                    h_str = str(h)
                text = (GOODFIRE_TEMPLATE
                        .replace("{k} days", f"{h_str} hours").format(z=DAYS[z]))
                is_exact = (h % HOURS_PER_DAY) == 0
                out.append(_spec(
                    text, "P2",
                    {"z": z, "hours": int(h), "k_days": k_days,
                     "rendering": rendering, "content_id": ri,
                     "u_expected": _u_expected(z, k_days),
                     "is_exact_day": is_exact,
                     "arith_load": "moderate", "role": "compute",
                     "control": "C-int-hours" if is_exact else None},
                    DAYS[z],
                    (z + int(k_days)) % N_DAYS if is_exact else NO_ANSWER_DAY))
    return out


def build_typo_probes(blends: Optional[Sequence[Dict[str, object]]] = None) -> List[PromptSpec]:
    """P3: misspelled day names with a one-day offset."""
    rows = list(TYPO_BLENDS if blends is None else blends)
    out: List[PromptSpec] = []
    k_str = _WORD_INTS[TYPO_OFFSET_K]
    for ci, row in enumerate(rows):
        typo = str(row["typo"])
        a, b = row["blends"]  # type: ignore[misc]
        single = int(a) == int(b)
        text = GOODFIRE_TEMPLATE.format(k=k_str, z=typo)
        out.append(_spec(
            text, "P3",
            {"typo": typo, "blends": (int(a), int(b)),
             "z": int(a) if single else None, "k_days": float(TYPO_OFFSET_K),
             "content_id": ci, "rendering": "word",
             "u_expected": _u_expected(int(a), TYPO_OFFSET_K) if single else None,
             "is_single_typo": single,
             "control": "C-typo-single" if single else None,
             "arith_load": "none", "role": "compute"},
            typo, (int(a) + TYPO_OFFSET_K) % N_DAYS if single else NO_ANSWER_DAY))
    return out


def build_deictic_probes(
    hours: Sequence[int] = DEICTIC_HOURS,
    times: Sequence[str] = TIMES_OF_DAY,
) -> List[PromptSpec]:
    """P4: hour offsets from a time of day ("twenty-four hours after Monday afternoon")."""
    out: List[PromptSpec] = []
    for ci, tod in enumerate(times):
        for z in range(N_DAYS):
            for h in hours:
                if h not in _WORD_HOURS:
                    raise ValueError(f"no word rendering for h={h}.")
                k_days = h / float(HOURS_PER_DAY)
                text = (GOODFIRE_TEMPLATE
                        .replace("{k} days", f"{_WORD_HOURS[h]} hours")
                        .format(z=f"{DAYS[z]} {tod}"))
                is_exact = (h % HOURS_PER_DAY) == 0
                out.append(_spec(
                    text, "P4",
                    {"z": z, "hours": int(h), "k_days": k_days, "time_of_day": tod,
                     "rendering": "word", "content_id": ci,
                     "u_expected": _u_expected(z, k_days),
                     "is_exact_day": is_exact,
                     "control": "C-deictic-exact" if is_exact else None,
                     "arith_load": "low", "role": "compute"},
                    DAYS[z],
                    (z + int(k_days)) % N_DAYS if is_exact else NO_ANSWER_DAY))
    return out


def build_integer_controls(
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    renderings: Sequence[str] = ("word", "digit"),
) -> List[PromptSpec]:
    """C-int: whole-day offsets written as words or digits."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for k in ks:
                k_str = str(k) if rendering == "digit" else _WORD_INTS[k]
                text = GOODFIRE_TEMPLATE.format(k=k_str, z=DAYS[z])
                out.append(_spec(
                    text, "C-int",
                    {"z": z, "k_days": float(k), "k": int(k), "rendering": rendering,
                     "content_id": ri, "u_expected": _u_expected(z, k),
                     "arith_load": "canonical", "role": "compute",
                     "control": f"C-int-{rendering}"},
                    DAYS[z], (z + k) % N_DAYS))
    return out


_UNITS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy"}


def number_word(n: int) -> str:
    """Spell an integer 1..79."""
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    if tens not in _TENS:
        raise ValueError(f"no word form for {n}.")
    return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")


def build_hour_sweep(hours: Sequence[int] = tuple(range(1, 73)),
                     anchors: Optional[Sequence[int]] = None) -> List[PromptSpec]:
    """H: hour offsets in one-hour steps ("one hour after Monday", "two hours after Monday", ...)."""
    anchors = list(range(N_DAYS)) if anchors is None else list(anchors)
    out: List[PromptSpec] = []
    for z in anchors:
        for h in hours:
            k_days = h / float(HOURS_PER_DAY)
            unit = "hour" if h == 1 else "hours"
            text = (GOODFIRE_TEMPLATE
                    .replace("{k} days", f"{number_word(h)} {unit}").format(z=DAYS[z]))
            exact = (h % HOURS_PER_DAY) == 0
            out.append(_spec(
                text, "H",
                {"z": z, "hours": int(h), "k_days": k_days, "rendering": "word",
                 "content_id": 0, "u_expected": _u_expected(z, k_days),
                 "is_exact_day": exact,
                 "control": "H-exact-day" if exact else None,
                 "arith_load": "moderate", "role": "compute"},
                DAYS[z], (z + h // HOURS_PER_DAY) % N_DAYS if exact else NO_ANSWER_DAY))
    return out


# Captured at the final ".".
TIME_TEMPLATES: List[str] = ["It was {t} on {z}."]


def clock_word(h24: int) -> str:
    """24-hour index -> 12-hour clock phrase ("one pm"). 0 = midnight, 12 = noon."""
    suffix = "am" if h24 < 12 else "pm"
    h12 = h24 % 12
    return f"{_UNITS[12 if h12 == 0 else h12]} {suffix}"


# Midnight and noon are excluded.
CLOCK_HOURS: tuple = tuple(h for h in range(24) if h not in (0, 12))


def build_time_statements(hours: Sequence[int] = CLOCK_HOURS,
                          templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """T: statements of a time and a weekday, with no question or arithmetic."""
    tmpls = list(TIME_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            for h in hours:
                t = clock_word(h)
                text = tmpl.format(z=DAYS[z], t=t)
                out.append(_spec(
                    text, "T",
                    {"z": z, "clock_hour": int(h), "time": t, "k_days": h / 24.0,
                     "rendering": "word", "content_id": ci, "template": tmpl,
                     "day_is_last": False, "capture": "sentence_final_stop",
                     "u_expected": _u_expected(z, h / 24.0),
                     "is_exact_day": False,
                     "control": None,
                     "arith_load": "none", "role": "read", "gate_exempt": True},
                    DAYS[z], z))
    for sp in out:
        assert sp.text.endswith("."), f"capture site must be the full stop: {sp.text!r}"
        assert not sp.text[:-1].rstrip().endswith(("am", "pm")), (
            f"am/pm token adjacent to the capture site: {sp.text!r}")
    return out


# M: each prompt is 8 tokens with BOS, with the weekday at a fixed index and "." last.
MENTION_STATEMENT_TEMPLATES: List[str] = [
    "I go to market on {z}.",
    "She flies to Berlin on {z}.",
    "The library is closed on {z}.",
    "He calls his mother on {z}.",
    "We paint the fence on {z}.",
    "The train leaves early on {z}.",
]


# ME: each prompt is 9 tokens with BOS, with the weekday at index 2.
MENTION_EARLY_TEMPLATES: List[str] = [
    # third person; the content does not depend on the day
    "On {z}, she missed her train.",
    "On {z}, they painted the fence.",
    "On {z}, he sold his bike.",
    "On {z}, the pipes froze solid.",
    "On {z}, rain flooded the road.",
    "On {z}, the baby slept well.",
    "On {z}, he burned the toast.",
    "On {z}, they closed the bridge.",
    "On {z}, the lamp stopped working.",
    "On {z}, the door stuck again.",
    "On {z}, the kettle boiled over.",
    "On {z}, she lost her keys.",
    "On {z}, he missed the bus.",
    "On {z}, they fixed the roof.",
    "On {z}, the letter finally arrived.",
    "On {z}, he dropped his phone.",
    "On {z}, the heater broke down.",
    "On {z}, she cut her finger.",
    "On {z}, the tap kept dripping.",
    "On {z}, they moved the piano.",
]

# Text appended to each sentence per variant; the capture site is the last token.
MENTION_EARLY_SUFFIX = {"stop": "", "thisday": " On this day"}
MENTION_EARLY_POS_DAY = 2


def build_mention_early(variant: str = "stop",
                        templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """ME: a weekday named early in a sentence, captured at the final token."""
    if variant not in MENTION_EARLY_SUFFIX:
        raise ValueError(f"variant must be one of {sorted(MENTION_EARLY_SUFFIX)}")
    suffix = MENTION_EARLY_SUFFIX[variant]
    tmpls = list(MENTION_EARLY_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            out.append(_spec(
                tmpl.format(z=DAYS[z]) + suffix, "ME",
                {"z": z, "rendering": "word", "content_id": ci, "template": tmpl,
                 "day_is_last": False, "capture": f"final_token::{variant}",
                 "variant": variant, "pos_day": MENTION_EARLY_POS_DAY,
                 "u_expected": float(z) / N_DAYS,
                 "is_exact_day": True, "control": None,
                 "arith_load": "none", "role": "mention", "gate_exempt": True},
                DAYS[z], z))
    return out


# Two-day prompts: day_b = day_a + gap, and lean in [-1, 1] is positive toward day_a.

def circular_gap(a: int, b: int, n: int = N_DAYS) -> int:
    """Number of steps between two days the short way round (1 to 3 for distinct days)."""
    d = abs(int(a) - int(b)) % n
    return min(d, n - d)


def _day_pairs():
    """(day_a, day_b, gap) for all 21 unordered pairs, day_b = day_a + gap."""
    return [((a, (a + g) % N_DAYS, g)) for g in (1, 2, 3) for a in range(N_DAYS)]


# Each adverb is one token at the start and in the middle of a sentence, so day-token positions are fixed.
FREQUENCY_LEVELS: List[tuple] = [
    ("always", "never", 1.0), ("usually", "sometimes", 0.5), ("sometimes", "sometimes", 0.0),
    ("sometimes", "usually", -0.5), ("never", "always", -1.0),
]
SCHEDULED_BODIES: List[str] = [
    "her flight leaves", "the parcel arrives", "the delivery lands", "the repair happens",
    "she flies out", "the shipment clears", "the callback comes", "the scan happens",
    "the results arrive", "the letter comes", "the boiler is serviced",
    "the cast comes off", "the keys change hands", "the refund clears",
    "the survey takes place", "the valuation happens", "the transfer completes",
    "the pickup occurs", "the rescan is booked", "the paperwork arrives",
    "he gets the keys", "they move house", "the engineer visits",
    "the sample is collected", "the window is fitted", "the invoice is issued",
    "the account is closed", "the badge arrives", "the meter is read",
    "the tank is refilled", "the filter is changed", "the lock is replaced",
]


def build_frequency_pairs(bodies: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """Prompts ``"{Adv1} on {first}, {adv2} on {second}, {body}."`` for every day pair in both orders."""
    bods = list(SCHEDULED_BODIES if bodies is None else bodies)
    out: List[PromptSpec] = []
    for bi, body in enumerate(bods):
        for a, b, gap in _day_pairs():
            for order, (first, second) in enumerate(((a, b), (b, a))):
                for adv1, adv2, lean_first in FREQUENCY_LEVELS:
                    lean = lean_first if first == a else -lean_first
                    out.append(_spec(
                        f"{adv1.capitalize()} on {DAYS[first]}, {adv2} on {DAYS[second]}, "
                        f"{body}.", "U-FREQ",
                        {"z": NO_ANSWER_DAY, "family_name": "frequency",
                         "level": f"{adv1}/{adv2}", "lean": lean, "day_a": a, "day_b": b,
                         "gap": gap, "first": first, "order": order, "body": body,
                         "content_id": bi, "rendering": "word", "day_is_last": False,
                         "capture": "sentence_final_stop", "u_expected": None,
                         "is_exact_day": False, "control": None, "arith_load": "none",
                         "role": "mention", "gate_exempt": True},
                        None, NO_ANSWER_DAY))
    return out


# p is the implied probability of the day with the stronger modifier. Modifiers in a family
# have the same Llama-3.1 token count.
HEDGE_FAMILIES: Dict[str, List[tuple]] = {
    "certainty": [("certainly", "not", 0.98), ("definitely", "possibly", 0.88),
                  ("probably", "possibly", 0.78), ("probably", "perhaps", 0.68),
                  ("likely", "perhaps", 0.58), ("perhaps", "perhaps", 0.50)],
    "frequency": [("invariably", "scarcely", 0.95), ("mostly", "seldom", 0.85),
                  ("mostly", "occasionally", 0.75), ("generally", "occasionally", 0.68),
                  ("typically", "commonly", 0.55), ("occasionally", "occasionally", 0.50)],
    "odds": [("nine in ten", "one in ten", 0.90), ("four in five", "one in five", 0.80),
             ("three in four", "one in four", 0.75), ("two in three", "one in three", 0.667),
             ("three in five", "two in five", 0.60), ("one in two", "one in two", 0.50)],
    "percent": [("ninety percent", "ten percent", 0.90), ("eighty percent", "twenty percent", 0.80),
                ("seventy percent", "thirty percent", 0.70), ("sixty percent", "forty percent", 0.60),
                ("fifty percent", "fifty percent", 0.50)],
    # the modifier sits between the two days, so the reversed order swaps the day names
    "either_or": [("and not", None, 0.97), ("rather than", None, 0.85),
                  ("sooner than", None, 0.78), ("or possibly", None, 0.70),
                  ("or perhaps", None, 0.62), ("or equally", None, 0.50)],
}
HEDGE_BODIES: List[str] = [
    "her flight leaves", "the parcel arrives", "the meter is read", "he files the report",
    "she visits her mother", "the deliveries come", "I work from home",
    "I take the long way round", "you can find me here", "the tide turns",
    "it rains on the coast", "my knee gives trouble",
]


def _hedge_text(family: str, strong: str, weak: Optional[str],
                d_strong: int, d_weak: int, body: str) -> str:
    if family == "either_or":
        t = f"{DAYS[d_strong]} {strong} {DAYS[d_weak]}, {body}."
    else:
        on = " on" if family in ("frequency", "odds") else ""
        t = f"{strong}{on} {DAYS[d_strong]}, {weak}{on} {DAYS[d_weak]}, {body}."
    return t[0].upper() + t[1:]


def build_hedge_families(families: Optional[Sequence[str]] = None,
                         bodies: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """Hedge prompts for every family, day pair, order and body."""
    names = list(HEDGE_FAMILIES if families is None else families)
    bods = list(HEDGE_BODIES if bodies is None else bodies)
    out: List[PromptSpec] = []
    for fam in names:
        for li, (strong, weak, p) in enumerate(HEDGE_FAMILIES[fam]):
            for a, b, gap in _day_pairs():
                for bi, body in enumerate(bods):
                    for order, (ds, dw) in enumerate(((a, b), (b, a))):
                        lean = (2 * p - 1) if ds == a else -(2 * p - 1)
                        out.append(_spec(
                            _hedge_text(fam, strong, weak, ds, dw, body), "U-HEDGE",
                            {"z": NO_ANSWER_DAY, "family_name": fam,
                             "level": strong if weak is None else f"{strong}/{weak}",
                             "level_index": li, "p_strong": p, "lean": round(lean, 3),
                             "day_a": a, "day_b": b, "gap": gap, "first": ds,
                             "order": order, "body": body, "content_id": bi,
                             "rendering": "word", "day_is_last": False,
                             "capture": "sentence_final_stop", "u_expected": None,
                             "is_exact_day": False, "control": None,
                             "arith_load": "none", "role": "mention",
                             "gate_exempt": True},
                            None, NO_ANSWER_DAY))
    return out


def build_mention_statements(
        templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """M: a weekday stated in a sentence, captured at the final full stop."""
    tmpls = list(MENTION_STATEMENT_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(_spec(
                text, "M",
                {"z": z, "rendering": "word", "content_id": ci, "template": tmpl,
                 "day_is_last": False, "capture": "sentence_final_stop",
                 "u_expected": float(z) / N_DAYS,
                 "is_exact_day": True, "control": None,
                 "arith_load": "none", "role": "mention", "gate_exempt": True},
                DAYS[z], z))
    for sp in out:
        assert sp.text.endswith("."), f"capture site must be the full stop: {sp.text!r}"
    return out


PROBE_BUILDERS = {
    "H": build_hour_sweep,
    "T": build_time_statements,
    "M": build_mention_statements,
    "ME": build_mention_early,
    "P1": build_fractional_probes,
    "P2": build_hours_probes,
    "P3": build_typo_probes,
    "P4": build_deictic_probes,
    "C-int": build_integer_controls,
}

PROBE_ORDER = ("C-int", "H", "T", "M")


def build_all_probes(families: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """All prompts of the families in ``PROBE_ORDER`` (or ``families``), with duplicate texts removed."""
    names = list(PROBE_ORDER if families is None else families)
    unknown = set(names) - set(PROBE_BUILDERS)
    if unknown:
        raise ValueError(f"unknown probe families: {sorted(unknown)}; "
                         f"choose from {sorted(PROBE_BUILDERS)}.")
    out: List[PromptSpec] = []
    seen = set()
    for name in names:
        for sp in PROBE_BUILDERS[name]():
            if sp.text in seen:
                continue
            seen.add(sp.text)
            out.append(sp)
    return out
