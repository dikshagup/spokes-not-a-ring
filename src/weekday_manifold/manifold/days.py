"""Weekday names, tokenization helpers, and prompt builders."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

from weekday_manifold.manifold.concept import Concept, PromptSpec

# index 0 = Monday
DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
N_DAYS = len(DAYS)

DIGIT_OFFSETS: Dict[int, str] = {k: str(k) for k in range(0, 8)}
WORD_OFFSETS: Dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
}


# Singular or plural, any case, whole word (possessives match via ``\b``).
DAY_PATTERNS: List["re.Pattern[str]"] = [re.compile(rf"\b{d}s?\b", re.I) for d in DAYS]


def named_days(text: str) -> List[int]:
    """Indices of every weekday named in ``text``, singular or plural."""
    return [d for d in range(N_DAYS) if DAY_PATTERNS[d].search(text)]


def swap_day(text: str, src: int, dst: int) -> Optional[str]:
    """Return ``text`` with day ``src`` replaced by ``dst`` (case and plural kept), or None if the result does not name only ``dst``."""
    def repl(m: "re.Match[str]") -> str:
        tok = m.group(0)
        new = DAYS[dst] + ("s" if tok.lower().endswith("s") else "")
        if tok.isupper():
            return new.upper()
        if tok[0].islower():
            return new.lower()
        return new

    out, n = DAY_PATTERNS[src].subn(repl, text)
    if n == 0 or named_days(out) != [dst]:
        return None
    return out


def day_index(name: str) -> int:
    """Index of a weekday name (case-insensitive); raises if unknown."""
    key = name.strip().capitalize()
    if key not in DAYS:
        raise ValueError(f"unknown weekday {name!r}; expected one of {DAYS}.")
    return DAYS.index(key)


def add_days(start: int, k: int) -> int:
    """Weekday index ``k`` days after ``start`` (cyclic, mod 7)."""
    return (start + k) % N_DAYS


def offset_word(k: int, style: str) -> str:
    """Render an offset ``k`` as a digit ("3") or spelled-out word ("three")."""
    table = DIGIT_OFFSETS if style == "digit" else WORD_OFFSETS
    if style not in ("digit", "word"):
        raise ValueError(f"offset style must be 'digit' or 'word', got {style!r}.")
    if k not in table:
        raise ValueError(f"no offset rendering for k={k} in style {style!r}.")
    return table[k]


def tokenize_day(
    tokenize_fn: Callable[[str], Sequence[int]],
    day: str,
    leading_space: bool = True,
) -> List[int]:
    """Token-id list for a weekday, with a leading space by default."""
    text = (" " + day) if leading_space else day
    ids = list(tokenize_fn(text))
    if not ids:
        raise ValueError(f"tokenizer returned no tokens for {text!r}.")
    return ids


def day_token_table(
    tokenize_fn: Callable[[str], Sequence[int]],
    leading_space: bool = True,
) -> Dict[str, List[int]]:
    """``{day: [token ids]}`` for all seven weekdays."""
    return {day: tokenize_day(tokenize_fn, day, leading_space) for day in DAYS}


def tokenization_report(table: Dict[str, List[int]]) -> str:
    """Readable summary of how each weekday tokenizes."""
    lines = ["weekday tokenization (leading space, no BOS):"]
    for day, ids in table.items():
        kind = "single-token" if len(ids) == 1 else f"{len(ids)}-token"
        lines.append(f"  {day:<9} -> {ids}  ({kind})")
    n_multi = sum(1 for ids in table.values() if len(ids) > 1)
    lines.append(
        f"  => {n_multi}/{len(table)} weekdays are multi-token; "
        "scorer teacher-forces the FULL day string."
    )
    return "\n".join(lines)


ARITH_TEMPLATES: List[str] = [
    "The day {k} days after {z} is",
    "{k} days after {z} is",
    "If today is {z}, then {k} days later it is",
    "Starting from {z} and counting forward {k} days, you reach",
    "Counting {k} days ahead from {z}, you land on",
]


def build_arithmetic_prompts(
    offset_style: str = "digit",
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6),
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Arithmetic prompts such as "The day {k} days after {z} is"."""
    templates = list(templates) if templates is not None else ARITH_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            for k in ks:
                text = tmpl.format(k=offset_word(k, offset_style), z=DAYS[z])
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, k),
                        capture_text=text,
                        formulation="arith",
                        meta={"z": z, "k": k, "offset_style": offset_style,
                              "template": tmpl},
                    )
                )
    return out


# Goodfire's template, unchanged; it has a newline before "A:" and uses word offsets.
INTERROGATIVE_TEMPLATES: List[str] = [
    "Q: What day is {k} days after {z}?\nA:",
]


def build_interrogative_prompts(
    offset_style: str = "word",
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Question prompts: "Q: What day is {k} days after {z}? A:"."""
    templates = list(templates) if templates is not None else INTERROGATIVE_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            for k in ks:
                text = tmpl.format(k=offset_word(k, offset_style), z=DAYS[z])
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, k),
                        capture_text=text,
                        formulation="interrogative",
                        meta={"z": z, "k": k, "offset_style": offset_style,
                              "template": tmpl},
                    )
                )
    return out


SEQ_RUN_LENGTHS = (2, 3, 4, 5, 6)
SEQ_SEPARATORS = ("plain", "comma")
_SEP_STR = {"plain": " ", "comma": ", "}


def build_sequence_prompts(
    run_lengths: Sequence[int] = SEQ_RUN_LENGTHS,
    separators: Sequence[str] = SEQ_SEPARATORS,
) -> List[PromptSpec]:
    """Sequence prompts such as "Monday Tuesday Wednesday", with answer "Thursday"."""
    out: List[PromptSpec] = []
    for sep_name in separators:
        if sep_name not in _SEP_STR:
            raise ValueError(f"unknown separator {sep_name!r}; "
                             f"choose from {sorted(_SEP_STR)}.")
        sep = _SEP_STR[sep_name]
        for L in run_lengths:
            for z in range(N_DAYS):
                seq = [DAYS[add_days(z, i)] for i in range(L)]
                text = sep.join(seq)
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, L),
                        capture_text=text,
                        formulation="seq",
                        meta={"z": z, "run_length": L, "separator": sep_name},
                    )
                )
    return out


RELATIONAL_TEMPLATES: List[str] = [
    "The day after {z} is",
    "The day that comes right after {z} is",
    "After {z} comes",
]


def build_relational_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """One-step prompts such as "The day after {z} is", with answer z + 1."""
    templates = list(templates) if templates is not None else RELATIONAL_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=add_days(z, 1),
                    capture_text=text,
                    formulation="relational",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


# Captured at the day token. Each template is 6 tokens and 6 words, ends in " {z}", and names no other weekday.
MENTION_TEMPLATES: List[str] = [
    "He left the city on {z}",
    "I saw them again last {z}",
    "She was born on a {z}",
    "They danced all night on {z}",
    "We flew back home last {z}",
    "I will call you next {z}",
    "They met up again last {z}",
    "We had dinner together last {z}",
    "The store was closed last {z}",
    "I woke up early last {z}",
    "She sent the email last {z}",
    "The children returned home last {z}",
    "I finished the book last {z}",
]


assert all(len(t.split()) == 6 and t.endswith(" {z}") for t in MENTION_TEMPLATES), (
    "MENTION_TEMPLATES must stay 6 whitespace words ending in ' {z}' so the day token "
    "sits at a fixed position; the token-count half of the invariant is tokenizer-"
    "specific and is checked in tests/test_manifold_days.py")


def build_mention_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Prompts that state the day, captured at the day token."""
    templates = list(templates) if templates is not None else MENTION_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            before, after = tmpl.split("{z}")
            capture_text = before + DAYS[z]            # ends at the day name
            text = capture_text + after
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=z,
                    capture_text=capture_text,
                    formulation="mention",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


# Captured at the final ".". Each template is 8 tokens, ends in a separate "." token, and names no other weekday.
TRAILING_TEMPLATES: List[str] = [
    "On {z}, she baked a cake.",
    "On {z}, he read a book.",
    "On {z}, I called my mother.",
    "On {z}, they cleaned the house.",
    "On {z}, they visited the museum.",
    "On {z}, I attended a meeting.",
    "On {z}, we cooked some pasta.",
    "On {z}, she wrote a letter.",
    "On {z}, he bought a newspaper.",
    "On {z}, we planted some flowers.",
]


def build_trailing_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Sentences that start with the day, captured at the final period."""
    templates = list(templates) if templates is not None else TRAILING_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=z,
                    capture_text=text,
                    formulation="trailing",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


FORMULATION_BUILDERS: Dict[str, Callable[..., List[PromptSpec]]] = {
    "arith": build_arithmetic_prompts,
    "interrogative": build_interrogative_prompts,
    "seq": build_sequence_prompts,
    "relational": build_relational_prompts,
    "mention": build_mention_prompts,
    "trailing": build_trailing_prompts,
}


def build_prompts(formulation: str, **kwargs) -> List[PromptSpec]:
    """Build the prompt set for a named formulation (see ``FORMULATION_BUILDERS``)."""
    if formulation not in FORMULATION_BUILDERS:
        raise ValueError(
            f"unknown formulation {formulation!r}; "
            f"choose from {sorted(FORMULATION_BUILDERS)}."
        )
    builder = FORMULATION_BUILDERS[formulation]
    per_formulation = {
        "arith": ("offset_style", "ks", "templates"),
        "interrogative": ("offset_style", "ks", "templates"),
        "seq": ("run_lengths", "separators"),
        "relational": ("templates",),
        "mention": ("templates",),
        "trailing": ("templates",),
    }
    allowed_keys = per_formulation.get(formulation, ())
    allowed = {k: v for k, v in kwargs.items()
               if k in allowed_keys and v is not None}
    return builder(**allowed)


def days_token_ids(model) -> Dict[int, List[int]]:
    """``{day_index: [token ids]}`` for each weekday, with a leading space and no BOS."""
    def tok(s: str) -> List[int]:
        return model.to_tokens(s, prepend_bos=False)[0].tolist()

    return {i: tokenize_day(tok, DAYS[i], leading_space=True) for i in range(N_DAYS)}


def make_days_concept() -> Concept:
    """The weekdays concept: seven cyclic labels, Monday to Sunday."""
    return Concept(
        name="days",
        labels=list(DAYS),
        is_cyclic=True,
        build_prompts=build_prompts,
        token_ids=days_token_ids,
        coordinate=None,   # label order 0..6 is the cyclic order
    )
