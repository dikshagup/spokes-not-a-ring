"""Carrier sentences for the Figure 2C steering prompts."""

from __future__ import annotations

from typing import List, Sequence

from weekday_manifold.manifold.days import DAYS

# Used as ``<carrier> on <Day> at around``; carriers contain no weekday or time words (see validate_carriers).
CARRIERS: List[str] = [
    "It happened",
    "The meeting is",
    "I'll see you",
    "She left",
    "We arrived",
    "The delivery came",
    "They departed",
    "The concert starts",
    "He called",
    "The train leaves",
    "My flight lands",
    "The shop opens",
    "The results came out",
    "We finished",
    "The package arrived",
    "She started work",
    "The class begins",
    "They announced it",
    "The storm hit",
    "I woke up",
    "The game ended",
    "He arrived home",
    "The office closes",
    "We spoke",
    "The letter came",
    "She phoned",
    "The bus departs",
    "It was decided",
    "The market opens",
    "They met",
    "The film premiered",
    "I sent it",
]

_TIME_WORDS = {
    "early", "late", "morning", "evening", "midday", "noon", "night",
    "afternoon", "dawn", "dusk", "hour", "time",
}


def validate_carriers(carriers: Sequence[str] = tuple(CARRIERS)) -> None:
    """Raise ValueError if a carrier contains a weekday or a time word."""
    lowered_days = {d.lower() for d in DAYS}
    for c in carriers:
        words = {w.strip(".,'").lower() for w in c.split()}
        bad_day = words & lowered_days
        bad_time = words & _TIME_WORDS
        if bad_day:
            raise ValueError(f"carrier {c!r} contains a weekday: {sorted(bad_day)}")
        if bad_time:
            raise ValueError(f"carrier {c!r} contains a time word: {sorted(bad_time)}")
