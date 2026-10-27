#!/usr/bin/env bash
# Figure 2: writes figures/fig2_time_of_day.pdf and the time-of-day numbers.
#
#     bash experiments/fig2_timeofday/repro.sh           # capture, steering ladder and plot
#     bash experiments/fig2_timeofday/repro.sh --plot    # plot and numbers from existing captures
#
# Stages (Llama-3.1-8B, one GPU):
#   1. capture_prompts.py   capture 364 clock, phrase and placebo prompts at the weekday token   ~1 min
#   2. carrier_screen.py    choose the twelve carrier sentences for steering                    ~1 min
#   3. steer_ladder.py      steering ladder: 3 arms x 7 doses x 84 prompts, layers 2-28         ~2.5 h
#   4. analysis.py          numbers for the text, written to experiments/results/              CPU
#   5. plate.py             the figure                                                          CPU
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig2_timeofday${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" != "--plot" ]]; then
  $PY experiments/fig2_timeofday/capture_prompts.py
  $PY experiments/fig2_timeofday/carrier_screen.py
  $PY experiments/fig2_timeofday/steer_ladder.py
fi
$PY experiments/fig2_timeofday/analysis.py
$PY experiments/fig2_timeofday/plate.py --also-png
