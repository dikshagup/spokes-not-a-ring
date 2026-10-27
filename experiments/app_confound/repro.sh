#!/usr/bin/env bash
# Confound appendix: writes figures/fig_confound.png (+ .pdf) and prints the caption numbers.
#
#   bash experiments/app_confound/repro.sh          # capture (GPU), then plot
#   bash experiments/app_confound/repro.sh --plot   # plot only, from the existing capture
#
# Stages
#   1. capture_ring_logits.py   GPU, about 10 min on one 32 GB card
#   2. figure_confound.py       CPU, about 1 min
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:experiments/common:experiments/app_confound

if [ "${1:-}" != "--plot" ]; then
    python experiments/app_confound/capture_ring_logits.py
fi
python experiments/app_confound/figure_confound.py
