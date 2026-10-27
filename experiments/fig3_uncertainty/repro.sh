#!/usr/bin/env bash
# Figure 3: figures/fig3_uncertainty.{png,pdf}
# Appendix: figures/fig_uncertainty_families.{png,pdf}
#
#     bash experiments/fig3_uncertainty/repro.sh            # capture (GPU) + plot
#     bash experiments/fig3_uncertainty/repro.sh --plot     # plot only, from existing captures
#
# Stages, in order:
#   1. fit_ruler.py         the reference ring, corpus templates x 7 days          GPU ~5 min
#   2. capture_prompts.py   frequency pairs and the five hedge families, layer 28    GPU ~10 min
#   3. plot_fig3.py         Figure 3                                                  CPU seconds
#      plot_families.py     the appendix plate                                        CPU seconds

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig3_uncertainty"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

if [[ "${1:-}" != "--plot" ]]; then
  echo "[fig3] 1/3 reference ring"
  $PY experiments/fig3_uncertainty/fit_ruler.py
  echo "[fig3] 2/3 captures"
  $PY experiments/fig3_uncertainty/capture_prompts.py --family frequency
  $PY experiments/fig3_uncertainty/capture_prompts.py --family hedges
fi

echo "[fig3] 3/3 plates"
$PY experiments/fig3_uncertainty/plot_fig3.py
$PY experiments/fig3_uncertainty/plot_families.py
echo "[fig3] done"
