#!/usr/bin/env bash
# Figure 1, perturbation sensitivity around the weekday ring: writes figures/fig1_sensitivity.png (+ .pdf).
#
#   bash experiments/fig1_sensitivity/repro.sh           # capture (GPU) + plot
#   bash experiments/fig1_sensitivity/repro.sh --plot    # plot only, from existing captures
#
# Run time: stage 1 about 2-3 h GPU; stage 2 about 1-2 h GPU; plot a few seconds.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig1_sensitivity"
DATA=$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('fig1_sensitivity'))")
WALK="$DATA/ring_walk.npz"
FIELD="$DATA/jacobian_field.npz"

if [[ "${1:-}" != "--plot" ]]; then
  # read position 8 is the full stop; both stages require float16
  echo "[fig1] 1/2 the ring walk (GPU) -> $WALK"
  $PY experiments/fig1_sensitivity/ring_walk.py --formulation mention_early \
      --readout-pos 8 --out "$WALK"

  if [[ -f "$FIELD" ]]; then
    echo "[fig1] 2/2 $FIELD exists, skipping"
  else
    echo "[fig1] 2/2 the polar Jacobian field (GPU) -> $FIELD"
    $PY experiments/fig1_sensitivity/jacobian_field.py --formulation mention_early \
        --readout-pos 8 --out "$FIELD"
  fi
fi

for f in "$WALK" "$FIELD"; do
  [[ -f "$f" ]] || { echo "missing $f: run without --plot" >&2; exit 1; }
done
echo "[fig1] plate"
$PY experiments/fig1_sensitivity/figure_sensitivity.py --cascade "$WALK" --field "$FIELD"
