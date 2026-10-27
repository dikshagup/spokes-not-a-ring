#!/usr/bin/env bash
# Appendix "Non-uniform sensitivity with weekday arithmetic" -> figures/fig6_arithmetic.png (+ .pdf)
#
#   bash experiments/fig6_arithmetic/repro.sh           # captures (GPU) + plot
#   bash experiments/fig6_arithmetic/repro.sh --plot    # plot only
#
# Stages:
#   1. day centroids at every layer and site                  GPU minutes
#   2. ring walk, both sites                                  GPU ~3-4 h
#   3. polar Jacobian field, the same 49 prompts              GPU ~1-2 h
#      plot                                                   CPU seconds
# The readout is the final ":" (--readout-pos -1); the weekday token is at position 9.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig6_arithmetic"
DATA=$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('fig6_arithmetic'))")
RINGS="$DATA/rings.npz"
WALK="$DATA/ring_walk.npz"
FIELD="$DATA/jacobian_field.npz"

if [[ "${1:-}" != "--plot" ]]; then
  echo "[fig6] 1/3 ring centroids at every layer (GPU) -> $RINGS"
  $PY experiments/fig1_sensitivity/ring_walk.py --formulation interrogative \
      --dump-rings "$RINGS" --out "$WALK"
  echo "[fig6] 2/3 the ring walk, weekday token and answer slot (GPU) -> $WALK"
  $PY experiments/fig1_sensitivity/ring_walk.py --formulation interrogative --out "$WALK"
  echo "[fig6] 3/3 the polar Jacobian field (GPU) -> $FIELD"
  $PY experiments/fig1_sensitivity/jacobian_field.py --formulation interrogative \
      --readout-pos -1 --n-prompts 49 --out "$FIELD"
fi

for f in "$RINGS" "$WALK" "$FIELD"; do
  [[ -f "$f" ]] || { echo "missing $f: run without --plot" >&2; exit 1; }
done
echo "[fig6] plate"
$PY experiments/fig6_arithmetic/figure_arithmetic.py --rings "$RINGS" --walk "$WALK" \
    --field "$FIELD"
