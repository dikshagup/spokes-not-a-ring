#!/usr/bin/env bash
# Appendix "Activation steering", the methods schematic -> figures/fig4_schematic.png (+ .pdf)
#
#   bash experiments/methods_schematic/repro.sh           # capture (GPU) + plot
#   bash experiments/methods_schematic/repro.sh --plot    # plot only
#
# Cost: free once figure 1 has run; otherwise its Jacobian field, ~1-2 h GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/methods_schematic"
FIELD="$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('fig1_sensitivity'))")/jacobian_field.npz"

if [[ "${1:-}" != "--plot" && ! -f "$FIELD" ]]; then
  echo "[schematic] 1/1 figure 1's polar Jacobian field (GPU) -> $FIELD"
  $PY experiments/fig1_sensitivity/jacobian_field.py --formulation mention_early \
      --readout-pos 8 --out "$FIELD"
fi

[[ -f "$FIELD" ]] || { echo "missing $FIELD: run without --plot" >&2; exit 1; }
echo "[schematic] plate"
$PY experiments/methods_schematic/figure_schematic.py --field "$FIELD"
