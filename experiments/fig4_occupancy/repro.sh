#!/usr/bin/env bash
# Figure 4: figures/fig4_occupancy.{pdf,png} and the natural-text numbers.
#
#     bash experiments/fig4_occupancy/repro.sh           # everything: ~50 min CPU + ~30 min GPU
#     bash experiments/fig4_occupancy/repro.sh --plot    # plate + numbers from existing captures
#
# Stages (data under ${WM_DATA:-~/.cache/weekday-manifold}/fig4_occupancy/):
#   1  select_windows.py      stream FineWeb, select 256-token windows      ~35 min CPU, online
#   2  common/dedup_windows.py  non-overlap + near-duplicate removal         ~6 min CPU
#   3  build_capture_list.py  caps, invariants -> capture_list_n256.npz       ~1 min CPU
#   4  capture_layer.py       layer-28 state at the capture token            ~10 min GPU, 1.3 GB
#   5  swap_build_set.py      seven weekday variants per weekday window     ~1 min CPU
#      swap_capture.py        their layer-28 states                        ~20 min GPU, 3.1 GB
#   6  figure_occupancy.py    the plate and its numbers                      ~2 min CPU
#      alignment_numbers.py   translation / scale / rotation per context     ~3 min CPU
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig4_occupancy"
D="experiments/fig4_occupancy"
DATA="$($PY -c 'from weekday_manifold.utils import data_dir; print(data_dir("fig4_occupancy"))')"

if [[ "${1:-}" != "--plot" ]]; then
  echo "[fig4] 1/6 select windows"
  HF_HUB_OFFLINE=0 $PY $D/select_windows.py
  echo "[fig4] 2/6 dedup"
  $PY experiments/common/dedup_windows.py --in "$DATA"
  echo "[fig4] 3/6 capture list"
  $PY $D/build_capture_list.py
  echo "[fig4] 4/6 layer-28 capture"
  $PY $D/capture_layer.py
  echo "[fig4] 5/6 matched-context set + capture"
  $PY $D/swap_build_set.py
  $PY $D/swap_capture.py
fi
echo "[fig4] 6/6 plate + numbers"
$PY $D/figure_occupancy.py
$PY $D/alignment_numbers.py
echo "[fig4] done"
