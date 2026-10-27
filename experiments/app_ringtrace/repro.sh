#!/usr/bin/env bash
# Ring-trace appendix: writes figures/fig_ring_trace.pdf (+ .png) and experiments/results/fig_ring_trace_numbers.json.
#
#   bash experiments/app_ringtrace/repro.sh           # select, capture and plot
#   bash experiments/app_ringtrace/repro.sh --plot    # plot only, from existing captures
#
# Run time: stage 1 about 15 min CPU; stages 2-3 about 5 min CPU; stage 4 about 2 min GPU; plot about 1 min CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/app_ringtrace"
DATA=$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('app_ringtrace'))")
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"

if [[ "${1:-}" != "--plot" ]]; then
  # stage 1 streams FineWeb and needs network access
  echo "[ringtrace] 1/4 select period-final windows -> $DATA"
  HF_HUB_OFFLINE=0 $PY experiments/app_ringtrace/select_period_windows.py --model "$MODEL" \
      --out "$DATA" --target-positives 5000 --target-negatives 60000 --max-docs 400000

  echo "[ringtrace] 2/4 near-duplicate removal"
  $PY experiments/common/dedup_windows.py --in "$DATA" --n-max 256

  echo "[ringtrace] 3/4 capture list + invariants"
  $PY experiments/app_ringtrace/build_capture_list.py --in "$DATA" --model "$MODEL"
  $PY experiments/app_ringtrace/verify_corpus.py --corpus "$DATA" --model "$MODEL"

  echo "[ringtrace] 4/4 layer x position trace (GPU)"
  $PY experiments/app_ringtrace/capture_trace.py --in "$DATA" --model "$MODEL"
fi

[[ -f "$DATA/trace_n16.npz" ]] || { echo "missing $DATA/trace_n16.npz: run without --plot" >&2; exit 1; }
echo "[ringtrace] plate"
$PY experiments/app_ringtrace/figure_ring_trace.py --trace "$DATA/trace_n16.npz" --also-png
