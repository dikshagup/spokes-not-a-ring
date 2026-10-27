#!/usr/bin/env bash
# Appendix "Reproducing sensitivity results with other models" -> figures/fig5_models.png (+ .pdf)
#
#   bash experiments/fig5_models/repro.sh           # captures (GPU) + plot
#   bash experiments/fig5_models/repro.sh --plot    # plot only
#
# Cost: ~1-2 h GPU per model (the Llama row reuses figure 1's field); plot seconds.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="src:experiments/common:experiments/fig5_models"
DATA=$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('fig5_models'))")
LLAMA="$($PY -c "from weekday_manifold.utils import data_dir; print(data_dir('fig1_sensitivity'))")/jacobian_field.npz"
MISTRAL="$DATA/jacobian_field_mistral.npz"
GPT2XL="$DATA/jacobian_field_gpt2xl.npz"

if [[ "${1:-}" != "--plot" ]]; then
  if [[ -f "$LLAMA" ]]; then
    echo "[fig5] 1/3 $LLAMA exists (figure 1's field), skipping"
  else
    echo "[fig5] 1/3 Llama-3.1-8B (GPU) -> $LLAMA"
    $PY experiments/fig1_sensitivity/jacobian_field.py --model meta-llama/Llama-3.1-8B \
        --formulation mention_early --readout-pos 8 --out "$LLAMA"
  fi
  # --require-seq-len 12 keeps only the 12-token prompts, since Mistral tokenises the templates to different lengths
  echo "[fig5] 2/3 Mistral-7B-v0.1 (GPU) -> $MISTRAL"
  $PY experiments/fig1_sensitivity/jacobian_field.py --model mistralai/Mistral-7B-v0.1 \
      --formulation mention_early --readout-pos 8 --require-seq-len 12 --out "$MISTRAL"
  echo "[fig5] 3/3 GPT-2 XL (GPU) -> $GPT2XL"
  $PY experiments/fig1_sensitivity/jacobian_field.py --model gpt2-xl --dtype float32 \
      --formulation mention_early --readout-pos 8 --readout-layer 42 --out "$GPT2XL"
fi

for f in "$LLAMA" "$MISTRAL" "$GPT2XL"; do
  [[ -f "$f" ]] || { echo "missing $f: run without --plot" >&2; exit 1; }
done
echo "[fig5] plate"
$PY experiments/fig5_models/figure_models.py --llama "$LLAMA" --mistral "$MISTRAL" \
    --gpt2xl "$GPT2XL"
