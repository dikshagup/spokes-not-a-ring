# Spokes, not a Ring: Revisiting the Weekday Manifold in Language Models

Diksha Gupta, Rieke Fruengel, Stefan Heimersheim

Code for the paper. Models: Llama-3.1-8B, Mistral-7B-v0.1 and GPT-2 XL, via TransformerLens.

## Quick start

```bash
git clone https://github.com/dikshagup/spokes-not-a-ring.git
cd spokes-not-a-ring
python -m venv .venv && . .venv/bin/activate     # Python >= 3.10
pip install -e '.[dev]'
sudo apt-get install fonts-urw-base35            # Helvetica-metric fonts for the plates
pytest -q                                        # CPU only

huggingface-cli login                            # Llama-3.1-8B is gated
bash experiments/fig3_uncertainty/repro.sh       # cheapest figure
```

The captures need a single GPU (timed on a 32 GB card). Rough times from scratch: figure 1
~4 h, figure 2 ~2.5 h, figure 3 ~15 min, figure 4 ~1.5 h (half of it CPU).

Each figure is reproduced by `bash experiments/<figure>/repro.sh` (`--plot` re-renders from
existing captures). Set `PY=/path/to/python` to use a different interpreter. Captures go to
`${WM_DATA:-~/.cache/weekday-manifold}/<figure>/` (~10 GB for everything). Once the weights
are cached you can set `HF_HUB_OFFLINE=1`; the two FineWeb-streaming stages turn it off for
themselves.

## Notes

float16 is required for figures 1 and 6. In bfloat16 the finite-difference speed measure
is dominated by rounding error, and the figure changes without any error message.

## Citation

See `CITATION.cff`. MIT licensed.
