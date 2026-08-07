# Bonded Patch Repair — ML Failure-Load Prediction

Machine-learning pipeline for predicting static failure loads of patched and unpatched metallic specimens using experimental, FE, and GMM-augmented training data.

## Setup

```bash
source ~/venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python orchestrator.py
```

Pipeline: `preprocess.py` → `synthetic.py` → `train.py` → `evaluate.py` → `visualize.py`

Options:
```bash
python orchestrator.py --dataset patched      # one dataset only
python orchestrator.py --skip-training        # replot from existing outputs
python orchestrator.py --no-synthetic         # skip GMM augmentation
``` 

## Data

- `data/patched.xlsx`, `data/unpatched.xlsx` — raw inputs
- `config.yaml` — splits, GMM settings, model grids

## Outputs

Results and PDF figures are written to `outputs/` and copied to `Figures/` for the paper.

Key files: `*_model_comparison.xlsx`, `*_test_predictions.csv`, `*_all_results.xlsx`, `*_feature_importance.csv`

## Paper

Manuscript materials and revision tooling live under `ML_Enabled_Static_MDPI/` (LaTeX sources, figure sync, and `makediff.sh` for latexdiff PDFs).

## Project Structure

| Path | Role |
|------|------|
| `orchestrator.py` | End-to-end pipeline runner |
| `preprocess.py` | Load/clean Excel data, hold-out splits, encoders/scalers |
| `synthetic.py` | GMM-based synthetic oversampling with quality checks |
| `train.py` | Nested grouped CV training and model comparison |
| `evaluate.py` | Permutation feature-importance / sensitivity analysis |
| `visualize.py` | Publication-ready result figures |
| `visualize_architecture.py` | ANN architecture diagrams from checkpoints |
| `models.py` | Model factory (Linear → KAN/ANN and classical regressors) |
| `kan_model.py` | Thin pykan KAN wrapper |
| `config.yaml` / `config_test.yaml` | Experiment configuration |
| `data/` | Raw patched/unpatched Excel inputs |
| `outputs/` | Metrics, predictions, and figures |
| `ML_Enabled_Static_MDPI/` | Manuscript / latexdiff helpers for the MDPI paper |

## Models Compared

Linear, Polynomial, SVR, RandomForest, GradientBoosting, XGBoost, LightGBM, GaussianProcess, KAN, ANN.
