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
