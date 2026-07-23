# Visualization module for generating publication-ready graphs.
# One graph per dataset.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 15
Y_CATEGORY_LABEL_FONTSIZE = 13

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.8,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "figure.dpi": 120,
})


def _style_axes(ax, xlabel=None, ylabel=None, tick_labelsize=TICK_LABEL_FONTSIZE,fontsize=AXIS_LABEL_FONTSIZE):
    # Publication styling: bold axis labels, bold tick labels, thicker spines.
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.8, length=6, labelsize=tick_labelsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=fontsize)

COLORS = {"primary": "#2E86AB", "secondary": "#A23B72", "accent": "#F18F01",
          "danger": "#C73E1D", "neutral": "#3B3B3B", "light": "#E8E8E8"}

MODEL_DISPLAY = {
    "Linear": "Linear Regression",
    "Polynomial": "Polynomial Regression",
    "SVR": "Support Vector Regression",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "GaussianProcess": "Gaussian Process",
    "ANN": "ANN",
    "KAN": "KAN",
}


# Primary reported metric = real-data-only nested CV (GMM augmentation is reported
# separately as an ablation). Fall back to the augmented columns if absent.
PRIMARY_MAPE = "test_mape_base"
PRIMARY_R2 = "test_r2_base"
PRIMARY_MAPE_STD = "test_mape_base_std"


def _mcol(df, name, fallback):
    return name if name in df.columns else fallback


def _best_model_row(comp: pd.DataFrame) -> pd.Series:
    # Return the best real-data-only model row (primary reported metric).
    col = _mcol(comp, PRIMARY_MAPE, "test_mape")
    return comp.sort_values(col, ascending=True).iloc[0]


def _dpi(config):
    # Read figure DPI from config with a publication default.
    return config.get("visualization", {}).get("figure_dpi", 300)


def model_comparison_fig(output_dir: Path, ds: str, dpi: int):
    # Plot hold-out MAPE for all models with selection-CV markers.
    path = output_dir / f"{ds}_model_comparison.xlsx"
    if not path.exists():
        return
    df = pd.read_excel(path)
    mcol = _mcol(df, PRIMARY_MAPE, "test_mape")
    r2col = _mcol(df, PRIMARY_R2, "test_r2")
    stdcol = _mcol(df, PRIMARY_MAPE_STD, "test_mape_std")
    df = df.sort_values(mcol, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ypos = np.arange(len(df))
    xerr = df[stdcol] if stdcol in df.columns else None
    bars = ax.barh(ypos, df[mcol], xerr=xerr, color=COLORS["primary"],
                   edgecolor=COLORS["neutral"], alpha=0.85, capsize=3)
    bars[0].set_color(COLORS["accent"])
    ax.scatter(df["cv_aug_mape"], ypos, color=COLORS["danger"], zorder=3, s=45,
               label="Selection CV MAPE (+GMM)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["model"], fontsize=Y_CATEGORY_LABEL_FONTSIZE, fontweight="bold")
    ax.invert_yaxis()
    _style_axes(ax, xlabel="Nested CV MAPE (%) — real data")
    for i, (m, r2) in enumerate(zip(df[mcol], df[r2col])):
        ax.text(m + 0.1, i, f"{m:.2f}%  (R²={r2:.2f})", va="center", fontsize=12, fontweight="bold")
    leg = ax.legend(loc="lower right")
    for text in leg.get_texts():
        text.set_fontweight("bold")
    ax.set_xlim(0, df[mcol].max() * 1.35)
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds}_model_comparison.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ds}_model_comparison.pdf")


def ann_grid_fig(output_dir: Path, ds: str, dpi: int):
    # Plot top, middle, and bottom ANN configurations from the CV grid.
    path = output_dir / f"{ds}_ann_grid.xlsx"
    if not path.exists():
        return
    df = pd.read_excel(path).sort_values("mean_mape").reset_index(drop=True)
    n = len(df)
    top = df.head(5)
    mid_start = max(0, n // 2 - 2)
    middle = df.iloc[mid_start:mid_start + 5]
    bottom = df.tail(5)
    combo = pd.concat([top, middle, bottom], ignore_index=True)

    labels = []
    for _, r in combo.iterrows():
        p = json.loads(r["params"])
        arch = str(p["architecture"]).replace(" ", "")
        labels.append(f"{p['activation'][:4]} | {arch} | lr={p['learning_rate']} | dr={p['dropout_rate']}")

    colors = [COLORS["primary"]] * 5 + [COLORS["secondary"]] * len(middle) + [COLORS["danger"]] * 5
    fig, ax = plt.subplots(figsize=(10, 7))
    ypos = np.arange(len(combo))
    bars = ax.barh(ypos, combo["mean_mape"], xerr=combo["std_mape"], color=colors,
                   edgecolor=COLORS["neutral"], alpha=0.85, capsize=3)
    bars[0].set_color(COLORS["accent"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=Y_CATEGORY_LABEL_FONTSIZE, fontweight="bold")
    ax.invert_yaxis()
    _style_axes(ax, xlabel="Selection CV MAPE (%)", fontsize=Y_CATEGORY_LABEL_FONTSIZE)
    for i, (m, s) in enumerate(zip(combo["mean_mape"], combo["std_mape"])):
        ax.text(m + s + 0.1, i, f"{m:.2f}%", va="center", fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[
        Patch(facecolor=COLORS["accent"], label="Best"),
        Patch(facecolor=COLORS["primary"], label="Top 5"),
        Patch(facecolor=COLORS["secondary"], label="Middle 5"),
        Patch(facecolor=COLORS["danger"], label="Bottom 5"),
    ], loc="lower right", fontsize=9)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds}_ann_grid_bar.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ds}_ann_grid_bar.pdf")


def _short_conf_label(model: str, params_str: str) -> str:
    # Compact 'Model | hyperparameters' label from a params JSON string.
    try:
        p = json.loads(params_str) if params_str else {}
    except Exception:
        p = {}
    if model == "ANN":
        arch = str(p.get("architecture", "")).replace(" ", "")
        act = str(p.get("activation", ""))[:4]
        return f"ANN | {arch} | {act} | dr={p.get('dropout_rate', 0)}"
    if model == "KAN":
        w = str(p.get("width", "")).replace(" ", "")
        return f"KAN | w={w} | g={p.get('grid_size')} | k={p.get('spline_order')}"
    if model == "Linear":
        return "Linear | OLS"
    if model == "Polynomial":
        return f"Poly | deg={p.get('degree')} | a={p.get('alpha')}"
    if model == "SVR":
        return f"SVR | C={p.get('C')} | eps={p.get('epsilon')}"
    if model == "GaussianProcess":
        return f"GP | noise={p.get('noise')}"
    # tree ensembles
    d = p.get("max_depth")
    lr = p.get("learning_rate")
    lead = {"RandomForest": "RF", "GradientBoosting": "GB",
            "XGBoost": "XGB", "LightGBM": "LGBM"}.get(model, model)
    parts = [f"n={p.get('n_estimators')}", f"d={d}"]
    if lr is not None:
        parts.append(f"lr={lr}")
    return f"{lead} | " + " | ".join(parts)


def all_conf_bar_fig(output_dir: Path, ds: str, dpi: int):
    # Hold-out test MAPE for all ten models (best config per model).
    #
    # Uses the same metrics as Table model_comparison and the scatter plot so
    # bar chart, table, and scatter stay consistent. CV results remain in the
    # appendix (tab:patched_allconfigs / tab:unpatched_allconfigs).
    #
    path = output_dir / f"{ds}_model_comparison.xlsx"
    if not path.exists():
        return
    df = pd.read_excel(path)
    mcol = _mcol(df, PRIMARY_MAPE, "test_mape")
    r2col = _mcol(df, PRIMARY_R2, "test_r2")
    stdcol = _mcol(df, PRIMARY_MAPE_STD, "test_mape_std")
    df = df.sort_values(mcol, ascending=True).reset_index(drop=True)
    n = len(df)
    # Top 3, middle 4, bottom 3 (10 models total)
    top_n, mid_n, bot_n = 3, 4, 3
    top = df.head(top_n)
    middle = df.iloc[top_n:top_n + mid_n]
    bottom = df.tail(bot_n)
    combo = pd.concat([top, middle, bottom], ignore_index=True)

    labels = [_short_conf_label(r["model"], r["params"]) for _, r in combo.iterrows()]
    colors = ([COLORS["accent"]] + [COLORS["primary"]] * (top_n - 1)
              + [COLORS["secondary"]] * mid_n + [COLORS["danger"]] * bot_n)

    fig, ax = plt.subplots(figsize=(15, max(6, len(combo) * 0.55 + 1)))
    ypos = np.arange(len(combo))
    xerr = combo[stdcol] if stdcol in combo.columns else None
    bars = ax.barh(ypos, combo[mcol], xerr=xerr, color=colors,
                   edgecolor=COLORS["neutral"], alpha=0.85, capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=Y_CATEGORY_LABEL_FONTSIZE, fontweight="bold")
    ax.invert_yaxis()
    _style_axes(ax, xlabel="Nested CV MAPE (%) — real data")
    xmax = combo[mcol].max() * 1.35
    ax.set_xlim(0, xmax)
    for i, (_, r) in enumerate(combo.iterrows()):
        ax.text(r[mcol] + xmax * 0.01, i-0.3,
                f"{r[mcol]:.2f}%  (R\u00b2={r[r2col]:.3f})",
                va="center", fontsize=13, fontweight="bold", color=COLORS["neutral"])
    best = combo.iloc[0]
    best_name = MODEL_DISPLAY.get(best["model"], best["model"])
    ax.text(-0.4, -0.1,
            f"Best:\n {best_name}\nMAPE={best[mcol]:.2f}%\nR\u00b2={best[r2col]:.3f}",
            transform=ax.transAxes, ha="right", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["neutral"]))
    from matplotlib.patches import Patch
    legend = ax.legend(handles=[
        Patch(facecolor=COLORS["accent"], label="Best (#1)"),
        Patch(facecolor=COLORS["primary"], label="Top 3"),
        Patch(facecolor=COLORS["secondary"], label="Middle 4"),
        Patch(facecolor=COLORS["danger"], label="Bottom 3"),
    ], loc="center right", bbox_to_anchor=(1.0, 0.55), fontsize=12, framealpha=1)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds}_all_models_bar.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ds}_all_models_bar.pdf (nested CV real data, {best_name})")


def scatter_fig(output_dir: Path, ds: str, dpi: int):
    # Plot actual vs predicted failure load for the best hold-out model.
    pred_path = output_dir / f"{ds}_test_predictions.csv"
    comp_path = output_dir / f"{ds}_model_comparison.xlsx"
    if not pred_path.exists() or not comp_path.exists():
        return
    preds = pd.read_csv(pred_path)
    comp = pd.read_excel(comp_path)
    mcol = _mcol(comp, PRIMARY_MAPE, "test_mape")
    r2col = _mcol(comp, PRIMARY_R2, "test_r2")
    best = _best_model_row(comp)
    model_key = best["model"]
    # Prefer the real-data-only out-of-fold predictions when available.
    pred_col = f"pred_base_{model_key}" if f"pred_base_{model_key}" in preds.columns else f"pred_{model_key}"
    if pred_col not in preds.columns:
        print(f"  Skipped {ds}_scatter.pdf — missing column {pred_col}")
        return

    model_label = MODEL_DISPLAY.get(model_key, model_key)
    y = preds["y_true"].values
    yp = preds[pred_col].values

    lo = min(y.min(), yp.min())
    hi = max(y.max(), yp.max())
    m = (hi - lo) * 0.08
    rng = [5,35]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, yp, c=COLORS["primary"], s=80, alpha=0.8,
               edgecolors=COLORS["neutral"], label="Out-of-fold prediction (real data)")
    ax.plot(rng, rng, "k--", lw=2, alpha=0.7, label="Perfect prediction")
    ax.fill_between(rng, [v * 0.9 for v in rng], [v * 1.1 for v in rng],
                    color=COLORS["accent"], alpha=0.25, label="±10% error band")
    ax.set_xlim([5,35])
    ax.set_ylim([5,35])
    ax.set_aspect("equal", "box")
    _style_axes(ax, xlabel="Actual Failure Load (kN)", ylabel="Predicted Failure Load (kN)")
    ax.annotate(f"{model_label}\nnested CV (out-of-fold, real data)\nMAPE={best[mcol]:.2f}%\nR²={best[r2col]:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["neutral"]))
    leg = ax.legend(loc="lower right", fontsize=12)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds}_scatter.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ds}_scatter.pdf ({model_label})")


def importance_fig(output_dir: Path, ds: str, dpi: int):
    # Plot permutation-importance bars for the ANN model.
    path = output_dir / f"{ds}_feature_importance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    ann = df[df["model"] == "ANN"] if "ANN" in df["model"].values else df[df["model"] == df["model"].iloc[0]]
    ann = ann.sort_values("delta_mape", ascending=True)
    xerr = ann["delta_mape_std"] if "delta_mape_std" in ann.columns else None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ann["feature"], ann["delta_mape"], xerr=xerr, color=COLORS["secondary"],
            edgecolor=COLORS["neutral"], alpha=0.85, capsize=3)
    _style_axes(ax, xlabel="Permutation importance (Δ MAPE, %)", fontsize=Y_CATEGORY_LABEL_FONTSIZE)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(Y_CATEGORY_LABEL_FONTSIZE)
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds}_feature_importance.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ds}_feature_importance.pdf")


def main():
    # CLI entry point for publication figure generation.
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, choices=["patched", "unpatched", "both"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["data"]["output_dir"])
    dpi = _dpi(config)

    datasets = ["patched", "unpatched"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"\n{'=' * 50}\nFigures for {ds}\n{'=' * 50}")
        model_comparison_fig(output_dir, ds, dpi)
        ann_grid_fig(output_dir, ds, dpi)
        all_conf_bar_fig(output_dir, ds, dpi)
        scatter_fig(output_dir, ds, dpi)
        importance_fig(output_dir, ds, dpi)


if __name__ == "__main__":
    main()
