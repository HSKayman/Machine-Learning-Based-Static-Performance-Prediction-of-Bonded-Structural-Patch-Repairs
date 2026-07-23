# Synthetic data generation (oversampling) module using a Gaussian Mixture Model (GMM).
# Includes quality validation for generated samples.

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# A numeric column with at most this many distinct baseline values is treated as
# a (near-)discrete design variable and its synthetic values are snapped.
DISCRETE_MAX_UNIQUE = 6


class GMMSyntheticGenerator:
    # Fit a GMM on baseline data and sample synthetic tabular rows.

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
        max_retries: int = 5,
        quality_threshold: float = 0.05,
        snap_discrete: bool = True,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.snap_discrete = snap_discrete

        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.encoded_shapes: Dict[str, int] = {}
        self.allowed_values: Dict[str, np.ndarray] = {}
        self.constraints: Dict[str, Tuple[float, float]] = {}
        self.original_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, float_cols: List[str], cat_cols: List[str], target_col: str):
        # Fit encoders, scaler, and GMM on the baseline training dataframe.
        self.float_cols = list(float_cols)
        self.cat_cols = list(cat_cols)
        self.target_col = target_col
        self.original_df = df.reset_index(drop=True).copy()

        encoded_parts = []
        for col in cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = enc.fit_transform(df[[col]].astype(str))
            self.encoders[col] = enc
            self.encoded_shapes[col] = encoded.shape[1]
            encoded_parts.append(encoded)

        numerical = df[self.float_cols + [target_col]].values
        X_full = np.hstack([*encoded_parts, numerical]) if encoded_parts else numerical

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_full)

        n_comp = max(1, min(self.n_components, len(df) // 2))
        # BIC scan to document how the number of GMM components was chosen.
        self.bic_scores = self._bic_scan(X_scaled, range(1, min(6, len(df) // 2 + 1)), self.random_state)
        self.selected_n_components = int(n_comp)
        self.gmm = GaussianMixture(
            n_components=n_comp,
            random_state=self.random_state,
            covariance_type="full",
            max_iter=200,
        )
        self.gmm.fit(X_scaled)

        # Physical ranges and near-discrete design grids from the baseline.
        for col in self.float_cols + [target_col]:
            self.constraints[col] = (float(df[col].min()), float(df[col].max()))
        if self.snap_discrete:
            for col in self.float_cols:
                uniques = np.sort(df[col].dropna().unique())
                if len(uniques) <= DISCRETE_MAX_UNIQUE:
                    self.allowed_values[col] = uniques.astype(float)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _bic_scan(X_scaled, components, random_state) -> Dict[int, float]:
        # BIC of the GMM for each candidate number of components (lower is better).
        scores: Dict[int, float] = {}
        for n in components:
            try:
                g = GaussianMixture(n_components=n, covariance_type="full",
                                    random_state=random_state, max_iter=200).fit(X_scaled)
                scores[int(n)] = float(g.bic(X_scaled))
            except Exception:
                continue
        return scores

    def conditional_report(self, df_synth: pd.DataFrame) -> Dict[str, Any]:
        # Per-class comparison (marginals only are not enough): within each
        # material / patch class, compare the failure-load distribution and its
        # correlation with each numeric input, so preserved conditional
        # relationships (and physical trends) can be checked, not just marginals.
        out: Dict[str, Any] = {}
        orig = self.original_df
        if self.cat_cols:
            ok = orig[self.cat_cols].astype(str).agg(" | ".join, axis=1)
            sk = df_synth[self.cat_cols].astype(str).agg(" | ".join, axis=1)
        else:
            ok = pd.Series(["all"] * len(orig)); sk = pd.Series(["all"] * len(df_synth))
        for cls in sorted(ok.unique()):
            o = orig[ok.values == cls]
            s = df_synth[sk.values == cls]
            entry: Dict[str, Any] = {"n_original": int(len(o)), "n_synthetic": int(len(s))}
            if len(o) >= 2 and len(s) >= 2:
                _, p = stats.ks_2samp(o[self.target_col].values, s[self.target_col].values)
                entry["target_ks_p"] = float(p)
                entry["target_mean_original"] = float(o[self.target_col].mean())
                entry["target_mean_synthetic"] = float(s[self.target_col].mean())
                corr = {}
                for col in self.float_cols:
                    if o[col].nunique() > 1 and s[col].nunique() > 1:
                        corr[col] = {"original": float(o[self.target_col].corr(o[col])),
                                     "synthetic": float(s[self.target_col].corr(s[col]))}
                if corr:
                    entry["target_input_corr"] = corr
            out[cls] = entry
        return out

    def _snap(self, values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        # Snap continuous values to the nearest allowed discrete design point.
        idx = np.abs(values[:, None] - allowed[None, :]).argmin(axis=1)
        return allowed[idx]

    def generate(self, n_samples: int) -> pd.DataFrame:
        # Sample n synthetic rows and decode them back to the original schema.
        if n_samples <= 0:
            return pd.DataFrame(columns=self.original_df.columns)

        X_scaled, _ = self.gmm.sample(n_samples)
        X = self.scaler.inverse_transform(X_scaled)

        # Decode categoricals (argmax -> existing classes only).
        start, cat_data = 0, {}
        for col in self.cat_cols:
            end = start + self.encoded_shapes[col]
            chunk = X[:, start:end]
            hard = np.zeros_like(chunk)
            hard[np.arange(len(chunk)), chunk.argmax(axis=1)] = 1
            cat_data[col] = self.encoders[col].inverse_transform(hard).ravel()
            start = end

        numerical = X[:, start:]
        num_cols = self.float_cols + [self.target_col]
        for i, col in enumerate(num_cols):
            lo, hi = self.constraints[col]
            numerical[:, i] = np.clip(numerical[:, i], lo, hi)
            if col in self.allowed_values:
                numerical[:, i] = self._snap(numerical[:, i], self.allowed_values[col])

        df_synth = pd.DataFrame(numerical, columns=num_cols)
        for col in self.cat_cols:
            df_synth[col] = cat_data[col]
        # Preserve original column ordering.
        return df_synth[[c for c in self.original_df.columns if c in df_synth.columns]]

    # ------------------------------------------------------------------
    def validate_quality(self, df_synth: pd.DataFrame) -> Dict[str, Any]:
        # Run KS and correlation checks comparing synthetic vs baseline data.
        report: Dict[str, Any] = {
            "passed": True,
            "n_original": len(self.original_df),
            "n_synthetic": len(df_synth),
            "ks_tests": {},
            "correlation_diff": None,
            "distribution_stats": {},
            "categorical_distribution": {},
        }
        num_cols = self.float_cols + [self.target_col]

        for col in num_cols:
            stat, p = stats.ks_2samp(self.original_df[col].values, df_synth[col].values)
            passed = p >= self.quality_threshold
            report["ks_tests"][col] = {"statistic": float(stat), "p_value": float(p), "passed": bool(passed)}
            if not passed:
                report["passed"] = False

        oc = np.nan_to_num(self.original_df[num_cols].corr().values)
        sc = np.nan_to_num(df_synth[num_cols].corr().values)
        report["correlation_diff"] = float(np.linalg.norm(oc - sc, "fro"))

        for col in num_cols:
            o, s = self.original_df[col], df_synth[col]
            report["distribution_stats"][col] = {
                "original": {"mean": float(o.mean()), "std": float(o.std()), "min": float(o.min()), "max": float(o.max())},
                "synthetic": {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())},
            }
        for col in self.cat_cols:
            report["categorical_distribution"][col] = {
                "original": {str(k): float(v) for k, v in self.original_df[col].value_counts(normalize=True).items()},
                "synthetic": {str(k): float(v) for k, v in df_synth[col].value_counts(normalize=True).items()},
            }
        return report

    def generate_with_validation(self, n_synthetic: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Generate synthetic rows, retrying until KS quality checks pass.
        if n_synthetic <= 0:
            return pd.DataFrame(columns=self.original_df.columns), {
                "passed": True, "n_original": len(self.original_df), "n_synthetic": 0,
                "message": "No synthetic data needed",
            }
        best_df, best_report, best_score = None, None, float("inf")
        base_state = self.gmm.random_state
        attempts_log = []
        for attempt in range(self.max_retries):
            seed = self.random_state + attempt
            self.gmm.random_state = seed
            df_synth = self.generate(n_synthetic)
            report = self.validate_quality(df_synth)
            self.gmm.random_state = base_state
            attempts_log.append({"attempt": attempt + 1, "seed": int(seed),
                                 "passed": bool(report["passed"]),
                                 "correlation_diff": float(report["correlation_diff"])})
            # Provenance/diagnostics common to accepted or best-effort output.
            report["n_components_selected"] = getattr(self, "selected_n_components", self.n_components)
            report["bic_scores"] = getattr(self, "bic_scores", {})
            report["conditional_checks"] = self.conditional_report(df_synth)
            if report["passed"]:
                report["accepted_on_attempt"] = attempt + 1
                report["n_attempts"] = attempt + 1
                report["seed_used"] = int(seed)
                report["rejection_rate"] = float(attempt / self.max_retries)
                report["attempts_log"] = attempts_log
                return df_synth, report
            if report["correlation_diff"] < best_score:
                best_score, best_df, best_report = report["correlation_diff"], df_synth, report
        warnings.warn(f"KS quality check not fully passed after {self.max_retries} attempts; using best attempt.")
        best_report["warning"] = f"Best of {self.max_retries} attempts (not all KS tests passed)."
        best_report["accepted_on_attempt"] = None
        best_report["n_attempts"] = self.max_retries
        best_report["rejection_rate"] = 1.0
        best_report["attempts_log"] = attempts_log
        return best_df, best_report


# ----------------------------------------------------------------------
# Fold-safe helper used by train.py
# ----------------------------------------------------------------------
def _cols(df: pd.DataFrame):
    # Reuse DataPreprocessor column detection on a raw dataframe.
    from preprocess import DataPreprocessor

    pre = DataPreprocessor("tmp")
    float_cols, cat_cols, target_col, _ = pre._identify_columns(df)
    return float_cols, cat_cols, target_col


def augment_dataframe(
    df_train: pd.DataFrame,
    target_n: int,
    config: Dict[str, Any],
    validate: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    # Fit a GMM on ``df_train`` only and return baseline+synthetic up to target_n.
    #
    # A ``Source`` column (if present) is preserved: synthetic rows are labelled
    # ``synthetic``. Returns (augmented_df, quality_report_or_None).
    #
    df_train = df_train.reset_index(drop=True).copy()
    n_synth = max(0, target_n - len(df_train))
    if n_synth == 0:
        return df_train, None

    float_cols, cat_cols, target_col = _cols(df_train)
    gen = GMMSyntheticGenerator(
        n_components=config.get("n_components", 3),
        random_state=config.get("random_state", 42),
        max_retries=config.get("max_retries", 5),
        quality_threshold=config.get("quality", {}).get("ks_test_alpha", 0.05),
        snap_discrete=config.get("snap_discrete", True),
    ).fit(df_train, float_cols, cat_cols, target_col)

    if validate:
        df_synth, report = gen.generate_with_validation(n_synth)
    else:
        df_synth, report = gen.generate(n_synth), None

    source_col = next((c for c in df_train.columns if c.lower() == "source"), None)
    if source_col:
        df_synth[source_col] = "synthetic"
    df_synth = df_synth.reindex(columns=df_train.columns)
    combined = pd.concat([df_train, df_synth], ignore_index=True)
    return combined, report


# ----------------------------------------------------------------------
# Distribution plots for the deliverable artifact
# ----------------------------------------------------------------------
def _save_distribution_plots(baseline: pd.DataFrame, synthetic: pd.DataFrame,
                             num_cols: List[str], report: Dict, path: Path) -> None:
    # Save baseline-vs-synthetic distribution comparison plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
    })

    n = len(num_cols)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), num_cols):
        ax.hist(baseline[col], bins=15, alpha=0.6, density=True, label="Baseline", color="#2E86AB")
        ax.hist(synthetic[col], bins=15, alpha=0.6, density=True, label="Synthetic", color="#C73E1D")
        ks = report["ks_tests"].get(col, {})
        ax.set_title(f"{col}\nKS p={ks.get('p_value', float('nan')):.3f}", fontweight="bold", fontsize=18)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.tick_params(axis="both", width=1.8, length=6, labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xlabel("Value", fontweight="bold", fontsize=18)
        ax.set_ylabel("Density", fontweight="bold", fontsize=18)
        leg = ax.legend(fontsize=9)
        for text in leg.get_texts():
            text.set_fontweight("bold")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _to_serializable(obj):
    # Convert numpy scalars/arrays into JSON-serializable Python types.
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    # CLI entry point for deliverable GMM augmentation.
    parser = argparse.ArgumentParser(description="Generate the deliverable augmented dataset (GMM)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, choices=["patched", "unpatched", "both"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["data"]["output_dir"])
    syn_cfg = config.get("synthetic", {})
    target_samples = syn_cfg.get("target_samples", 300)

    datasets = ["patched", "unpatched"] if args.dataset == "both" else [args.dataset]
    for dataset_type in datasets:
        print(f"\n{'=' * 60}\nDeliverable augmented dataset for {dataset_type} (GMM)\n{'=' * 60}")
        baseline_path = output_dir / f"{dataset_type}_baseline.csv"
        if not baseline_path.exists():
            print(f"  ERROR: {baseline_path} not found. Run preprocess.py first.")
            continue
        baseline = pd.read_csv(baseline_path)

        float_cols, cat_cols, target_col = _cols(baseline)
        gen = GMMSyntheticGenerator(
            n_components=syn_cfg.get("n_components", 3),
            random_state=syn_cfg.get("random_state", 42),
            max_retries=syn_cfg.get("max_retries", 5),
            quality_threshold=syn_cfg.get("quality", {}).get("ks_test_alpha", 0.05),
            snap_discrete=syn_cfg.get("snap_discrete", True),
        ).fit(baseline, float_cols, cat_cols, target_col)

        n_synth = max(0, target_samples - len(baseline))
        df_synth, report = gen.generate_with_validation(n_synth)
        print(f"  Baseline: {len(baseline)}  +  Synthetic: {len(df_synth)}  =  {len(baseline) + len(df_synth)}")
        print(f"  KS quality passed: {report.get('passed')}  |  corr diff: {report.get('correlation_diff'):.4f}")
        print(f"  GMM components selected: {report.get('n_components_selected')}  |  BIC scan: "
              f"{ {k: round(v, 1) for k, v in report.get('bic_scores', {}).items()} }")
        print(f"  Accepted on attempt {report.get('accepted_on_attempt')} of {report.get('n_attempts')} "
              f"(seed {report.get('seed_used')}, rejection rate {report.get('rejection_rate')})")
        for col, r in report.get("ks_tests", {}).items():
            print(f"    KS {col}: p={r['p_value']:.4f} [{'PASS' if r['passed'] else 'FAIL'}]")

        source_col = next((c for c in baseline.columns if c.lower() == "source"), None)
        if source_col:
            df_synth[source_col] = "synthetic"
        df_synth = df_synth.reindex(columns=baseline.columns)
        augmented = pd.concat([baseline, df_synth], ignore_index=True)
        augmented.to_csv(output_dir / f"{dataset_type}_augmented.csv", index=False)

        report_path = output_dir / f"{dataset_type}_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(_to_serializable(report), f, indent=2)

        plot_path = output_dir / f"{dataset_type}_synthetic_distributions.pdf"
        _save_distribution_plots(baseline, df_synth, float_cols + [target_col], report, plot_path)
        print(f"  Saved: {dataset_type}_augmented.csv, {report_path.name}, {plot_path.name}")


if __name__ == "__main__":
    main()
