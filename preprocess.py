# Preprocessing module for the ML pipeline.
# Responsibilities:
# 1. Load raw Excel files and clean them (strip names, fix material typos).
# 2. Separate real baseline rows from synthetic rows via the Source column.
# 3. Provide a per-fold-fittable DataPreprocessor (one-hot + scaling).
# 4. Create a reproducible outer hold-out test set of real physical data.

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sources that count as *real* (non-synthetic) baseline data.
BASELINE_SOURCES = {
    "experimental",
    "fe predictions",
    "fe prediction",
    "fe",
    "calculated",
    "theoretical",
    "theory",
}

# Known data-entry typos: GMM/spreadsheet artifacts that created phantom
# material classes. All of these are really Al 6061.
MATERIAL_TYPO_FIX = {
    "Al6061": "Al 6061",
    "Al 6062": "Al 6061",
    "Al 6063": "Al 6061",
    "Al 6064": "Al 6061",
    "Al 6065": "Al 6061",
    "Al 6066": "Al 6061",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Strip column names / string cells and repair known material typos.
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Strip whitespace from all object (string) columns.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Repair phantom material classes wherever they appear.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(MATERIAL_TYPO_FIX)

    return df


class DataPreprocessor:
    # One-hot + standard-scaling transformer, fittable per CV fold.

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.ohe: Optional[OneHotEncoder] = None
        self.float_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.target_col: str = ""
        self.source_col: Optional[str] = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Column identification
    # ------------------------------------------------------------------
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str], str, Optional[str]]:
        # Detect numeric, categorical, target, and source columns.
        # Target column (failure load).
        target_candidates = [c for c in df.columns if "failure" in c.lower() or "load" in c.lower()]
        target_col = target_candidates[0] if target_candidates else df.columns[-1]

        # Source column (provenance label).
        source_candidates = [c for c in df.columns if "source" in c.lower() or "unnamed" in c.lower()]
        source_col = source_candidates[0] if source_candidates else None

        exclude = {target_col}
        if source_col:
            exclude.add(source_col)
        feature_cols = [c for c in df.columns if c not in exclude]

        float_cols, cat_cols = [], []
        for col in feature_cols:
            is_object = df[col].dtype == "object" or df[col].dtype.name == "category"
            is_lowcard_int = df[col].dtype.kind in "iu" and df[col].nunique() / len(df) < 0.05
            if is_object or is_lowcard_int:
                cat_cols.append(col)
            else:
                float_cols.append(col)
        return float_cols, cat_cols, target_col, source_col

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        # Fit scalers/encoders on the provided training dataframe.
        df = clean_dataframe(df)
        self.float_cols, self.cat_cols, self.target_col, self.source_col = self._identify_columns(df)

        if self.float_cols:
            self.scaler.fit(df[self.float_cols])

        if self.cat_cols:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(df[self.cat_cols].astype(str))

        self.feature_names = list(self.float_cols)
        if self.cat_cols:
            self.feature_names += list(self.ohe.get_feature_names_out(self.cat_cols))
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Transform a dataframe into model-ready feature and target arrays.
        df = clean_dataframe(df)
        parts = []
        if self.float_cols:
            parts.append(self.scaler.transform(df[self.float_cols]))
        if self.cat_cols:
            parts.append(self.ohe.transform(df[self.cat_cols].astype(str)))
        X = np.hstack(parts).astype(np.float32) if parts else np.empty((len(df), 0), np.float32)
        y = df[self.target_col].values.astype(np.float32)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Fit on df and return the transformed arrays.
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Persistence / reporting
    # ------------------------------------------------------------------
    def get_feature_info(self) -> Dict:
        # Return column names, encoding metadata, and feature count.
        cats = {}
        if self.ohe is not None:
            for col, classes in zip(self.cat_cols, self.ohe.categories_):
                cats[col] = list(map(str, classes))
        return {
            "dataset_type": self.dataset_type,
            "float_columns": self.float_cols,
            "categorical_columns": self.cat_cols,
            "target_column": self.target_col,
            "source_column": self.source_col,
            "encoding": "one-hot",
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "categorical_values": cats,
        }

    def save(self, path: Path) -> None:
        # Persist preprocessor state and feature metadata to disk.
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{self.dataset_type}_preprocessor.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)
        with open(path / f"{self.dataset_type}_feature_info.json", "w") as f:
            json.dump(self.get_feature_info(), f, indent=2)

    @classmethod
    def load(cls, path: Path, dataset_type: str) -> "DataPreprocessor":
        # Load a saved preprocessor from disk.
        with open(Path(path) / f"{dataset_type}_preprocessor.pkl", "rb") as f:
            state = pickle.load(f)
        obj = cls(dataset_type)
        obj.__dict__.update(state)
        return obj


# ----------------------------------------------------------------------
# Baseline extraction helpers
# ----------------------------------------------------------------------
def extract_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    # Return only the real (non-synthetic) rows, using the Source column.
    df = clean_dataframe(df)
    source_candidates = [c for c in df.columns if "source" in c.lower()]
    if not source_candidates:
        return df.reset_index(drop=True), None
    source_col = source_candidates[0]
    mask = df[source_col].astype(str).str.strip().str.lower().isin(BASELINE_SOURCES)
    return df[mask].reset_index(drop=True), source_col


def _stratify_key(df: pd.DataFrame) -> Optional[pd.Series]:
    # Build a stratification key (material x temperature) if possible.
    mat = next((c for c in df.columns if c.strip().lower() == "material"), None)
    temp = next((c for c in df.columns if "temperature" in c.lower()), None)
    if mat is None:
        return None
    key = df[mat].astype(str)
    if temp is not None:
        key = key + "|" + df[temp].astype(str)
    # Strata with a single member break stratification; fall back to material.
    if key.value_counts().min() < 2:
        key = df[mat].astype(str)
    if key.value_counts().min() < 2:
        return None
    return key


def main():
    # CLI entry point for dataset preprocessing.
    parser = argparse.ArgumentParser(description="Preprocess data for the ML pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, choices=["patched", "unpatched", "both"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pre_cfg = config.get("preprocessing", {})
    test_size = pre_cfg.get("test_size", 0.2)
    random_state = pre_cfg.get("random_state", 42)
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["patched", "unpatched"] if args.dataset == "both" else [args.dataset]

    for dataset_type in datasets:
        print(f"\n{'=' * 55}\nPreprocessing {dataset_type} dataset\n{'=' * 55}")

        raw = pd.read_excel(config["data"][dataset_type])
        raw = clean_dataframe(raw)
        print(f"  Raw rows: {len(raw)}")

        baseline, source_col = extract_baseline(raw)
        print(f"  Real baseline rows: {len(baseline)} (source column: {source_col})")
        if source_col:
            print("  Baseline provenance:")
            for k, v in baseline[source_col].value_counts().items():
                print(f"    {k}: {v}")

        # Reproducible outer hold-out test set of real physical data.
        strat = _stratify_key(baseline)
        train_df, test_df = train_test_split(
            baseline,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        print(f"  Outer split -> train: {len(train_df)}, hold-out test: {len(test_df)}")

        # Persist raw baseline splits (train.py re-fits preprocessing per fold).
        baseline.to_csv(output_dir / f"{dataset_type}_baseline.csv", index=False)
        train_df.to_csv(output_dir / f"{dataset_type}_baseline_train.csv", index=False)
        test_df.to_csv(output_dir / f"{dataset_type}_baseline_test.csv", index=False)

        # Fit a reference preprocessor on the training pool (for schema/reporting).
        pre = DataPreprocessor(dataset_type).fit(train_df)
        pre.save(output_dir)
        info = pre.get_feature_info()
        print(f"  Numeric: {info['float_columns']}")
        print(f"  Categorical: {info['categorical_columns']} -> one-hot")
        print(f"  Target: {info['target_column']}  |  Features: {info['n_features']}")


if __name__ == "__main__":
    main()
