"""
Preprocessing module for ANN pipeline.
Handles data loading, cleaning, encoding, and scaling.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any, Optional
import yaml

# Preprocessor for patched/unpatched specimen data.
class DataPreprocessor:
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.float_cols: list = []
        self.cat_cols: list = []
        self.target_col: str = ""
        self.source_col: str = ""
        self.feature_order: list = []


    # Clean column names by stripping whitespace    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        return df
    
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[list, list, str, str]:
        # Target column (failure load)
        target_candidates = [c for c in df.columns if 'failure' in c.lower() or 'load' in c.lower()]
        target_col = target_candidates[0] if target_candidates else df.columns[-2]
        
        # Source column (for stratification)
        source_candidates = [c for c in df.columns if 'source' in c.lower() or 'unnamed' in c.lower()]
        source_col = source_candidates[0] if source_candidates else None
        
        # Feature columns (exclude target and source)
        exclude_cols = {target_col}
        if source_col:
            exclude_cols.add(source_col)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Separate categorical and numerical
        float_cols = []
        cat_cols = []
        
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                cat_cols.append(col)
            else:
                # Check if it's actually categorical (few unique values that are integers)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].dtype in ['int64', 'int32']:
                    cat_cols.append(col)
                else:
                    float_cols.append(col)
        
        return float_cols, cat_cols, target_col, source_col
   
    # Fit the preprocessor on the data
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':

        df = self._clean_column_names(df.copy())
        
        self.float_cols, self.cat_cols, self.target_col, self.source_col = \
            self._identify_columns(df)
        
        self.feature_order = self.float_cols + self.cat_cols
        
        # Fit scaler on numerical columns
        if self.float_cols:
            self.scaler.fit(df[self.float_cols])
        
        # Fit label encoders on categorical columns
        for col in self.cat_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].astype(str))
        
        return self
    
    #  Transform the data.
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        df = self._clean_column_names(df.copy())
        
        # Transform numerical features
        X_num = self.scaler.transform(df[self.float_cols]) if self.float_cols else np.empty((len(df), 0))
        
        # Transform categorical features
        X_cat_list = []
        for col in self.cat_cols:
            encoded = self.encoders[col].transform(df[col].astype(str))
            X_cat_list.append(encoded.reshape(-1, 1))
        
        X_cat = np.hstack(X_cat_list) if X_cat_list else np.empty((len(df), 0))
        
        # Combine features
        X = np.hstack([X_num, X_cat]).astype(np.float32)
        
        # Target
        y = df[self.target_col].values.astype(np.float32)
        
        # Source labels (for stratification)
        source = None
        if self.source_col and self.source_col in df.columns:
            source = df[self.source_col].values
        
        return X, y, source
    
    # Fit and transform in one step.
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self.fit(df)
        return self.transform(df)
    
    # Inverse transform features back to original scale
    def inverse_transform_X(self, X: np.ndarray) -> pd.DataFrame:

        n_num = len(self.float_cols)
        n_cat = len(self.cat_cols)
        
        # Inverse transform numerical
        X_num = X[:, :n_num]
        if n_num > 0:
            X_num = self.scaler.inverse_transform(X_num)
        
        # Inverse transform categorical
        X_cat_list = []
        for i, col in enumerate(self.cat_cols):
            encoded = X[:, n_num + i].astype(int)
            decoded = self.encoders[col].inverse_transform(encoded)
            X_cat_list.append(decoded.reshape(-1, 1))
        
        # Build DataFrame
        df_num = pd.DataFrame(X_num, columns=self.float_cols)
        
        if X_cat_list:
            X_cat = np.hstack(X_cat_list)
            df_cat = pd.DataFrame(X_cat, columns=self.cat_cols)
            return pd.concat([df_num, df_cat], axis=1)
        
        return df_num
    # Get information about features for reporting.
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'dataset_type': self.dataset_type,
            'float_columns': self.float_cols,
            'categorical_columns': self.cat_cols,
            'target_column': self.target_col,
            'source_column': self.source_col,
            'feature_order': self.feature_order,
            'n_features': len(self.feature_order),
            'categorical_mappings': {
                col: dict(zip(range(len(enc.classes_)), enc.classes_.tolist()))
                for col, enc in self.encoders.items()
            }
        }
    # Save preprocessor to disk
    def save(self, path: Path) -> None:

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn objects
        with open(path / f'{self.dataset_type}_preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'encoders': self.encoders,
                'float_cols': self.float_cols,
                'cat_cols': self.cat_cols,
                'target_col': self.target_col,
                'source_col': self.source_col,
                'feature_order': self.feature_order,
                'dataset_type': self.dataset_type
            }, f)
        
        # Save feature info as JSON for reference
        with open(path / f'{self.dataset_type}_feature_info.json', 'w') as f:
            json.dump(self.get_feature_info(), f, indent=2)
    
    # Load preprocessor from disk
    @classmethod
    def load(cls, path: Path, dataset_type: str) -> 'DataPreprocessor':
        path = Path(path)
        
        with open(path / f'{dataset_type}_preprocessor.pkl', 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(dataset_type)
        preprocessor.scaler = data['scaler']
        preprocessor.encoders = data['encoders']
        preprocessor.float_cols = data['float_cols']
        preprocessor.cat_cols = data['cat_cols']
        preprocessor.target_col = data['target_col']
        preprocessor.source_col = data['source_col']
        preprocessor.feature_order = data['feature_order']
        
        return preprocessor

# Load data from Excel and preprocess
def load_and_preprocess(data_path: str, dataset_type: str, output_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], DataPreprocessor]:

    # Load data
    df = pd.read_excel(data_path)
    print(f"Loaded {dataset_type} data: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Preprocess
    preprocessor = DataPreprocessor(dataset_type)
    X, y, source = preprocessor.fit_transform(df)
    
    # Save preprocessor
    output_path = Path(output_dir)
    preprocessor.save(output_path)
    
    # Print summary
    info = preprocessor.get_feature_info()
    print(f"Numerical features: {info['float_columns']}")
    print(f"Categorical features: {info['categorical_columns']}")
    print(f"Target: {info['target_column']}")
    print(f"Total features: {info['n_features']}")
    
    return X, y, source, preprocessor

# Run preprocessing as standalone script
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for ANN training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['patched', 'unpatched', 'both'],
                        default='both')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = ['patched', 'unpatched'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_type in datasets:
        print(f"\n{'='*50}")
        print(f"Preprocessing {dataset_type} dataset")
        print('='*50)
        
        data_path = config['data'][dataset_type]
        output_dir = config['data']['output_dir']
        
        X, y, source, preprocessor = load_and_preprocess(data_path, dataset_type, output_dir)
        
        # Save processed data
        np.savez(
            f"{output_dir}/{dataset_type}_processed.npz",
            X=X, y=y, source=source if source is not None else np.array([])
        )
        
        print(f"Saved processed data to {output_dir}/{dataset_type}_processed.npz")


if __name__ == '__main__':
    main()

