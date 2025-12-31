"""
Synthetic data generation(basically oversampling) module using Gaussian Mixture Model (GMM).
Includes quality validation for generated samples.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
import yaml
import warnings

# Generate synthetic data using Gaussian Mixture Model.
class GMMSyntheticGenerator:
    # Initialize GMM generator.
    def __init__(self,
                 n_components: int = 3,
                 target_samples: int = 300,
                 random_state: int = 42,
                 max_retries: int = 5,
                 quality_threshold: float = 0.05):
     
        self.n_components = n_components
        self.target_samples = target_samples
        self.random_state = random_state
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        
        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.encoded_shapes: Dict[str, int] = {}
        self.regressor: Optional[RandomForestRegressor] = None
        self.original_df: Optional[pd.DataFrame] = None
    
    # Fit the GMM on the original data.   
    def fit(self,
            df: pd.DataFrame,
            float_cols: List[str],
            cat_cols: List[str],
            target_col: str) -> 'GMMSyntheticGenerator':

        self.float_cols = float_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.original_df = df.copy()
        
        # One hot encode categorical columns
        encoded_parts = []
        for col in cat_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[[col]])
            self.encoders[col] = encoder
            self.encoded_shapes[col] = encoded.shape[1]
            encoded_parts.append(encoded)
        
        # Combine with numerical columns (including target)
        numerical_data = df[float_cols + [target_col]].values
        
        if encoded_parts:
            X_full = np.hstack([*encoded_parts, numerical_data])
        else:
            X_full = numerical_data
        
        # Normalize all features for GMM
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_full)
        
        # Fit GMM with full covariance to capture correlations
        self.gmm = GaussianMixture(
            n_components=min(self.n_components, len(df) // 2),  # Don't have more components than samples/2
            random_state=self.random_state,
            covariance_type='full',
            max_iter=200
        )
        self.gmm.fit(X_scaled)
        
        # Train RandomForest regressor for target refinement
        X_reg = pd.get_dummies(df[cat_cols + float_cols])
        y_reg = df[target_col]
        
        self.regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.regressor.fit(X_reg, y_reg)
        self.regressor_columns = X_reg.columns.tolist()
        
        # Store constraints for numerical columns
        self.constraints = {}
        for col in float_cols + [target_col]:
            self.constraints[col] = (df[col].min(), df[col].max())
        
        print(f"  GMM fitted with {self.gmm.n_components} components")
        
        return self
    # Generate synthetic samples from the fitted GMM.
    def generate(self, n_samples: Optional[int] = None) -> pd.DataFrame:

        if n_samples is None:
            n_samples = max(0, self.target_samples - len(self.original_df))
        
        if n_samples <= 0:
            return pd.DataFrame(columns=self.original_df.columns)
        
        # Sample from GMM
        X_scaled, _ = self.gmm.sample(n_samples)
        X_unscaled = self.scaler.inverse_transform(X_scaled)
        
        # Decode categorical columns
        start_idx = 0
        cat_data = {}
        
        for col in self.cat_cols:
            end_idx = start_idx + self.encoded_shapes[col]
            encoded_chunk = X_unscaled[:, start_idx:end_idx]
            
            # Convert soft probabilities to hard assignments
            hard_encoded = np.zeros_like(encoded_chunk)
            hard_encoded[np.arange(len(encoded_chunk)), encoded_chunk.argmax(axis=1)] = 1
            
            # Inverse transform
            cat_data[col] = self.encoders[col].inverse_transform(hard_encoded).ravel()
            start_idx = end_idx
        
        # Extract numerical columns
        numerical_data = X_unscaled[:, start_idx:]
        all_num_cols = self.float_cols + [self.target_col]
        
        # Apply constraints (clip to original min/max)
        for i, col in enumerate(all_num_cols):
            if col in self.constraints:
                min_val, max_val = self.constraints[col]
                numerical_data[:, i] = np.clip(numerical_data[:, i], min_val, max_val)
        
        # Build DataFrame
        df_synth = pd.DataFrame(numerical_data, columns=all_num_cols)
        for col in self.cat_cols:
            df_synth[col] = cat_data[col]
        
        # Refine target using RandomForest regressor
        X_pred = pd.get_dummies(df_synth[self.cat_cols + self.float_cols])
        
        # Align columns with training data
        for col in self.regressor_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.regressor_columns]
        
        df_synth[self.target_col] = self.regressor.predict(X_pred)
        
        # Apply target constraints
        min_val, max_val = self.constraints[self.target_col]
        df_synth[self.target_col] = np.clip(df_synth[self.target_col], min_val, max_val)
        
        return df_synth
    #  Validate the quality of generated synthetic data.
    def validate_quality(self, df_synth: pd.DataFrame) -> Dict[str, Any]:
        report = {
            'passed': True,
            'n_original': len(self.original_df),
            'n_synthetic': len(df_synth),
            'ks_tests': {},
            'correlation_diff': None,
            'distribution_stats': {},
            'categorical_distribution': {}
        }
        
        all_num_cols = self.float_cols + [self.target_col]
        
        # KS-test for each numerical column
        for col in all_num_cols:
            original = self.original_df[col].values
            synthetic = df_synth[col].values
            
            stat, p_value = stats.ks_2samp(original, synthetic)
            
            passed = p_value >= self.quality_threshold
            report['ks_tests'][col] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'passed': passed
            }
            
            if not passed:
                report['passed'] = False
        
        # Correlation matrix comparison
        original_corr = self.original_df[all_num_cols].corr().values
        synthetic_corr = df_synth[all_num_cols].corr().values
        
        # Handle NaN in correlation matrices
        original_corr = np.nan_to_num(original_corr, nan=0.0)
        synthetic_corr = np.nan_to_num(synthetic_corr, nan=0.0)
        
        corr_diff = np.linalg.norm(original_corr - synthetic_corr, 'fro')
        report['correlation_diff'] = float(corr_diff)
        
        # Distribution statistics
        for col in all_num_cols:
            original = self.original_df[col]
            synthetic = df_synth[col]
            
            report['distribution_stats'][col] = {
                'original': {
                    'mean': float(original.mean()),
                    'std': float(original.std()),
                    'min': float(original.min()),
                    'max': float(original.max())
                },
                'synthetic': {
                    'mean': float(synthetic.mean()),
                    'std': float(synthetic.std()),
                    'min': float(synthetic.min()),
                    'max': float(synthetic.max())
                }
            }
        
        # Categorical distribution comparison
        for col in self.cat_cols:
            original_dist = self.original_df[col].value_counts(normalize=True).to_dict()
            synthetic_dist = df_synth[col].value_counts(normalize=True).to_dict()
            
            report['categorical_distribution'][col] = {
                'original': {str(k): float(v) for k, v in original_dist.items()},
                'synthetic': {str(k): float(v) for k, v in synthetic_dist.items()}
            }
        
        return report
    # Generate synthetic data with quality validation.
    # Retries if quality check fails.
    def generate_with_validation(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
      
        n_synthetic = max(0, self.target_samples - len(self.original_df))
        
        if n_synthetic <= 0:
            print(f"  No synthetic data needed (have {len(self.original_df)}, target {self.target_samples})")
            return pd.DataFrame(columns=self.original_df.columns), {
                'passed': True,
                'n_original': len(self.original_df),
                'n_synthetic': 0,
                'message': 'No synthetic data needed'
            }
        
        print(f"  Generating {n_synthetic} synthetic samples...")
        
        best_df = None
        best_report = None
        best_score = float('inf')
        
        for attempt in range(self.max_retries):
            # Vary random state for each attempt
            original_state = self.gmm.random_state
            self.gmm.random_state = self.random_state + attempt
            
            df_synth = self.generate(n_synthetic)
            report = self.validate_quality(df_synth)
            
            # Restore original state
            self.gmm.random_state = original_state
            
            # Score based on correlation difference (lower is better)
            score = report['correlation_diff']
            
            if report['passed']:
                print(f"  Quality check passed on attempt {attempt + 1}")
                return df_synth, report
            
            if score < best_score:
                best_score = score
                best_df = df_synth
                best_report = report
            
            print(f"  Attempt {attempt + 1}: Quality check failed (corr diff: {score:.4f})")
        
        # Return best attempt even if quality check failed
        warnings.warn(f"Quality check failed after {self.max_retries} attempts. Using best attempt.")
        best_report['warning'] = f"Used best attempt after {self.max_retries} failed quality checks"
        
        return best_df, best_report

# Generate synthetic data using GMM and validate quality.
def generate_synthetic_data(X: np.ndarray,
                            y: np.ndarray,
                            preprocessor,
                            config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    
    # Build DataFrame from preprocessed data for GMM
    df = preprocessor.inverse_transform_X(X)
    df[preprocessor.target_col] = y
    
    # Initialize generator
    generator = GMMSyntheticGenerator(
        n_components=config.get('n_components', 3),
        target_samples=config.get('target_samples', 300),
        random_state=config.get('random_state', 42),
        max_retries=config.get('max_retries', 5),
        quality_threshold=config.get('quality', {}).get('ks_test_alpha', 0.05)
    )
    
    # Fit and generate
    generator.fit(
        df,
        float_cols=preprocessor.float_cols,
        cat_cols=preprocessor.cat_cols,
        target_col=preprocessor.target_col
    )
    
    df_synth, quality_report = generator.generate_with_validation()
    
    if len(df_synth) == 0:
        # No synthetic data generated
        return X, y, np.zeros(len(X), dtype=bool), quality_report
    
    # Transform synthetic data back to preprocessed format
    X_synth, y_synth, _ = preprocessor.transform(df_synth)
    
    # Combine original and synthetic
    X_aug = np.vstack([X, X_synth])
    y_aug = np.concatenate([y, y_synth])
    
    # Create mask
    synthetic_mask = np.zeros(len(X_aug), dtype=bool)
    synthetic_mask[len(X):] = True
    
    return X_aug, y_aug, synthetic_mask, quality_report

# Run synthetic data generation as standalone script.
def main():
    import argparse
    from preprocess import DataPreprocessor
    
    parser = argparse.ArgumentParser(description='Generate synthetic data using GMM')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['patched', 'unpatched', 'both'],
                        default='both')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = ['patched', 'unpatched'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_type in datasets:
        print(f"\n{'='*60}")
        print(f"Generating synthetic data for {dataset_type} (GMM)")
        print('='*60)
        
        output_dir = Path(config['data']['output_dir'])
        
        # Load processed data
        data_path = output_dir / f"{dataset_type}_processed.npz"
        if not data_path.exists():
            print(f"  ERROR: Processed data not found at {data_path}")
            print(f"  Run preprocess.py first!")
            continue
            
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        source = data['source']
        
        # Load preprocessor
        preprocessor = DataPreprocessor.load(output_dir, dataset_type)
        
        print(f"  Original data: {len(X)} samples, {X.shape[1]} features")
        print(f"  Float columns: {preprocessor.float_cols}")
        print(f"  Categorical columns: {preprocessor.cat_cols}")
        
        # Generate synthetic data
        synthetic_config = config.get('synthetic', {})
        X_aug, y_aug, synthetic_mask, quality_report = generate_synthetic_data(
            X, y, preprocessor, synthetic_config
        )
        
        # Print quality report
        print(f"\n  Quality Report:")
        print(f"    Status: {'PASSED' if quality_report.get('passed', False) else 'FAILED'}")
        print(f"    Original samples: {quality_report.get('n_original', len(X))}")
        print(f"    Synthetic samples: {quality_report.get('n_synthetic', 0)}")
        
        if 'correlation_diff' in quality_report:
            print(f"    Correlation difference: {quality_report['correlation_diff']:.4f}")
        
        if 'ks_tests' in quality_report:
            for col, result in quality_report['ks_tests'].items():
                status = "PASS" if result['passed'] else "FAIL"
                print(f"    KS-test {col}: p={result['p_value']:.4f} [{status}]")
        
        # Create source labels
        if len(source) > 0 and source.size > 0:
            source_aug = np.concatenate([
                source,
                np.array(['synthetic'] * synthetic_mask.sum())
            ])
        else:
            source_aug = np.where(synthetic_mask, 'synthetic', 'original')
        
        # Save augmented data
        output_path = output_dir / f"{dataset_type}_augmented.npz"
        np.savez(
            output_path,
            X=X_aug,
            y=y_aug,
            source=source_aug,
            synthetic_mask=synthetic_mask
        )
        print(f"\n  Saved augmented data: {len(X_aug)} samples to {output_path}")
        
        # Save quality report
        report_path = output_dir / f"{dataset_type}_quality_report.json"
        
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(report_path, 'w') as f:
            json.dump(convert_to_serializable(quality_report), f, indent=2)
        print(f"  Saved quality report to {report_path}")
    
    print("\n" + "="*60)
    print("Synthetic data generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
