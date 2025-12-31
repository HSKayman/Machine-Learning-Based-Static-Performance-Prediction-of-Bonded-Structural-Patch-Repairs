"""
PyTorch training module with hyperparameter grid search.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
from pathlib import Path
import time
import json
import yaml
import warnings

warnings.filterwarnings('ignore')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DynamicANN(nn.Module):
    """Dynamic ANN with configurable architecture."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        """
        Initialize ANN.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes, e.g., [64, 32, 16, 8]
            activation: Activation function name
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_single_fold(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int,
                      learning_rate: float,
                      patience: int) -> Tuple[float, float, float, int]:
    """
    Train model for one fold.
    
    Returns:
        Tuple of (val_mape, val_mse, val_r2, epochs_trained)
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    
    epochs_trained = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = model(X_batch)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(y_batch.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_mse = mean_squared_error(val_targets, val_preds)
        
        epochs_trained = epoch + 1
        
        # Early stopping
        if early_stopping(val_mse, model):
            break
    
    # Restore best model
    early_stopping.restore_best(model)
    
    # Final evaluation
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            val_preds.extend(outputs.squeeze().cpu().numpy())
            val_targets.extend(y_batch.numpy())
    
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    val_mape = mean_absolute_percentage_error(val_targets, val_preds) * 100
    val_mse = mean_squared_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    return val_mape, val_mse, val_r2, epochs_trained


def train_with_kfold(X: np.ndarray,
                     y: np.ndarray,
                     hidden_layers: List[int],
                     activation: str,
                     learning_rate: float,
                     dropout_rate: float,
                     config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train model with K-Fold cross-validation.
    
    Returns:
        Results dictionary with metrics for all folds
    """
    n_folds = config.get('k_folds', 5)
    epochs = config.get('epochs', 2000)
    batch_size = config.get('batch_size', 16)
    patience = config.get('early_stopping_patience', 50)
    random_state = config.get('random_state', 42)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = DynamicANN(
            input_dim=X.shape[1],
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Train
        val_mape, val_mse, val_r2, epochs_trained = train_single_fold(
            model, train_loader, val_loader, epochs, learning_rate, patience
        )
        
        fold_results.append({
            'fold': fold + 1,
            'mape': val_mape,
            'mse': val_mse,
            'r2': val_r2,
            'epochs': epochs_trained
        })
    
    # Aggregate results
    mapes = [r['mape'] for r in fold_results]
    mses = [r['mse'] for r in fold_results]
    r2s = [r['r2'] for r in fold_results]
    epochs_list = [r['epochs'] for r in fold_results]
    
    return {
        'fold_results': fold_results,
        'mean_mape': np.mean(mapes),
        'std_mape': np.std(mapes),
        'mean_mse': np.mean(mses),
        'std_mse': np.std(mses),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s),
        'mean_epochs': np.mean(epochs_list),
        'best_fold': int(np.argmin(mapes) + 1),
        'best_mape': float(np.min(mapes))
    }


def run_grid_search(X: np.ndarray,
                    y: np.ndarray,
                    config: Dict[str, Any],
                    dataset_type: str) -> pd.DataFrame:
    """
    Run hyperparameter grid search.
    
    Args:
        X: Features
        y: Targets
        config: Training configuration
        dataset_type: 'patched' or 'unpatched'
        
    Returns:
        DataFrame with all experiment results
    """
    training_config = config.get('training', {})
    
    architectures = training_config.get('architectures', [[64, 32, 16, 8]])
    activations = training_config.get('activation_functions', ['relu'])
    learning_rates = training_config.get('learning_rates', [0.001])
    dropout_rates = training_config.get('dropout_rates', [0.0])
    
    # Generate all combinations
    combinations = list(product(architectures, activations, learning_rates, dropout_rates))
    total_experiments = len(combinations)
    
    print(f"\n  Total experiments: {total_experiments}")
    print(f"  Architectures: {len(architectures)}")
    print(f"  Activations: {len(activations)}")
    print(f"  Learning rates: {len(learning_rates)}")
    print(f"  Dropout rates: {len(dropout_rates)}")
    
    results = []
    
    for i, (arch, act, lr, dropout) in enumerate(combinations):
        print(f"\n  [{i+1}/{total_experiments}] arch={arch}, act={act}, lr={lr}, dropout={dropout}")
        
        start_time = time.time()
        
        try:
            result = train_with_kfold(
                X, y,
                hidden_layers=arch,
                activation=act,
                learning_rate=lr,
                dropout_rate=dropout,
                config=training_config
            )
            
            elapsed = time.time() - start_time
            
            results.append({
                'dataset': dataset_type,
                'architecture': str(arch),
                'n_layers': len(arch),
                'total_neurons': sum(arch),
                'activation': act,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'mean_mape': result['mean_mape'],
                'std_mape': result['std_mape'],
                'mean_mse': result['mean_mse'],
                'std_mse': result['std_mse'],
                'mean_r2': result['mean_r2'],
                'std_r2': result['std_r2'],
                'mean_epochs': result['mean_epochs'],
                'best_fold_mape': result['best_mape'],
                'training_time_sec': elapsed
            })
            
            print(f"    MAPE: {result['mean_mape']:.2f}% ± {result['std_mape']:.2f}%, "
                  f"R²: {result['mean_r2']:.4f}, Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'dataset': dataset_type,
                'architecture': str(arch),
                'n_layers': len(arch),
                'total_neurons': sum(arch),
                'activation': act,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'mean_mape': float('nan'),
                'std_mape': float('nan'),
                'mean_mse': float('nan'),
                'std_mse': float('nan'),
                'mean_r2': float('nan'),
                'std_r2': float('nan'),
                'mean_epochs': 0,
                'best_fold_mape': float('nan'),
                'training_time_sec': 0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def train_best_model(X: np.ndarray,
                     y: np.ndarray,
                     best_config: Dict[str, Any],
                     training_config: Dict[str, Any],
                     output_path: Path) -> nn.Module:
    """
    Train the best model on full data and save it.
    
    Args:
        X: Full features
        y: Full targets
        best_config: Best hyperparameters
        training_config: Training settings
        output_path: Path to save model
        
    Returns:
        Trained model
    """
    import ast
    
    # Parse architecture string back to list
    arch = ast.literal_eval(best_config['architecture'])
    
    # Create model
    model = DynamicANN(
        input_dim=X.shape[1],
        hidden_layers=arch,
        activation=best_config['activation'],
        dropout_rate=best_config['dropout_rate']
    )
    model = model.to(DEVICE)
    
    # Training setup
    epochs = training_config.get('epochs', 2000)
    batch_size = training_config.get('batch_size', 16)
    patience = training_config.get('early_stopping_patience', 50)
    lr = best_config['learning_rate']
    
    # Split for early stopping validation
    n_val = max(1, int(len(X) * 0.1))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_loss += criterion(outputs.squeeze(), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        
        if early_stopping(val_loss, model):
            break
    
    early_stopping.restore_best(model)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': best_config,
        'input_dim': X.shape[1]
    }, output_path)
    
    return model


def main():
    """Run training as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ANN models with grid search')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['patched', 'unpatched', 'both'],
                        default='both', help='Which dataset to train on')
    parser.add_argument('--use-augmented', action='store_true', default=True,
                        help='Use augmented data (default: True)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Using device: {DEVICE}")
    
    datasets = ['patched', 'unpatched'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_type in datasets:
        print(f"\n{'='*60}")
        print(f"Training on {dataset_type} dataset")
        print('='*60)
        
        output_dir = Path(config['data']['output_dir'])
        
        # Load data
        data_file = f"{dataset_type}_augmented.npz" if args.use_augmented else f"{dataset_type}_processed.npz"
        data_path = output_dir / data_file
        
        if not data_path.exists():
            # Try without augmentation
            data_path = output_dir / f"{dataset_type}_processed.npz"
        
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        
        # Run grid search
        results_df = run_grid_search(X, y, config, dataset_type)
        
        # Sort by MAPE and save
        results_df = results_df.sort_values('mean_mape', ascending=True)
        results_path = output_dir / f"{dataset_type}_results.xlsx"
        results_df.to_excel(results_path, index=False)
        print(f"\n  Saved results to {results_path}")
        
        # Print best results
        best = results_df.iloc[0]
        print(f"\n  Best configuration:")
        print(f"    Architecture: {best['architecture']}")
        print(f"    Activation: {best['activation']}")
        print(f"    Learning rate: {best['learning_rate']}")
        print(f"    Dropout: {best['dropout_rate']}")
        print(f"    MAPE: {best['mean_mape']:.2f}% ± {best['std_mape']:.2f}%")
        print(f"    R²: {best['mean_r2']:.4f}")
        
        # Train and save best model
        best_config = {
            'architecture': best['architecture'],
            'activation': best['activation'],
            'learning_rate': best['learning_rate'],
            'dropout_rate': best['dropout_rate']
        }
        
        model_path = output_dir / f"{dataset_type}_best_model.pt"
        train_best_model(X, y, best_config, config.get('training', {}), model_path)
        print(f"  Saved best model to {model_path}")


if __name__ == '__main__':
    main()

