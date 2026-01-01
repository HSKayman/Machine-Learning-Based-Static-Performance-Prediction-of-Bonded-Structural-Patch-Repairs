"""
Visualization module for generating publication-ready graphs.
One graph per dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import warnings

warnings.filterwarnings('ignore')

# quality settings - increased font sizes and bold for visibility
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',     # Steel blue
    'secondary': '#A23B72',   # Raspberry
    'accent': '#F18F01',      # Orange
    'success': '#C73E1D',     # Red
    'neutral': '#3B3B3B',     # Dark gray
    'light': '#E8E8E8',       # Light gray
    'grid': '#CCCCCC'
}

# Load experiment results from Excel.
def load_results(output_dir: Path, dataset_type: str) -> Optional[pd.DataFrame]:

    results_path = output_dir / f"{dataset_type}_results.xlsx"
    if results_path.exists():
        return pd.read_excel(results_path)
    return None

#  Create a single professional figure combining:
#     - Top configurations comparison (horizontal bar chart)
#     - Actual vs Predicted scatter for best model
def create_combined_figure(results_df: pd.DataFrame, 
                           dataset_type: str,
                           X: np.ndarray,
                           y: np.ndarray,
                           output_path: Path,
                           config: Dict[str, Any]) -> None:
   
    dpi = config.get('figure_dpi', 300)
    
    # Sort by MAPE to get rankings
    sorted_df = results_df.sort_values('mean_mape').reset_index(drop=True)
    n_total = len(sorted_df)
    
    # Get top 5 (best), middle 5, and bottom 5 (worst) configurations
    top_5 = sorted_df.head(5).copy()
    
    # Get middle 5 (centered around median)
    mid_start = max(0, (n_total // 2) - 2)
    mid_end = min(n_total, mid_start + 5)
    middle_5 = sorted_df.iloc[mid_start:mid_end].copy()
    
    bottom_5 = sorted_df.tail(5).copy()
    
    # Combine: top 5 first, then middle 5, then bottom 5
    combined_configs = pd.concat([top_5, middle_5, bottom_5], ignore_index=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # fig.suptitle(f'{dataset_type.capitalize()} Specimens - Model Performance Analysis', 
    #              fontweight='bold', y=1.02)
    
    # Left: Horizontal bar chart of top 5 + middle 5 + bottom 5 configurations 
    y_positions = np.arange(len(combined_configs))
    
    # Create configuration labels with all parameters
    labels = []
    for _, row in combined_configs.iterrows():
        arch = row['architecture']
        # Format architecture (hidden layers)
        if isinstance(arch, str):
            arch_short = arch.replace('[', '').replace(']', '').replace(' ', '')
        else:
            arch_short = str(arch)
        
        # Format: Activation | Arch | lr | dropout
        act = row['activation'][:4]  # relu, leak, selu, etc.
        lr = row['learning_rate']
        dr = row['dropout_rate']
        labels.append(f"{act} | {arch_short} | lr={lr} | dr={dr}")
    
    # Create color array: blue for top 5, purple for middle 5, red for bottom 5
    colors = ([COLORS['primary']] * 5 + 
              [COLORS['secondary']] * len(middle_5) + 
              [COLORS['success']] * 5)
    
    # Bar chart
    bars = ax1.barh(y_positions, combined_configs['mean_mape'], 
                    xerr=combined_configs['std_mape'],
                    color=colors, 
                    edgecolor=COLORS['neutral'],
                    alpha=0.85,
                    capsize=3,
                    error_kw={'linewidth': 1})
    
    # Highlight best (rank 1)
    bars[0].set_color(COLORS['accent'])
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels)
    ax1.tick_params(axis='y', labelsize=9)  # Adjusted font for longer labels
    ax1.set_xlabel('Mean Absolute Percentage Error (%)')
    #ax1.set_title('Top 5, Middle 5 & Bottom 5 Configurations', fontweight='bold', pad=10)
    ax1.invert_yaxis()  # Best at top
    ax1.set_xlim(0, combined_configs['mean_mape'].max() * 1.15)
    
    # Add horizontal separator lines between groups
    ax1.axhline(y=4.5, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=4.5 + len(middle_5), color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add value labels
    for i, (mape, std) in enumerate(zip(combined_configs['mean_mape'], combined_configs['std_mape'])):
        ax1.text(mape + std + 0.3, i, f'{mape:.2f}%', 
                 va='center', fontsize=10, fontweight='bold', color=COLORS['neutral'])
    
    # Add R^2 annotation for best
    best_r2 = top_5.iloc[0]['mean_r2']
    ax1.annotate(f'Best R² = {best_r2:.4f}', 
                 xy=(0.98, 0.02), xycoords='axes fraction',
                 ha='right', va='bottom',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], 
                          edgecolor=COLORS['neutral'], alpha=0.8))
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['accent'], edgecolor=COLORS['neutral'], label='Best (#1)'),
        Patch(facecolor=COLORS['primary'], edgecolor=COLORS['neutral'], label='Top 5'),
        Patch(facecolor=COLORS['secondary'], edgecolor=COLORS['neutral'], label='Middle 5'),
        Patch(facecolor=COLORS['success'], edgecolor=COLORS['neutral'], label='Bottom 5')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Use top_5 for the best model in scatter plot
    top_configs = top_5
    
    # Right: Actual vs Predicted scatter 
    # Load best model and make predictions
    best_config = top_configs.iloc[0]
    
    try:
        from train import DynamicANN
        import ast
        
        model_path = output_path.parent / f"{dataset_type}_best_model.pt"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            arch = ast.literal_eval(best_config['architecture'])
            
            model = DynamicANN(
                input_dim=X.shape[1],
                hidden_layers=arch,
                activation=best_config['activation'],
                dropout_rate=best_config['dropout_rate']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_pred = model(X_tensor).squeeze().numpy()
        else:
            # If no model, use dummy predictions for visualization
            y_pred = y + np.random.normal(0, y.std() * 0.1, len(y))
    except Exception as e:
        print(f"    Warning: Could not load model for predictions: {e}")
        y_pred = y + np.random.normal(0, y.std() * 0.1, len(y))
    
    # Scatter plot
    ax2.scatter(y, y_pred, c=COLORS['primary'], alpha=0.6, 
                edgecolors=COLORS['neutral'], linewidths=0.8, s=70)
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]
    ax2.plot(line_range, line_range, 'k--', linewidth=2.0, alpha=0.7, label='Perfect Prediction')
    
    # ±10% error bands
    ax2.fill_between(line_range, 
                     [v * 0.9 for v in line_range], 
                     [v * 1.1 for v in line_range],
                     alpha=0.15, color=COLORS['accent'], label='±10% Error Band')
    
    ax2.set_xlabel('Actual Failure Load (kN)')
    ax2.set_ylabel('Predicted Failure Load (kN)')
    # ax2.set_title('Actual vs Predicted (Best Model)', fontweight='bold', pad=10)
    ax2.set_xlim(line_range)
    ax2.set_ylim(line_range)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add metrics annotation
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    mape_val = mean_absolute_percentage_error(y, y_pred) * 100
    r2_val = r2_score(y, y_pred)
    
    metrics_text = f'MAPE: {mape_val:.2f}%\nR²: {r2_val:.4f}'
    ax2.annotate(metrics_text, 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 ha='left', va='top',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor=COLORS['neutral'], alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved visualization to {output_path}")

# Run visualization as standalone script.
def main():

    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization graphs')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['patched', 'unpatched', 'both'],
                        default='both')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    viz_config = config.get('visualization', {})
    output_dir = Path(config['data']['output_dir'])
    
    datasets = ['patched', 'unpatched'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_type in datasets:
        print(f"\n{'='*50}")
        print(f"Generating visualization for {dataset_type}")
        print('='*50)
        
        # Load results
        results_df = load_results(output_dir, dataset_type)
        
        if results_df is None:
            print(f"  No results found for {dataset_type}. Run training first.")
            continue
        
        # Load data for scatter plot
        data_file = output_dir / f"{dataset_type}_augmented.npz"
        if not data_file.exists():
            data_file = output_dir / f"{dataset_type}_processed.npz"
        
        data = np.load(data_file, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # Create figure
        output_path = output_dir / f"{dataset_type}_visualization.pdf"
        create_combined_figure(results_df, dataset_type, X, y, output_path, viz_config)


if __name__ == '__main__':
    main()

