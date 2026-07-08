# Generate pictorial representations of the ANN architectures used in the study.
# Loads saved .pt model checkpoints and produces publication-quality diagrams.
#
# Usage:
#     source ~/venv/bin/activate
#     python visualize_architecture.py

import ast
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from typing import List, Dict, Any, Optional


class DynamicANN(nn.Module):
    # Feed-forward ANN used only for architecture visualization.

    def __init__(self, input_dim: int, hidden_layers: List[int],
                 activation: str = 'relu', dropout_rate: float = 0.0):
        # Build the layer stack from input size, hidden topology, and activation.
        super().__init__()
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        # Return the torch activation module for a named activation.
        activations = {
            'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(), 'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(), 'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the sequential network.
        return self.network(x)


ACTIVATION_LABELS = {
    'relu': 'ReLU', 'leaky_relu': 'Leaky ReLU', 'elu': 'ELU',
    'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'selu': 'SELU', 'gelu': 'GELU'
}

LAYER_COLORS = {
    'input': '#4CAF50',
    'hidden': '#2196F3',
    'output': '#FF5722',
    'activation': '#FFC107',
    'dropout': '#9E9E9E',
}


def load_model_info(model_path: str, preprocessor_path: str) -> Dict[str, Any]:
    # Load model checkpoint and preprocessor to extract full architecture info.
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    input_dim = ckpt['input_dim']
    arch = ast.literal_eval(config['architecture'])

    with open(preprocessor_path, 'rb') as f:
        prep = pickle.load(f)

    model = DynamicANN(input_dim, arch, config['activation'], config['dropout_rate'])
    model.load_state_dict(ckpt['model_state_dict'])

    return {
        'model': model,
        'config': config,
        'input_dim': input_dim,
        'hidden_layers': arch,
        'activation': config['activation'],
        'dropout_rate': config['dropout_rate'],
        'learning_rate': config['learning_rate'],
        'feature_names': prep.get('feature_order', []),
        'target_col': prep.get('target_col', 'Output'),
        'state_dict': ckpt['model_state_dict'],
    }


def draw_neuron_diagram(info: Dict[str, Any], title: str, save_path: str):
    # Draw a neuron-level architecture diagram with connections between layers.
    hidden = info['hidden_layers']
    features = info['feature_names']
    activation = info['activation']
    dropout = info['dropout_rate']
    target = info['target_col']
    input_dim = info['input_dim']

    all_layer_sizes = [input_dim] + hidden + [1]
    n_layers = len(all_layer_sizes)
    max_neurons = max(all_layer_sizes)

    max_draw = 14
    fig_width = 3.5 + n_layers * 2.5
    fig_height = max(7, min(max_draw, max_neurons) * 0.55 + 4)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    x_positions = np.linspace(0.1, 0.9, n_layers)
    neuron_radius = min(0.016, 0.35 / max(max_draw, 6))
    neuron_spacing = neuron_radius * 3.0

    y_top = 0.88
    y_bot = 0.12

    layer_neuron_positions = []

    for layer_idx, (n_neurons, x) in enumerate(zip(all_layer_sizes, x_positions)):
        draw_n = min(n_neurons, max_draw)
        truncated = n_neurons > max_draw

        if truncated:
            show_top = max_draw // 2
            show_bot = max_draw - show_top
            yt = np.linspace(y_top, y_top - (show_top - 1) * neuron_spacing, show_top)
            yb = np.linspace(y_bot + (show_bot - 1) * neuron_spacing, y_bot, show_bot)
            y_vals = list(yt) + list(yb)
        else:
            total_h = (draw_n - 1) * neuron_spacing
            y_center = (y_top + y_bot) / 2
            y_vals = np.linspace(y_center + total_h / 2,
                                 y_center - total_h / 2, draw_n).tolist()

        if layer_idx == 0:
            color = LAYER_COLORS['input']
        elif layer_idx == n_layers - 1:
            color = LAYER_COLORS['output']
        else:
            color = LAYER_COLORS['hidden']

        positions = []
        for i, y in enumerate(y_vals):
            if truncated and i == show_top:
                mid_y = (yt[-1] + yb[0]) / 2
                ax.text(x, mid_y, '⋮', fontsize=16, ha='center', va='center',
                        transform=ax.transAxes, fontweight='bold', color='#666')

            circle = plt.Circle((x, y), neuron_radius, transform=ax.transAxes,
                                color=color, ec='white', linewidth=1.5,
                                zorder=10, alpha=0.92)
            ax.add_patch(circle)
            positions.append((x, y))

        layer_neuron_positions.append(positions)

        if layer_idx == 0:
            layer_label = 'Input'
            for fi, fname in enumerate(features):
                short = fname.replace(' (mm)', '').replace(' (F)', ' (°F)')
                ax.text(x - neuron_radius - 0.015, y_vals[fi], short,
                        fontsize=6.5, ha='right', va='center',
                        transform=ax.transAxes, color='#333')
        elif layer_idx == n_layers - 1:
            layer_label = 'Output'
            ax.text(x + neuron_radius + 0.015, y_vals[0], target,
                    fontsize=7, ha='left', va='center',
                    transform=ax.transAxes, color='#333', fontweight='bold')
        else:
            layer_label = f'Hidden {layer_idx}'

        ax.text(x, 0.04, f'{layer_label}\n({n_neurons})', fontsize=8,
                ha='center', va='center', transform=ax.transAxes,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          alpha=0.15, edgecolor=color, linewidth=1.2))

    for l in range(n_layers - 1):
        src_pos = layer_neuron_positions[l]
        dst_pos = layer_neuron_positions[l + 1]
        n_conn = len(src_pos) * len(dst_pos)
        alpha = max(0.06, min(0.45, 30.0 / n_conn))

        for sx, sy in src_pos:
            for dx, dy in dst_pos:
                ax.annotate('', xy=(dx - neuron_radius * 1.1, dy),
                            xytext=(sx + neuron_radius * 1.1, sy),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-', color='#546E7A',
                                            lw=1.2, alpha=alpha))

    act_label = ACTIVATION_LABELS.get(activation, activation)
    param_count = sum(p.numel() for p in info['model'].parameters())
    subtitle_parts = [f'Activation: {act_label}']
    if dropout > 0:
        subtitle_parts.append(f'Dropout: {dropout}')
    subtitle_parts.append(f'LR: {info["learning_rate"]}')
    subtitle_parts.append(f'Parameters: {param_count:,}')
    subtitle = '   |   '.join(subtitle_parts)

    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold', pad=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved neuron diagram: {save_path}")


def draw_block_diagram(info: Dict[str, Any], title: str, save_path: str):
    # Draw a block/box-style architecture diagram (layer-by-layer).
    hidden = info['hidden_layers']
    features = info['feature_names']
    activation = info['activation']
    dropout = info['dropout_rate']
    target = info['target_col']
    input_dim = info['input_dim']
    act_label = ACTIVATION_LABELS.get(activation, activation)

    blocks = []
    blocks.append({
        'type': 'input',
        'label': f'Input Layer\n{input_dim} features',
        'detail_right': '\n'.join(features) if features else None,
        'detail_below': None,
        'color': LAYER_COLORS['input'],
    })

    for i, h in enumerate(hidden):
        in_dim = hidden[i - 1] if i > 0 else input_dim
        blocks.append({
            'type': 'dense',
            'label': f'Dense({h})',
            'detail_right': f'{in_dim} → {h}',
            'detail_below': None,
            'color': LAYER_COLORS['hidden'],
        })
        blocks.append({
            'type': 'activation',
            'label': act_label,
            'detail_right': None,
            'detail_below': None,
            'color': LAYER_COLORS['activation'],
        })
        if dropout > 0:
            blocks.append({
                'type': 'dropout',
                'label': f'Dropout({dropout})',
                'detail_right': None,
                'detail_below': None,
                'color': LAYER_COLORS['dropout'],
            })

    blocks.append({
        'type': 'output',
        'label': f'Output Layer\nDense(1)',
        'detail_right': f'{hidden[-1]} → 1',
        'detail_below': target,
        'color': LAYER_COLORS['output'],
    })

    n_blocks = len(blocks)
    box_height_big = 0.7
    box_height_small = 0.4
    gap = 0.35
    box_width = 0.55
    x_center = 0.5

    total_h = 0
    for b in blocks:
        bh = box_height_small if b['type'] in ('activation', 'dropout') else box_height_big
        total_h += bh + gap
    total_h -= gap

    fig_height = total_h + 2.5
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_cursor = fig_height - 1.3

    y_positions_abs = []
    bh_list = []
    for b in blocks:
        bh = box_height_small if b['type'] in ('activation', 'dropout') else box_height_big
        y_positions_abs.append(y_cursor)
        bh_list.append(bh)
        y_cursor -= (bh + gap)

    for i, (block, y, bh) in enumerate(zip(blocks, y_positions_abs, bh_list)):
        is_small = block['type'] in ('activation', 'dropout')
        bw = box_width * 0.65 if is_small else box_width

        rect = FancyBboxPatch(
            (x_center - bw / 2, y - bh / 2),
            bw, bh,
            boxstyle="round,pad=0.06",
            facecolor=block['color'],
            edgecolor='white',
            alpha=0.9,
            linewidth=2,
            zorder=5
        )
        ax.add_patch(rect)

        ax.text(x_center, y, block['label'],
                fontsize=10 if not is_small else 9,
                ha='center', va='center',
                fontweight='bold', color='white', zorder=6)

        if block['detail_right']:
            ax.text(x_center + bw / 2 + 0.04, y, block['detail_right'],
                    fontsize=7.5, ha='left', va='center', color='#444',
                    style='italic', zorder=6)

        if block['detail_below']:
            ax.text(x_center, y - bh / 2 - 0.1, block['detail_below'],
                    fontsize=8, ha='center', va='top', color='#555',
                    fontweight='bold', zorder=6)

        if i < n_blocks - 1:
            next_y = y_positions_abs[i + 1]
            next_bh = bh_list[i + 1]
            ax.annotate('',
                        xy=(x_center, next_y + next_bh / 2 + 0.03),
                        xytext=(x_center, y - bh / 2 - 0.03),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                        lw=1.8, mutation_scale=15))

    param_count = sum(p.numel() for p in info['model'].parameters())
    fig_title = (f'{title}\n'
                 f'LR = {info["learning_rate"]}  |  Parameters: {param_count:,}')
    ax.set_title(fig_title, fontsize=13, fontweight='bold', pad=12)

    y_min = y_positions_abs[-1] - 1.2
    y_max = y_positions_abs[0] + 1.0
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved block diagram: {save_path}")


def draw_torchviz_graph(info: Dict[str, Any], title: str, save_path: str):
    # Generate computation graph via torchviz (requires graphviz system package).
    try:
        from torchviz import make_dot
        model = info['model']
        model.eval()
        x = torch.randn(1, info['input_dim'])
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()),
                       show_attrs=False, show_saved=False)
        dot.attr(label=title, fontsize='14')
        dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
        print(f"  Saved torchviz graph: {save_path}")
    except Exception as e:
        print(f"  torchviz graph skipped: {e}")
        print("  (Install system graphviz: sudo apt-get install graphviz)")


def print_model_summary(info: Dict[str, Any], title: str):
    # Print a text summary of the model architecture.
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Input features ({info['input_dim']}): {info['feature_names']}")
    print(f"  Hidden layers: {info['hidden_layers']}")
    print(f"  Activation: {ACTIVATION_LABELS.get(info['activation'], info['activation'])}")
    print(f"  Dropout: {info['dropout_rate']}")
    print(f"  Learning rate: {info['learning_rate']}")
    print(f"  Target: {info['target_col']}")

    param_count = sum(p.numel() for p in info['model'].parameters())
    print(f"  Total parameters: {param_count:,}")
    print(f"\n  Layer details:")
    for name, param in info['model'].named_parameters():
        print(f"    {name}: {list(param.shape)}")
    print()


def main():
    # CLI entry point for ANN architecture diagram generation.
    output_dir = Path('outputs')
    arch_dir = output_dir / 'architecture_diagrams'
    arch_dir.mkdir(parents=True, exist_ok=True)

    models = {
        'patched': {
            'model_path': output_dir / 'patched_best_model.pt',
            'preprocessor_path': output_dir / 'patched_preprocessor.pkl',
            'title': 'Patched Specimen ANN Architecture',
        },
        'unpatched': {
            'model_path': output_dir / 'unpatched_best_model.pt',
            'preprocessor_path': output_dir / 'unpatched_preprocessor.pkl',
            'title': 'Unpatched Specimen ANN Architecture',
        },
    }

    for name, paths in models.items():
        if not paths['model_path'].exists():
            print(f"  Model not found: {paths['model_path']}")
            continue

        info = load_model_info(str(paths['model_path']),
                               str(paths['preprocessor_path']))
        print_model_summary(info, paths['title'])

        draw_neuron_diagram(info, paths['title'],
                            str(arch_dir / f'{name}_neuron_diagram.png'))
        draw_block_diagram(info, paths['title'],
                           str(arch_dir / f'{name}_block_diagram.png'))
        draw_torchviz_graph(info, paths['title'],
                            str(arch_dir / f'{name}_computation_graph.png'))

    print(f"\nAll diagrams saved to: {arch_dir.resolve()}")


if __name__ == '__main__':
    main()
