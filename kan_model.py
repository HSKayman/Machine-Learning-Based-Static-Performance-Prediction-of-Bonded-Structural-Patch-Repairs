# KAN (Kolmogorov-Arnold Network) wrapper around the pykan library.
# pykan provides numerically stable B-spline computation, grid refinement, and
# spline regularisation. This thin wrapper keeps a simple constructor signature.

import torch.nn as nn
from kan import KAN as _PyKAN


class KAN(nn.Module):
    # Kolmogorov-Arnold Network (thin wrapper around pykan).

    def __init__(self, width, grid_size=5, spline_order=3, seed=42):
        # Build a pykan network with the requested width and spline settings.
        super().__init__()
        self.width = width
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.net = _PyKAN(width=width, grid=grid_size, k=spline_order, seed=seed)

    def forward(self, x):
        # Forward pass through the wrapped pykan model.
        return self.net(x)
