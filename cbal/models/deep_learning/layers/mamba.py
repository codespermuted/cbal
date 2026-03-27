"""
Mamba (Selective State Space Model) block — pure PyTorch implementation.

Implements the core Mamba architecture from Gu & Dao (2023):
1. Input projection (expand)
2. Causal Conv1d
3. Selective SSM with input-dependent B, C, delta
4. Output projection

The selective scan is implemented as a sequential recurrence (O(L) per step).
For production speed, install ``mamba-ssm`` which provides fused CUDA kernels.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (arXiv 2312.00752, 2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model — the core of Mamba.

    Given input x of shape (B, L, D_inner), compute:
    - B_t, C_t from input (data-dependent, "selective")
    - delta_t (discretization step) from input
    - Run discretized SSM recurrence

    Parameters
    ----------
    d_inner : int
        Inner dimension (after expansion).
    d_state : int
        SSM state dimension N. Default 16.
    dt_rank : int or None
        Rank of delta projection. Default d_inner // 16.
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)

        # A parameter (not input-dependent, learned)
        # Initialize as log of a range for stability (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (D, N)

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections: x → (delta, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Delta projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # Initialize dt bias for stability
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_softplus = torch.log(torch.exp(dt_init) - 1)
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, D_inner)
        Returns: (B, L, D_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Project to get delta, B_input, C_input
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_input, C_input = torch.split(
            x_dbl, [self.dt_rank, N, N], dim=-1
        )

        # Delta: project and softplus for positivity
        delta = F.softplus(self.dt_proj(dt))  # (B, L, D)

        # A: from log parameterization
        A = -torch.exp(self.A_log)  # (D, N), negative for stability

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # For efficiency, compute per-step in the scan
        return self._selective_scan(x, delta, A, B_input, C_input)

    def _selective_scan(self, x, delta, A, B_input, C_input):
        """Sequential selective scan (recurrence).

        x : (B, L, D)
        delta : (B, L, D)
        A : (D, N)
        B_input : (B, L, N)
        C_input : (B, L, N)
        """
        B_batch, L, D = x.shape
        N = self.d_state
        device = x.device

        # Initialize state
        h = torch.zeros(B_batch, D, N, device=device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            # Current step
            x_t = x[:, t, :]          # (B, D)
            dt_t = delta[:, t, :]      # (B, D)
            B_t = B_input[:, t, :]     # (B, N)
            C_t = C_input[:, t, :]     # (B, N)

            # Discretize A and B for this step
            # A_bar = exp(dt * A) — (B, D, N)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
            # B_bar = dt * B — (B, D, N)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)

            # State update: h = A_bar * h + B_bar * x
            h = dA * h + dB * x_t.unsqueeze(-1)  # (B, D, N)

            # Output: y = C * h + D * x
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, D)
            y_t = y_t + self.D * x_t

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, D)


class MambaBlock(nn.Module):
    """Single Mamba block with input projection, conv, SSM, output projection.

    Architecture::

        x → Linear(D→2*D_inner) → split → [z path (gate), x path]
                                             ↓
                                         Conv1d → SiLU → SSM
                                             ↓
                                         y * SiLU(z)  ← gating
                                             ↓
                                         Linear(D_inner→D)

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension (default 16).
    expand : int
        Expansion factor for inner dimension (default 2).
    d_conv : int
        Causal conv kernel size (default 4).
    dropout : float
        Dropout rate (default 0.0).
    use_causal_conv : bool
        Whether to use causal conv (default True). Set False for TMB variant.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        use_causal_conv: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand
        self.d_inner = d_inner

        # Input projection: D → 2 * D_inner (for x and z paths)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal conv (optional — TMB removes this)
        self.use_causal_conv = use_causal_conv
        if use_causal_conv:
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=d_conv,
                padding=d_conv - 1, groups=d_inner,  # depthwise
            )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner, d_state)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # Project and split into x_path and z_path (gate)
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each (B, L, D_inner)

        # Causal conv on x_path
        if self.use_causal_conv:
            x_path = x_path.permute(0, 2, 1)  # (B, D_inner, L)
            x_path = self.conv1d(x_path)[:, :, :L]  # causal: trim right
            x_path = x_path.permute(0, 2, 1)  # (B, L, D_inner)

        x_path = F.silu(x_path)

        # Selective SSM
        y = self.ssm(x_path)  # (B, L, D_inner)

        # Gate with z path
        y = y * F.silu(z)

        # Output projection
        return self.dropout(self.out_proj(y))


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba: forward + backward scans, then combine.

    Used in S-Mamba for capturing both past→future and future→past dependencies.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 d_conv: int = 4, dropout: float = 0.0, use_causal_conv: bool = True):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, expand, d_conv, dropout, use_causal_conv)
        self.backward_mamba = MambaBlock(d_model, d_state, expand, d_conv, dropout, use_causal_conv)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        fwd = self.forward_mamba(x)
        bwd = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        return self.norm(x + fwd + bwd)
