"""
Probability distribution heads for probabilistic time series forecasting.

Each distribution takes raw network output and produces:
- ``sample()``: draw random samples
- ``log_prob()``: compute log likelihood (for training loss)
- ``quantile()``: compute specific quantiles (for prediction intervals)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionOutput(nn.Module):
    """Base class for distribution output heads."""

    n_params: int = 2  # number of distribution parameters

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, self.n_params)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project hidden state to distribution parameters.

        Parameters
        ----------
        x : Tensor (batch, seq_len, input_dim)

        Returns
        -------
        Tuple of parameter tensors, each (batch, seq_len).
        """
        params = self.proj(x)  # (batch, seq_len, n_params)
        return self._split_params(params)

    def _split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def loss(self, params: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss."""
        raise NotImplementedError

    def sample(self, params: tuple[torch.Tensor, ...], n_samples: int = 1) -> torch.Tensor:
        """Draw samples. Returns (n_samples, batch, seq_len)."""
        raise NotImplementedError

    def quantile(self, params: tuple[torch.Tensor, ...], levels: Sequence[float]) -> torch.Tensor:
        """Compute quantiles. Returns (batch, seq_len, n_quantiles)."""
        raise NotImplementedError

    def mean(self, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Point forecast (distributional mean). Returns (batch, seq_len)."""
        raise NotImplementedError


class GaussianOutput(DistributionOutput):
    """Gaussian (normal) distribution output.

    Parameters: ``mu`` (mean) and ``sigma`` (std, positive).
    """

    n_params = 2

    def _split_params(self, params):
        mu = params[..., 0]
        sigma = F.softplus(params[..., 1]) + 1e-6  # ensure positive
        return mu, sigma

    def loss(self, params, target):
        mu, sigma = params
        nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2
        return nll.mean()

    def sample(self, params, n_samples=1):
        mu, sigma = params
        eps = torch.randn((n_samples,) + mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def quantile(self, params, levels):
        mu, sigma = params
        quantiles = []
        for q in levels:
            z = torch.erfinv(torch.tensor(2.0 * q - 1.0, device=mu.device)) * math.sqrt(2.0)
            quantiles.append(mu + sigma * z)
        return torch.stack(quantiles, dim=-1)  # (batch, seq_len, n_quantiles)

    def mean(self, params):
        mu, _ = params
        return mu


class StudentTOutput(DistributionOutput):
    """Student-t distribution output.

    Parameters: ``mu``, ``sigma`` (scale), ``nu`` (degrees of freedom > 2).
    Heavier tails than Gaussian — better for financial / volatile data.
    """

    n_params = 3

    def _split_params(self, params):
        mu = params[..., 0]
        sigma = F.softplus(params[..., 1]) + 1e-6
        nu = F.softplus(params[..., 2]) + 2.01  # df > 2 for finite variance
        return mu, sigma, nu

    def loss(self, params, target):
        mu, sigma, nu = params
        z = (target - mu) / sigma
        nll = (
            torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * math.pi)
            - torch.log(sigma)
            - (nu + 1) / 2 * torch.log(1 + z**2 / nu)
        )
        return -nll.mean()

    def sample(self, params, n_samples=1):
        mu, sigma, nu = params
        # Use Gamma-Normal mixture (Student-t)
        shape = (n_samples,) + mu.shape
        chi2 = torch.distributions.Chi2(nu.unsqueeze(0).expand(shape))
        s = chi2.sample()
        z = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * z * torch.sqrt(nu.unsqueeze(0) / s)

    def mean(self, params):
        mu, _, _ = params
        return mu

    def quantile(self, params, levels):
        # Approximate via sampling
        mu, sigma, nu = params
        samples = self.sample(params, n_samples=200)  # (200, batch, seq)
        quantiles = []
        for q in levels:
            quantiles.append(torch.quantile(samples, q, dim=0))
        return torch.stack(quantiles, dim=-1)


class NegativeBinomialOutput(DistributionOutput):
    """Negative binomial distribution — for count data (demand forecasting).

    Parameters: ``mu`` (mean, positive), ``alpha`` (dispersion, positive).
    """

    n_params = 2

    def _split_params(self, params):
        mu = F.softplus(params[..., 0]) + 1e-6
        alpha = F.softplus(params[..., 1]) + 1e-6
        return mu, alpha

    def loss(self, params, target):
        mu, alpha = params
        r = 1.0 / alpha
        nll = -(
            torch.lgamma(target + r)
            - torch.lgamma(r)
            - torch.lgamma(target + 1)
            + r * torch.log(r / (r + mu))
            + target * torch.log(mu / (r + mu))
        )
        return nll.mean()

    def sample(self, params, n_samples=1):
        mu, alpha = params
        r = 1.0 / alpha
        p = r / (r + mu)
        shape = (n_samples,) + mu.shape
        dist = torch.distributions.NegativeBinomial(
            total_count=r.unsqueeze(0).expand(shape),
            probs=p.unsqueeze(0).expand(shape),
        )
        return dist.sample().float()

    def mean(self, params):
        mu, _ = params
        return mu

    def quantile(self, params, levels):
        samples = self.sample(params, n_samples=200)
        quantiles = []
        for q in levels:
            quantiles.append(torch.quantile(samples.float(), q, dim=0))
        return torch.stack(quantiles, dim=-1)


class QuantileOutput(nn.Module):
    """Direct quantile regression head.

    Instead of fitting a parametric distribution, directly predicts each
    quantile level.  Trained with pinball (quantile) loss, which is the
    proper scoring rule for quantile forecasts.

    This is often more robust than distribution-based approaches when the
    true distribution is unknown or complex (e.g., multi-modal, skewed).

    Parameters
    ----------
    input_dim : int
        Dimension of the hidden state.
    quantile_levels : tuple of float
        Quantile levels to predict. Default ``(0.1, 0.5, 0.9)``.

    Example
    -------
    ::

        head = QuantileOutput(hidden_dim, quantile_levels=(0.1, 0.5, 0.9))
        q_preds = head(hidden)          # (B, H, 3)
        loss = head.loss(q_preds, target) # scalar
    """

    def __init__(
        self,
        input_dim: int,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.n_quantiles = len(quantile_levels)
        self.proj = nn.Linear(input_dim, self.n_quantiles)
        self._register_quantile_buffer(quantile_levels)

    def _register_quantile_buffer(self, levels):
        self.register_buffer(
            "_tau", torch.tensor(levels, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden state to quantile predictions.

        Parameters
        ----------
        x : (B, H, D) or (B, D)

        Returns
        -------
        (B, H, n_quantiles) or (B, n_quantiles)
        """
        raw = self.proj(x)  # (..., n_quantiles)
        # Enforce monotonicity via cumulative softplus (ensures q0.1 <= q0.5 <= q0.9)
        if self.n_quantiles > 1:
            # First quantile is raw, subsequent are raw + softplus(delta)
            base = raw[..., :1]
            deltas = F.softplus(raw[..., 1:])
            raw = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
        return raw

    def loss(self, q_preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pinball (quantile) loss.

        Parameters
        ----------
        q_preds : (B, H, n_quantiles)
        target : (B, H) — ground truth values.

        Returns
        -------
        Scalar loss tensor.
        """
        if target.dim() == q_preds.dim() - 1:
            target = target.unsqueeze(-1)  # (B, H, 1)
        errors = target - q_preds  # (B, H, Q)
        tau = self._tau.view(1, 1, -1) if errors.dim() == 3 else self._tau.view(1, -1)
        loss = torch.where(errors >= 0, tau * errors, (tau - 1) * errors)
        return loss.mean()

    def mean(self, q_preds: torch.Tensor) -> torch.Tensor:
        """Point forecast = median quantile (0.5) if present, else middle."""
        if 0.5 in self.quantile_levels:
            idx = list(self.quantile_levels).index(0.5)
        else:
            idx = self.n_quantiles // 2
        return q_preds[..., idx]

    def quantile(self, q_preds: torch.Tensor, levels: Sequence[float]) -> dict[float, torch.Tensor]:
        """Extract quantiles from predictions.

        If a requested level matches a predicted level, returns it directly.
        Otherwise, linearly interpolates between the two nearest predicted levels.
        """
        result = {}
        pred_levels = list(self.quantile_levels)
        for q in levels:
            if q in pred_levels:
                result[q] = q_preds[..., pred_levels.index(q)]
            else:
                # Linear interpolation between nearest bracketing quantiles
                lower_idx = max(i for i, l in enumerate(pred_levels) if l <= q) if any(l <= q for l in pred_levels) else 0
                upper_idx = min(i for i, l in enumerate(pred_levels) if l >= q) if any(l >= q for l in pred_levels) else len(pred_levels) - 1
                if lower_idx == upper_idx:
                    result[q] = q_preds[..., lower_idx]
                else:
                    frac = (q - pred_levels[lower_idx]) / (pred_levels[upper_idx] - pred_levels[lower_idx])
                    result[q] = q_preds[..., lower_idx] * (1 - frac) + q_preds[..., upper_idx] * frac
        return result


DISTRIBUTION_REGISTRY = {
    "gaussian": GaussianOutput,
    "normal": GaussianOutput,
    "student_t": StudentTOutput,
    "negative_binomial": NegativeBinomialOutput,
}


def get_distribution_output(name: str, input_dim: int) -> DistributionOutput:
    """Create a distribution output head by name."""
    key = name.lower().replace("-", "_")
    if key not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {name!r}. Available: {list(DISTRIBUTION_REGISTRY.keys())}")
    return DISTRIBUTION_REGISTRY[key](input_dim)
