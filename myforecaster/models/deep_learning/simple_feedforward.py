"""
SimpleFeedForward — Simple MLP-based probabilistic forecasting model.

A multi-layer perceptron that predicts the parameters of a probability
distribution for each forecast horizon step.  The default distribution
is Student-t (df, loc, scale), providing heavier tails than Gaussian.

This is the simplest deep learning baseline in AutoGluon.  It flattens
the context window into a single vector and passes it through an MLP.

Reference: GluonTS ``SimpleFeedForwardEstimator`` (PyTorch backend).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from myforecaster.models.deep_learning.base import AbstractDLModel


class SimpleFeedForwardNetwork(nn.Module):
    """MLP that maps context → distribution parameters per horizon step.

    Architecture:
        context (B, C) → Linear → ReLU → ... → Linear → (B, H, n_params)

    For Student-t:  n_params = 3 (df, loc, scale)
    For Gaussian:   n_params = 2 (loc, scale)
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dims: list[int] = (40, 40),
        dropout_rate: float = 0.1,
        distribution: str = "student_t",
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distribution = distribution
        self.n_params = 3 if distribution == "student_t" else 2

        layers = []
        in_dim = context_length
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, prediction_length * self.n_params))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, context_length) — flattened past values
        Returns: (B, prediction_length, n_params)
        """
        out = self.mlp(x)  # (B, H * n_params)
        out = out.view(-1, self.prediction_length, self.n_params)
        return out

    def sample(self, params: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Sample from the predicted distribution.

        params: (B, H, n_params)
        Returns: (B, num_samples, H)
        """
        if self.distribution == "student_t":
            df = 2.0 + F.softplus(params[..., 0])      # df > 2
            loc = params[..., 1]
            scale = 0.01 + F.softplus(params[..., 2])   # scale > 0
            dist = torch.distributions.StudentT(df, loc, scale)
        else:
            loc = params[..., 0]
            scale = 0.01 + F.softplus(params[..., 1])
            dist = torch.distributions.Normal(loc, scale)
        # (B, H) → expand → (B, num_samples, H)
        samples = dist.rsample((num_samples,))  # (num_samples, B, H)
        return samples.permute(1, 0, 2)  # (B, num_samples, H)

    def mean(self, params: torch.Tensor) -> torch.Tensor:
        """Extract the mean (loc) from distribution parameters."""
        if self.distribution == "student_t":
            return params[..., 1]
        return params[..., 0]


class SimpleFeedForwardModel(AbstractDLModel):
    """SimpleFeedForward: MLP-based probabilistic forecasting.

    Matches the AutoGluon/GluonTS ``SimpleFeedForwardModel``.

    Other Parameters
    ----------------
    hidden_dims : list of int
        Hidden layer sizes (default ``[40, 40]``).
    dropout_rate : float
        Dropout probability (default ``0.1``).
    distribution : str
        ``"student_t"`` (default) or ``"normal"``.
    num_samples : int
        Number of samples for quantile estimation (default ``100``).
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "hidden_dims": [40, 40],
        "dropout_rate": 0.1,
        "distribution": "student_t",
        "num_samples": 100,
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "loss_type": "mse",            # "mse", "nll", or "quantile"
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from myforecaster.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return SimpleFeedForwardNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            hidden_dims=self.get_hyperparameter("hidden_dims"),
            dropout_rate=self.get_hyperparameter("dropout_rate"),
            distribution=self.get_hyperparameter("distribution"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)        # (B, C)
        future = batch["future_target"]    # (B, H)

        if self._quantile_head is not None:
            params = self._network(past)       # (B, H, n_params)
            pred = self._network.mean(params)  # (B, H)
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        params = self._network(past)       # (B, H, n_params)
        dist_name = self.get_hyperparameter("distribution")

        if dist_name == "student_t":
            df = 2.0 + F.softplus(params[..., 0])
            loc = params[..., 1]
            scale = 0.01 + F.softplus(params[..., 2])
            dist = torch.distributions.StudentT(df, loc, scale)
        else:
            loc = params[..., 0]
            scale = 0.01 + F.softplus(params[..., 1])
            dist = torch.distributions.Normal(loc, scale)

        # Negative log-likelihood
        loss = -dist.log_prob(future).mean()
        return loss

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        past = self._enrich_target(batch)        # (B, C)
        params = self._network(past)       # (B, H, n_params)

        if self._quantile_head is not None:
            pred = self._network.mean(params)  # (B, H)
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        mean = self._network.mean(params)  # (B, H)
        num_samples = self.get_hyperparameter("num_samples")
        samples = self._network.sample(params, num_samples)  # (B, S, H)

        result = {"mean": mean, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = torch.quantile(samples, q, dim=1)
        return result
