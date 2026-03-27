"""
Search space primitives for hyperparameter optimization.

Usage::

    from myforecaster.hpo import Int, Real, Categorical, space

    # Define search space for a model
    hp_space = {
        "d_model": Int(32, 256, log=True),
        "learning_rate": Real(1e-5, 1e-2, log=True),
        "dropout": Real(0.0, 0.5),
        "n_layers": Int(1, 6),
        "activation": Categorical("relu", "gelu", "silu"),
    }

    # Sample a configuration
    config = space.sample(hp_space)
    # → {"d_model": 128, "learning_rate": 0.001, "dropout": 0.2, ...}

Inspired by AutoGluon's ag.space and Optuna's distributions.
"""

from __future__ import annotations

import abc
import math
import random
from typing import Any, Sequence


class SearchSpace(abc.ABC):
    """Base class for hyperparameter search spaces."""

    @abc.abstractmethod
    def sample(self, rng: random.Random | None = None) -> Any:
        """Sample a random value from this space."""
        ...

    @abc.abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if value is within this space."""
        ...

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Serialize to dict for logging."""
        ...


class Int(SearchSpace):
    """Integer hyperparameter space.

    Parameters
    ----------
    lower : int
        Lower bound (inclusive).
    upper : int
        Upper bound (inclusive).
    log : bool
        If True, sample in log space (useful for parameters like d_model
        where doubling matters more than adding).
    default : int or None
        Default value for non-HPO mode.
    """

    def __init__(self, lower: int, upper: int, log: bool = False,
                 default: int | None = None):
        if lower > upper:
            raise ValueError(f"lower ({lower}) > upper ({upper})")
        self.lower = lower
        self.upper = upper
        self.log = log
        self.default = default or lower

    def sample(self, rng: random.Random | None = None) -> int:
        rng = rng or random.Random()
        if self.log:
            log_low = math.log(max(self.lower, 1))
            log_high = math.log(self.upper)
            return int(round(math.exp(rng.uniform(log_low, log_high))))
        return rng.randint(self.lower, self.upper)

    def contains(self, value: Any) -> bool:
        return isinstance(value, int) and self.lower <= value <= self.upper

    def to_dict(self) -> dict:
        return {"type": "Int", "lower": self.lower, "upper": self.upper,
                "log": self.log}

    def __repr__(self):
        return f"Int({self.lower}, {self.upper}, log={self.log})"


class Real(SearchSpace):
    """Continuous hyperparameter space.

    Parameters
    ----------
    lower : float
        Lower bound (inclusive).
    upper : float
        Upper bound (inclusive).
    log : bool
        If True, sample in log space.
    default : float or None
    """

    def __init__(self, lower: float, upper: float, log: bool = False,
                 default: float | None = None):
        if lower > upper:
            raise ValueError(f"lower ({lower}) > upper ({upper})")
        self.lower = lower
        self.upper = upper
        self.log = log
        self.default = default or lower

    def sample(self, rng: random.Random | None = None) -> float:
        rng = rng or random.Random()
        if self.log:
            log_low = math.log(max(self.lower, 1e-12))
            log_high = math.log(self.upper)
            return math.exp(rng.uniform(log_low, log_high))
        return rng.uniform(self.lower, self.upper)

    def contains(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and self.lower <= value <= self.upper

    def to_dict(self) -> dict:
        return {"type": "Real", "lower": self.lower, "upper": self.upper,
                "log": self.log}

    def __repr__(self):
        return f"Real({self.lower}, {self.upper}, log={self.log})"


class Categorical(SearchSpace):
    """Categorical hyperparameter space.

    Parameters
    ----------
    *choices
        Possible values.
    default
        Default value.
    """

    def __init__(self, *choices, default=None):
        if len(choices) == 0:
            raise ValueError("Categorical needs at least one choice")
        self.choices = list(choices)
        self.default = default or choices[0]

    def sample(self, rng: random.Random | None = None) -> Any:
        rng = rng or random.Random()
        return rng.choice(self.choices)

    def contains(self, value: Any) -> bool:
        return value in self.choices

    def to_dict(self) -> dict:
        return {"type": "Categorical", "choices": self.choices}

    def __repr__(self):
        return f"Categorical({self.choices})"


# ---------------------------------------------------------------------------
# Utility: sample a full config dict
# ---------------------------------------------------------------------------

def sample_config(
    space: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Sample a concrete config from a search space dict.

    Keys mapped to SearchSpace objects are sampled; other values pass through.

    Parameters
    ----------
    space : dict
        Mapping of param_name → SearchSpace | fixed_value.
    rng : random.Random, optional

    Returns
    -------
    dict
        Concrete hyperparameter configuration.
    """
    rng = rng or random.Random()
    config = {}
    for key, val in space.items():
        if isinstance(val, SearchSpace):
            config[key] = val.sample(rng)
        else:
            config[key] = val
    return config


def get_defaults(space: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from a search space dict."""
    config = {}
    for key, val in space.items():
        if isinstance(val, SearchSpace):
            config[key] = val.default
        else:
            config[key] = val
    return config
