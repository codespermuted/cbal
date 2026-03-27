"""
HPO Searchers — generate hyperparameter configurations to try.

Two built-in searchers:
- **RandomSearcher**: uniform random sampling (always available)
- **BayesianSearcher**: Optuna TPE sampler (requires ``pip install optuna``)

Usage::

    from myforecaster.hpo.searcher import RandomSearcher
    from myforecaster.hpo.space import Int, Real, Categorical

    space = {
        "d_model": Int(32, 256, log=True),
        "learning_rate": Real(1e-5, 1e-2, log=True),
        "dropout": Real(0.0, 0.5),
    }

    searcher = RandomSearcher(space, seed=42)
    config = searcher.suggest()
    searcher.report(config, score=0.85)
    # repeat...
"""

from __future__ import annotations

import abc
import logging
import math
import random
from typing import Any

from myforecaster.hpo.space import Categorical, Int, Real, SearchSpace, sample_config

logger = logging.getLogger(__name__)


class BaseSearcher(abc.ABC):
    """Base class for HPO searchers."""

    def __init__(self, search_space: dict[str, Any], seed: int = 0):
        self.search_space = search_space
        self.seed = seed
        self.history: list[dict] = []  # list of {config, score}

    @abc.abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Suggest next configuration to try."""
        ...

    def report(self, config: dict[str, Any], score: float):
        """Report the result of a trial."""
        self.history.append({"config": config, "score": score})

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Return config with lowest score."""
        if not self.history:
            return None
        return min(self.history, key=lambda x: x["score"])["config"]

    @property
    def best_score(self) -> float | None:
        if not self.history:
            return None
        return min(self.history, key=lambda x: x["score"])["score"]

    @property
    def n_trials(self) -> int:
        return len(self.history)


class RandomSearcher(BaseSearcher):
    """Random hyperparameter search.

    Samples configurations uniformly at random from the search space.
    Simple, embarrassingly parallel, and surprisingly competitive.

    Parameters
    ----------
    search_space : dict
        Param name → SearchSpace or fixed value.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, search_space: dict[str, Any], seed: int = 0):
        super().__init__(search_space, seed)
        self._rng = random.Random(seed)

    def suggest(self) -> dict[str, Any]:
        return sample_config(self.search_space, self._rng)


class BayesianSearcher(BaseSearcher):
    """Bayesian HPO using Optuna's TPE sampler.

    Builds a surrogate model from past trials to focus search on
    promising regions. Much more sample-efficient than random search.

    Requires: ``pip install optuna``

    Parameters
    ----------
    search_space : dict
        Param name → SearchSpace or fixed value.
    seed : int
    direction : str
        ``"minimize"`` (default) or ``"maximize"``.
    """

    def __init__(self, search_space: dict[str, Any], seed: int = 0,
                 direction: str = "minimize"):
        super().__init__(search_space, seed)
        self.direction = direction
        self._study = None
        self._init_optuna()

    def _init_optuna(self):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            self._study = optuna.create_study(
                direction=self.direction, sampler=sampler,
            )
        except ImportError:
            raise ImportError(
                "BayesianSearcher requires optuna. Install: pip install optuna"
            )

    def suggest(self) -> dict[str, Any]:
        import optuna

        trial = self._study.ask()
        config = {}
        for key, val in self.search_space.items():
            if isinstance(val, Int):
                config[key] = trial.suggest_int(
                    key, val.lower, val.upper, log=val.log,
                )
            elif isinstance(val, Real):
                config[key] = trial.suggest_float(
                    key, val.lower, val.upper, log=val.log,
                )
            elif isinstance(val, Categorical):
                config[key] = trial.suggest_categorical(key, val.choices)
            else:
                config[key] = val

        self._current_trial = trial
        return config

    def report(self, config: dict[str, Any], score: float):
        super().report(config, score)
        if hasattr(self, "_current_trial"):
            self._study.tell(self._current_trial, score)


def get_searcher(
    method: str,
    search_space: dict[str, Any],
    seed: int = 0,
) -> BaseSearcher:
    """Factory: create a searcher by name.

    Parameters
    ----------
    method : str
        ``"random"`` or ``"bayesian"`` (or ``"bayes"``/``"optuna"``).
    search_space : dict
    seed : int
    """
    method = method.lower()
    if method == "random":
        return RandomSearcher(search_space, seed=seed)
    elif method in ("bayesian", "bayes", "optuna"):
        return BayesianSearcher(search_space, seed=seed)
    else:
        raise ValueError(
            f"Unknown HPO method '{method}'. Choose 'random' or 'bayesian'."
        )
