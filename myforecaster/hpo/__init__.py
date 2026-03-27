"""
myforecaster.hpo — Hyperparameter Optimization
===============================================

Search space primitives and searchers for tuning model hyperparameters.

Quick start::

    from myforecaster.hpo import Int, Real, Categorical, tune_model

    best_config, best_score, history = tune_model(
        model_name="PatchTST",
        search_space={"d_model": Int(32, 256, log=True)},
        train_data=train_data,
        val_data=val_data,
        freq="D",
        prediction_length=14,
        num_trials=20,
    )
"""

from myforecaster.hpo.space import Categorical, Int, Real, SearchSpace, sample_config, get_defaults
from myforecaster.hpo.searcher import (
    BaseSearcher,
    BayesianSearcher,
    RandomSearcher,
    get_searcher,
)
from myforecaster.hpo.runner import tune_model, get_default_search_space

__all__ = [
    "Int", "Real", "Categorical", "SearchSpace",
    "sample_config", "get_defaults",
    "BaseSearcher", "RandomSearcher", "BayesianSearcher", "get_searcher",
    "tune_model", "get_default_search_space",
]
