"""Deep learning models (PyTorch) — DLinear, DeepAR, and more.

These models require PyTorch. Install with::

    pip install 'myforecaster[deep]'

Imports are lazy — torch is only loaded when you actually use a DL model.
"""


def __getattr__(name):
    _exports = {
        "AbstractDLModel": "myforecaster.models.deep_learning.base",
        "DLinearModel": "myforecaster.models.deep_learning.dlinear",
        "DeepARModel": "myforecaster.models.deep_learning.deepar",
        "PatchTSTModel": "myforecaster.models.deep_learning.patchtst",
        "TFTModel": "myforecaster.models.deep_learning.tft",
        "iTransformerModel": "myforecaster.models.deep_learning.itransformer",
        "SMambaModel": "myforecaster.models.deep_learning.s_mamba",
        "MambaTSModel": "myforecaster.models.deep_learning.mambats",
        "NHiTSModel": "myforecaster.models.deep_learning.nhits",
        "TSMixerModel": "myforecaster.models.deep_learning.tsmixer",
        "SegRNNModel": "myforecaster.models.deep_learning.segrnn",
        "TimeMixerModel": "myforecaster.models.deep_learning.timemixer",
        "TimesNetModel": "myforecaster.models.deep_learning.timesnet",
        "ModernTCNModel": "myforecaster.models.deep_learning.moderntcn",
        "MTGNNModel": "myforecaster.models.deep_learning.mtgnn",
        "CrossGNNModel": "myforecaster.models.deep_learning.crossgnn",
        "SimpleFeedForwardModel": "myforecaster.models.deep_learning.simple_feedforward",
        "TiDEModel": "myforecaster.models.deep_learning.tide",
    }
    if name in _exports:
        import importlib
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractDLModel",
    "DLinearModel", "DeepARModel",
    "PatchTSTModel", "TFTModel", "iTransformerModel",
    "SMambaModel", "MambaTSModel",
    "NHiTSModel", "TSMixerModel", "SegRNNModel",
    "TimeMixerModel", "TimesNetModel", "ModernTCNModel",
    "MTGNNModel", "CrossGNNModel",
    "SimpleFeedForwardModel", "TiDEModel",
]
