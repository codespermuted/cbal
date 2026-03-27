"""Deep learning models (PyTorch) — DLinear, DeepAR, and more.

These models require PyTorch. Install with::

    pip install 'cbal[deep]'

Imports are lazy — torch is only loaded when you actually use a DL model.
"""


def __getattr__(name):
    _exports = {
        "AbstractDLModel": "cbal.models.deep_learning.base",
        "DLinearModel": "cbal.models.deep_learning.dlinear",
        "DeepARModel": "cbal.models.deep_learning.deepar",
        "PatchTSTModel": "cbal.models.deep_learning.patchtst",
        "TFTModel": "cbal.models.deep_learning.tft",
        "iTransformerModel": "cbal.models.deep_learning.itransformer",
        "SMambaModel": "cbal.models.deep_learning.s_mamba",
        "MambaTSModel": "cbal.models.deep_learning.mambats",
        "NHiTSModel": "cbal.models.deep_learning.nhits",
        "TSMixerModel": "cbal.models.deep_learning.tsmixer",
        "SegRNNModel": "cbal.models.deep_learning.segrnn",
        "TimeMixerModel": "cbal.models.deep_learning.timemixer",
        "TimesNetModel": "cbal.models.deep_learning.timesnet",
        "ModernTCNModel": "cbal.models.deep_learning.moderntcn",
        "MTGNNModel": "cbal.models.deep_learning.mtgnn",
        "CrossGNNModel": "cbal.models.deep_learning.crossgnn",
        "SimpleFeedForwardModel": "cbal.models.deep_learning.simple_feedforward",
        "TiDEModel": "cbal.models.deep_learning.tide",
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
