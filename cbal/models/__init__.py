"""Model Zoo — all forecasting models live here."""

# Model registry: maps string names -> model classes
MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.

    Usage::

        @register_model("MyModel")
        class MyModel(AbstractTimeSeriesModel):
            ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def list_models() -> list[str]:
    """Return names of all registered models."""
    return sorted(MODEL_REGISTRY.keys())
