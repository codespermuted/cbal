"""Step 1: Scaffolding verification tests."""

import importlib
import subprocess
import sys


def test_version():
    """패키지 버전이 올바르게 노출되는지 확인."""
    import myforecaster
    assert myforecaster.__version__ == "0.1.0"


def test_subpackages_importable():
    """모든 서브패키지가 임포트 가능한지 확인."""
    packages = [
        "myforecaster",
        "myforecaster.dataset",
        "myforecaster.metrics",
        "myforecaster.models",
        "myforecaster.models.naive",
        "myforecaster.models.statsforecast",
        "myforecaster.models.tabular",
        "myforecaster.models.deep_learning",
        "myforecaster.models.ensemble",
        "myforecaster.serving",
        "myforecaster.cli",
        "myforecaster.predictor",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_model_registry():
    """모델 레지스트리 API가 동작하는지 확인."""
    from myforecaster.models import MODEL_REGISTRY, register_model, list_models

    @register_model("TestModel")
    class _TestModel:
        pass

    assert "TestModel" in MODEL_REGISTRY
    assert "TestModel" in list_models()

    # Cleanup
    del MODEL_REGISTRY["TestModel"]


def test_predictor_lazy_import():
    """TimeSeriesPredictor가 lazy import로 접근 가능한지 확인."""
    from myforecaster import TimeSeriesPredictor
    p = TimeSeriesPredictor(prediction_length=7)
    assert p.prediction_length == 7


def test_cli_version():
    """CLI --version이 동작하는지 확인."""
    result = subprocess.run(
        [sys.executable, "-m", "myforecaster.cli", "--version"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_core_dependencies():
    """핵심 의존성이 설치되어 있는지 확인."""
    import numpy
    import pandas
    import sklearn

    assert numpy.__version__ is not None
    assert pandas.__version__ is not None
    assert sklearn.__version__ is not None
