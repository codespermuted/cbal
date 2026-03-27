"""
MyForecaster — AutoML Time Series Forecasting Library
=====================================================

Inspired by AutoGluon-TimeSeries, fully independent.

Quick Start::

    from myforecaster import TimeSeriesPredictor, TimeSeriesDataFrame

    data = TimeSeriesDataFrame.from_data_frame(df)
    predictor = TimeSeriesPredictor(prediction_length=14, eval_metric="MASE")
    predictor.fit(data, presets="medium_quality")
    predictions = predictor.predict(data)
"""

__version__ = "0.1.0"

# Lazy imports — avoid heavy dependencies at package import time
def __getattr__(name):
    if name == "TimeSeriesPredictor":
        from myforecaster.predictor import TimeSeriesPredictor
        return TimeSeriesPredictor
    if name == "TimeSeriesDataFrame":
        from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
        return TimeSeriesDataFrame
    raise AttributeError(f"module 'myforecaster' has no attribute {name!r}")


__all__ = [
    "__version__",
    "TimeSeriesPredictor",
    "TimeSeriesDataFrame",
]
