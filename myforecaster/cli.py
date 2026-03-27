"""
Command-line interface for MyForecaster.

Usage::

    myforecaster info
    myforecaster fit --data train.csv --prediction-length 14 --presets medium_quality --output my_predictor
    myforecaster predict --predictor my_predictor --data test.csv --output predictions.csv
    myforecaster leaderboard --predictor my_predictor
    myforecaster serve --predictor my_predictor --port 8000
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="myforecaster",
        description="MyForecaster — AutoML Time Series Forecasting Library",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command")

    # --- info ---
    subparsers.add_parser("info", help="Show library info")

    # --- fit ---
    fit_p = subparsers.add_parser("fit", help="Train a predictor")
    fit_p.add_argument("--data", required=True, help="Training CSV path")
    fit_p.add_argument("--prediction-length", type=int, required=True)
    fit_p.add_argument("--presets", default="medium_quality")
    fit_p.add_argument("--eval-metric", default="MASE")
    fit_p.add_argument("--output", "-o", default="myforecaster_predictor")
    fit_p.add_argument("--time-limit", type=float, default=None)
    fit_p.add_argument("--num-val-windows", type=int, default=1)
    fit_p.add_argument("--refit-full", action="store_true")
    fit_p.add_argument("--id-column", default="item_id")
    fit_p.add_argument("--timestamp-column", default="timestamp")
    fit_p.add_argument("--target-column", default="target")

    # --- predict ---
    pred_p = subparsers.add_parser("predict", help="Generate forecasts")
    pred_p.add_argument("--predictor", required=True, help="Saved predictor path")
    pred_p.add_argument("--data", required=True, help="Context data CSV")
    pred_p.add_argument("--output", "-o", default="predictions.csv")
    pred_p.add_argument("--model", default=None)
    pred_p.add_argument("--id-column", default="item_id")
    pred_p.add_argument("--timestamp-column", default="timestamp")
    pred_p.add_argument("--target-column", default="target")

    # --- leaderboard ---
    lb_p = subparsers.add_parser("leaderboard", help="Show model leaderboard")
    lb_p.add_argument("--predictor", required=True)

    # --- serve ---
    srv_p = subparsers.add_parser("serve", help="Start REST API server")
    srv_p.add_argument("--predictor", required=True)
    srv_p.add_argument("--host", default="0.0.0.0")
    srv_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.version:
        from myforecaster import __version__
        print(f"myforecaster {__version__}")
        sys.exit(0)

    dispatch = {
        "info": _cmd_info,
        "fit": _cmd_fit,
        "predict": _cmd_predict,
        "leaderboard": _cmd_leaderboard,
        "serve": _cmd_serve,
    }
    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


# ------------------------------------------------------------------

def _cmd_info(_args):
    import platform
    from myforecaster import __version__
    print("=" * 50)
    print(f"  MyForecaster v{__version__}")
    print("=" * 50)
    print(f"  Python:    {platform.python_version()}")
    print(f"  Platform:  {platform.system()} {platform.machine()}")
    _deps = {
        "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
        "statsforecast": "statsforecast", "lightgbm": "lightgbm",
        "torch": "torch", "fastapi": "fastapi", "optuna": "optuna",
    }
    print("  Dependencies:")
    for mod, name in _deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"    {name:20s} {ver}")
        except ImportError:
            print(f"    {name:20s} NOT INSTALLED")
    print("=" * 50)


def _cmd_fit(args):
    import pandas as pd
    from myforecaster.predictor import TimeSeriesPredictor
    from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df, id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
    )
    print(f"  Items: {tsdf.num_items}, Rows: {len(tsdf)}")

    predictor = TimeSeriesPredictor(
        prediction_length=args.prediction_length,
        eval_metric=args.eval_metric,
        path=args.output,
    )
    predictor.fit(
        tsdf, presets=args.presets,
        time_limit=args.time_limit,
        num_val_windows=args.num_val_windows,
        refit_full=args.refit_full,
    )
    predictor.save()
    print(f"\nPredictor saved to: {args.output}")
    predictor.leaderboard()


def _cmd_predict(args):
    import pandas as pd
    from myforecaster.predictor import TimeSeriesPredictor
    from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame

    print(f"Loading predictor from {args.predictor}...")
    predictor = TimeSeriesPredictor.load(args.predictor)
    df = pd.read_csv(args.data)
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df, id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
    )
    preds = predictor.predict(tsdf, model=args.model)
    preds.to_csv(args.output)
    print(f"Predictions saved to: {args.output}  (shape: {preds.shape})")


def _cmd_leaderboard(args):
    from myforecaster.predictor import TimeSeriesPredictor
    predictor = TimeSeriesPredictor.load(args.predictor)
    predictor.leaderboard()


def _cmd_serve(args):
    from myforecaster.serving.app import run_server
    print(f"Starting server at {args.host}:{args.port}...")
    run_server(args.predictor, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
