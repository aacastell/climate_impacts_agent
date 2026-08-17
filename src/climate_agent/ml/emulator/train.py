import shutil
import sys

import mlflow
import numpy as np
import pandas as pd

from climate_agent.ml.emulator.generate_data import build_training_points
from climate_agent.ml.emulator.model import LinearYearRegressor
from climate_agent.ml.emulator.registry import MODELS_DIR

MODE_TO_MODEL_NAME = {
    "gwl": "window_from_gwl",
    "heat_extreme": "window_from_heat_extremes",
    "precip_extreme": "window_from_precip_extremes",
}

# Generous relative to the known-weak real result (precip_extreme ~17 years, gwl/heat_extreme
# ~5 years) — this isn't meant to catch "the fit is mediocre" (already true and documented),
# it's meant to catch a real regression: training erroring, a code bug, or a data problem
# blowing the fit up far past what any of these three real models have ever measured at.
MAX_ACCEPTABLE_MAE_YEARS = 25.0


def _loocv_mae_years(years: list[int], values: list[float]) -> float:
    """Leave-one-out cross-validation mean absolute error (years) for the year~value linear fit.

    The appropriate evaluation technique for this sample size (14 points) — a held-out
    train/test split would leave too few points on either side to mean anything. Fits on all
    but one point, predicts the left-out point, repeats for each point, averages the error.

    Args: years, values — real training points (parallel lists).
    Returns: mean absolute error in years.
    """
    errors = []
    for i in range(len(years)):
        train_years = years[:i] + years[i + 1 :]
        train_values = values[:i] + values[i + 1 :]
        slope, intercept = np.polyfit(train_values, train_years, deg=1)
        predicted = slope * values[i] + intercept
        errors.append(abs(predicted - years[i]))
    return float(np.mean(errors))


def train() -> dict[str, float]:
    """Fit and save one LinearYearRegressor per mode from real cached climate data, tracked
    via MLflow (params, LOOCV eval metric, saved model artifacts).

    Returns: {mode: loocv_mae_years} — also used by CI as a validate-only check (item 17's
    CT: fail the build if training errors or MAE is unreasonably high, without needing to
    persist any trained artifact from CI).
    """
    points = build_training_points()
    results: dict[str, float] = {}

    with mlflow.start_run(run_name="emulator_training"):
        for mode, series in points.items():
            years = list(series.keys())
            values = list(series.values())

            mae = _loocv_mae_years(years, values)
            results[mode] = mae

            model = LinearYearRegressor(years=years, values=values)
            mlflow.log_params(
                {f"{mode}_n_points": len(years), f"{mode}_slope": model.slope, f"{mode}_intercept": model.intercept}
            )
            mlflow.log_metric(f"{mode}_loocv_mae_years", mae)

            model_name = MODE_TO_MODEL_NAME[mode]
            path = MODELS_DIR / model_name
            if path.exists():
                shutil.rmtree(path)
            mlflow.pyfunc.save_model(
                path=str(path), python_model=model, input_example=pd.DataFrame({"value": [values[0]]})
            )

        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")

    return results


def main() -> None:
    """Train and print results. Exits non-zero if any mode's MAE exceeds
    MAX_ACCEPTABLE_MAE_YEARS — the real check CI (item 17's CT workflow) uses to catch a
    genuine regression, not local run output. Doesn't fail merely for a mediocre-but-known fit.
    """
    results = train()
    regressed = []
    for mode, mae in results.items():
        flag = " [REGRESSION]" if mae > MAX_ACCEPTABLE_MAE_YEARS else ""
        print(f"{mode}: LOOCV MAE = {mae:.1f} years{flag}")
        if mae > MAX_ACCEPTABLE_MAE_YEARS:
            regressed.append(mode)

    if regressed:
        print(f"\nFAILED: {regressed} exceeded MAX_ACCEPTABLE_MAE_YEARS={MAX_ACCEPTABLE_MAE_YEARS}")
        sys.exit(1)


if __name__ == "__main__":
    main()
