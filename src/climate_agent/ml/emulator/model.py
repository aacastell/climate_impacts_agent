import mlflow.pyfunc
import numpy as np
import pandas as pd

WINDOW_YEARS_BEFORE = 3
WINDOW_YEARS_AFTER = 2
MIN_CENTER_YEAR = 2011  # earliest year any cached scenario data actually covers
MAX_CENTER_YEAR = 2100  # SSP3-7.0 scenario runs don't extend past this


class LinearYearRegressor(mlflow.pyfunc.PythonModel):
    """Predicts a target year from a value via linear regression on real (year, value) points,
    then returns a 6-year window centered on that year.

    Real, trained on real data (see ml/emulator/generate_data.py) — but only 14 real yearly
    points exist, spanning two disjoint windows (2011-2014, 2051-2060) with a 37-year gap
    between them (item 8's real data scoping). Predictions for values outside the observed
    range are real linear extrapolation across that gap, not interpolation — a coarse estimate,
    not a precise scientific one. One instance of this class is independently fit per mode
    (gwl/heat_extreme/precip_extreme, see train.py) — same architecture across all three
    turned out to be the honest answer once real data was available, not a design shortcut.

    Predicted years are clamped to [MIN_CENTER_YEAR, MAX_CENTER_YEAR] — found this was a real,
    not theoretical, problem: precip_extreme's fit is genuinely weak (real LOOCV MAE ~17 years,
    the underlying dry-day values are nearly flat/noisy across both cached periods), and without
    a bound, an out-of-range input value extrapolated to a physically absurd predicted year
    (e.g. 1955 — before this dataset or the tool itself existed).
    """

    def __init__(self, years: list[int], values: list[float]):
        self.slope, self.intercept = (float(c) for c in np.polyfit(values, years, deg=1))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        predicted_years = self.slope * model_input["value"].to_numpy() + self.intercept
        predicted_years = np.clip(predicted_years, MIN_CENTER_YEAR, MAX_CENTER_YEAR)
        start_years = np.round(predicted_years - WINDOW_YEARS_BEFORE).astype(int)
        end_years = np.round(predicted_years + WINDOW_YEARS_AFTER).astype(int)
        return pd.DataFrame({"start_year": start_years, "end_year": end_years})
