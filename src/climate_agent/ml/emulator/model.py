import mlflow.pyfunc
import pandas as pd


class GWLWindowModel(mlflow.pyfunc.PythonModel):
    """Real stub: always predicts the one verified window (GFDL-ESM4/SSP3-7.0, 2°C -> 2057),
    ignoring input, until train.py (item 17) trains this on real GWL-crossing data."""

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"start_year": [2054] * len(model_input), "end_year": [2059] * len(model_input)}
        )


class HeatExtremeWindowModel(mlflow.pyfunc.PythonModel):
    """Real stub: always predicts the one verified window, ignoring input, until train.py
    (item 17) trains this on real heat-extreme-days trajectory data."""

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"start_year": [2054] * len(model_input), "end_year": [2059] * len(model_input)}
        )


class PrecipExtremeWindowModel(mlflow.pyfunc.PythonModel):
    """Real stub: always predicts the one verified window, ignoring input, until train.py
    (item 17) trains this on real precip-extreme trajectory data."""

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"start_year": [2054] * len(model_input), "end_year": [2059] * len(model_input)}
        )
