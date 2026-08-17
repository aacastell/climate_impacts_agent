from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from climate_agent.schemas import Window

MODELS_DIR = Path(__file__).parent / "models"
MODEL_NAMES = ["window_from_gwl", "window_from_heat_extremes", "window_from_precip_extremes"]


def predict_window(model_name: str, value: float) -> Window:
    """Load a trained window model from disk and predict a window for one input value.

    No auto-train fallback — model provisioning is `dvc repro`'s job (train.py depends on the
    cached climate data via DVC), not something the tool layer silently regenerates on demand.
    Same reasoning as the item-13 fix for the RAG corpus: a real-data-dependent artifact needs a
    real dependency-tracked trigger, not an if-missing check that can't detect stale data.

    Args: model_name — one of MODEL_NAMES. value — the target for that model
    (e.g. 2.0 for a 2°C GWL target).
    Returns: predicted Window.
    """
    model = mlflow.pyfunc.load_model(str(MODELS_DIR / model_name))
    result = model.predict(pd.DataFrame({"value": [value]}))
    return Window(start_year=int(result["start_year"][0]), end_year=int(result["end_year"][0]))
