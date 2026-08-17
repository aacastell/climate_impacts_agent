from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from climate_agent.ml.emulator.model import (
    GWLWindowModel,
    HeatExtremeWindowModel,
    PrecipExtremeWindowModel,
)
from climate_agent.schemas import Window

MODELS_DIR = Path(__file__).parent / "models"
MODEL_CLASSES = {
    "window_from_gwl": GWLWindowModel,
    "window_from_heat_extremes": HeatExtremeWindowModel,
    "window_from_precip_extremes": PrecipExtremeWindowModel,
}


def save_stub_models() -> None:
    """One-time setup: saves each stub model under its own name in MODELS_DIR, skipping any
    that already exist.

    Real training (train.py, item 17) overwrites these same paths with trained models via the
    same mlflow.pyfunc.save_model mechanism — nothing downstream changes.
    """
    for name, model_class in MODEL_CLASSES.items():
        path = MODELS_DIR / name
        if not path.exists():
            mlflow.pyfunc.save_model(
                path=str(path),
                python_model=model_class(),
                input_example=pd.DataFrame({"value": [2.0]}),
            )


def predict_window(model_name: str, value: float) -> Window:
    """Load a saved window model from disk and predict a window for one input value.

    Args: model_name — one of MODEL_CLASSES' keys. value — the target for that model
    (e.g. 2.0 for a 2°C GWL target).
    Returns: predicted Window.
    """
    save_stub_models()
    model = mlflow.pyfunc.load_model(str(MODELS_DIR / model_name))
    result = model.predict(pd.DataFrame({"value": [value]}))
    return Window(start_year=int(result["start_year"][0]), end_year=int(result["end_year"][0]))
