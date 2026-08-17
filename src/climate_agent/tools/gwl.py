from climate_agent.ml.emulator.registry import predict_window
from climate_agent.schemas import Window


def window_from_gwl(gwl_celsius: float) -> Window:
    """Predict a target window from a direct GWL target (e.g. 2.0 for "2°C")."""
    return predict_window("window_from_gwl", gwl_celsius)


def window_from_heat_extremes(heat_extreme_days: float) -> Window:
    """Predict a target window from a target heat-extreme-days value."""
    return predict_window("window_from_heat_extremes", heat_extreme_days)


def window_from_precip_extremes(precip_extreme_pct: float) -> Window:
    """Predict a target window from a target precip-extreme value."""
    return predict_window("window_from_precip_extremes", precip_extreme_pct)
