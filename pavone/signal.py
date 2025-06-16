from numpy.typing import ArrayLike
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

from pavone.types import PavoneKey


def savgol_smoothing(
    data: ArrayLike, window_len: int = 251, polyorder: int = 3
) -> ArrayLike:
    if window_len % 2 == 0:
        window_len += 1  # Ensure window length is odd

    return savgol_filter(data, window_length=window_len, polyorder=polyorder)


def baseline_correct(data: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:

    from pavone.experiment import split_by_phase

    approach_data, dwell_data, retract_data = split_by_phase(data, metadata)

    approach_baseline = approach_data.iloc[: int(len(approach_data) * 0.7)]
    retract_baseline = retract_data.iloc[-int(len(retract_data) * 0.7) :]

    baseline_data = pd.concat([approach_baseline, retract_baseline])
    baseline_time = baseline_data[PavoneKey.time].values.reshape(-1, 1)
    baseline_force = baseline_data[PavoneKey.force].values

    approach_weight = 0.5 / len(approach_baseline)
    retract_weight = 0.5 / len(retract_baseline)
    weights = ([approach_weight] * len(approach_baseline)) + (
        [retract_weight] * len(retract_baseline)
    )

    # Fit a linear regression to the baseline data
    reg = LinearRegression().fit(baseline_time, baseline_force, sample_weight=weights)
    baseline_pred = reg.predict(baseline_time)

    time = data[PavoneKey.time].values.reshape(-1, 1)
    force = data[PavoneKey.force].values
    baseline_line = reg.predict(time)
    force_corrected = force - baseline_line

    corrected_data = data.copy()
    corrected_data[PavoneKey.force] = force_corrected

    return corrected_data
