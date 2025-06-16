import pandas as pd
from typing import Tuple, Optional

from .types import PavoneKey


def zero_contact_point(
    data: pd.DataFrame, metadata: pd.Series, contact_point: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Set the contact point to zero in the data.
    """
    if contact_point is None:
        return data, metadata, contact_point

    contact_time = contact_point[PavoneKey.time]
    data[PavoneKey.displacement] -= contact_point[PavoneKey.displacement]
    data[PavoneKey.time] -= contact_time
    # data[PavoneKey.force] -= contact_point[PavoneKey.force]

    # Set the contact point to zero
    # data.loc[data[PavoneKey.time] == contact_time, PavoneKey.displacement] = 0
    # data.loc[data[PavoneKey.time] == contact_time, PavoneKey.force] = 0

    metadata["approach start (s)"] -= contact_time
    metadata["approach end (s)"] -= contact_time

    metadata["dwell start (s)"] -= contact_time
    metadata["dwell end (s)"] -= contact_time

    metadata["retract start (s)"] -= contact_time
    metadata["retract end (s)"] -= contact_time

    contact_point -= contact_point

    return data, metadata, contact_point

def get_contact_point(approach_data: pd.DataFrame) -> pd.Series:
    # Find the last point in the approach data where the force is below zero
    max_idx = approach_data[PavoneKey.force_gradient].idxmax()
    contact_data = approach_data.iloc[: max_idx + 1]
    contact_data = contact_data[contact_data[PavoneKey.force_gradient] < 0]

    if contact_data.empty:
        return None

    contact_point = contact_data.iloc[-1]
    return contact_point


def get_pull_off_point(retract_data: pd.DataFrame) -> pd.Series:
    # Get all points after the minimum force point in the retract data
    min_force_idx = retract_data[PavoneKey.force].idxmin()
    pull_off_data = retract_data.iloc[min_force_idx + 1 :]
    pull_off_data = pull_off_data[pull_off_data[PavoneKey.force] > 0]

    if pull_off_data.empty:
        return None

    pull_off_point = pull_off_data.iloc[0]
    return pull_off_point
