# import pandas as pd
# from pavone.types import PavoneKey

# def find_contact_point(approach_data: pd.DataFrame, window=251, threshold_factor=5) -> pd.Series:
#     # Rolling variance
#     rolling_var = approach_data[PavoneKey.force].rolling(window=window).var()

#     # Baseline variance (from early data)
#     baseline_var = rolling_var.iloc[window : window * 2].median()

#     # Find first point above threshold
#     threshold = baseline_var * threshold_factor
#     contact_idx = (rolling_var > threshold).idxmax()

#     # Return the contact point
#     return approach_data.loc[contact_idx]

import pandas as pd
from pavone.types import PavoneKey


# def find_contact_point(
#     approach_data: pd.DataFrame, window=251, threshold_factor=5, min_consecutive=100
# ) -> pd.Series:
#     # Rolling variance
#     rolling_var = approach_data[PavoneKey.force].rolling(window=window).var()

#     # Baseline variance (from early data)
#     baseline_var = rolling_var.iloc[window : window * 2].median()
#     threshold = baseline_var * threshold_factor

#     # Find consecutive points above threshold
#     above_threshold = rolling_var > threshold

#     # Check for consecutive True values
#     for i in range(len(above_threshold) - min_consecutive + 1):
#         if above_threshold.iloc[i : i + min_consecutive].all():
#             contact_idx = above_threshold.index[i]
#             return approach_data.loc[contact_idx]


#     return None  # No contact found
# def find_contact_point(
#     approach_data: pd.DataFrame, window=251, threshold_factor=10, min_consecutive=200
# ) -> pd.Series:
#     # Rolling variance
#     rolling_var = approach_data[PavoneKey.force].rolling(window=window).var()

#     # Baseline variance (from early data)
#     # baseline_var = rolling_var.iloc[window : 2000].median()
#     # Get the first 2 seconds of data for baseline
#     baseline_var = rolling_var.iloc[window : 5 * window].median()
#     threshold = baseline_var * threshold_factor

#     above_threshold = rolling_var > threshold

#     # Check if current point AND next N-1 points are all above threshold
#     consecutive = above_threshold
#     for i in range(1, min_consecutive):
#         consecutive = consecutive & above_threshold.shift(-i)


#     contact_idx = consecutive.idxmax() if consecutive.any() else None
#     return approach_data.loc[contact_idx] if contact_idx is not None else None
def find_contact_point(
    approach_data: pd.DataFrame,
    window=251,
    threshold_factor=5,
    min_consecutive_pct=0.05,  # 5% of approach data
    baseline_pct=0.5,  # Use first 50% for baseline
) -> pd.Series:

    # Rolling variance
    rolling_var = approach_data[PavoneKey.force].rolling(window=window).var()

    # Baseline variance (from first X% of data)
    baseline_end = int(len(rolling_var) * baseline_pct)
    baseline_var = rolling_var.iloc[window:baseline_end].median()

    threshold = baseline_var * threshold_factor
    above_threshold = rolling_var > threshold

    # Consecutive points (as percentage of total data)
    min_consecutive = max(1, int(len(approach_data) * min_consecutive_pct))

    # Check for consecutive points
    consecutive = above_threshold
    for i in range(1, min_consecutive):
        consecutive = consecutive & above_threshold.shift(-i)

    contact_idx = consecutive.idxmax() if consecutive.any() else None
    return approach_data.loc[contact_idx] if contact_idx is not None else None


def find_transition_point(
    data: pd.DataFrame,
    window: int = 251,
    threshold_factor=5,
    min_consecutive_pct=0.02,  # 5% of data
    baseline_pct=0.5,
    reverse: bool = False,
) -> pd.Series:

    # Reverse the data for retract phase
    if reverse:
        data = data.iloc[::-1].reset_index(drop=True)

    # Calculate rolling variance
    rolling_var = data[PavoneKey.force_gradient].rolling(window=window).var()

    # Baseline variance (from first X% of data)
    baseline_end = int(len(rolling_var) * baseline_pct)
    baseline_var = rolling_var.iloc[:baseline_end].median()

    threshold = baseline_var * threshold_factor
    above_threshold = rolling_var > threshold

    # Consecutive points (as percentage of total data)
    min_consecutive = max(1, int(len(data) * min_consecutive_pct))

    # Check for consecutive points
    consecutive = above_threshold
    for i in range(1, min_consecutive):
        consecutive = consecutive & above_threshold.shift(-i)

    transition_idx = consecutive.idxmax() if consecutive.any() else None

    if transition_idx is None:
        return None

    return data.loc[transition_idx]

    # if not reverse:
    #     return data.loc[transition_idx]
    # else:
    #     # If reverse, return the corresponding point in the original data
    #     original_idx = len(data) - 1 - transition_idx
    #     return data.loc[original_idx]

    # if transition_idx is not None:
    #     if reverse

    pass


#     """
# )

# # Usage
# # contact_point = find_contact_point(approach_data)
# # contact_time = approach_data.loc[contact_point, PavoneKey.time]
