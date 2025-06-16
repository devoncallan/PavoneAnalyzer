import pandas as pd
from typing import Tuple, Optional, Dict, List

from .types import PavoneKey


def split_by_phase(
    data: pd.DataFrame, metadata: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into approach, dwell, and retract phases based on the metadata timings.
    """

    # Approach phase data
    approach_start = metadata["approach start (s)"]
    approach_end = metadata["approach end (s)"]
    approach_data = data[
        (data[PavoneKey.time] >= approach_start)
        & (data[PavoneKey.time] <= approach_end)
    ].reset_index(drop=True)

    # Dwell phase data
    dwell_start = metadata["dwell start (s)"]
    dwell_end = metadata["dwell end (s)"]
    dwell_data = data[
        (data[PavoneKey.time] >= dwell_start) & (data[PavoneKey.time] <= dwell_end)
    ].reset_index(drop=True)

    # Retract phase data
    retract_start = metadata["retract start (s)"]
    retract_end = metadata["retract end (s)"]
    retract_data = data[
        (data[PavoneKey.time] >= retract_start) & (data[PavoneKey.time] <= retract_end)
    ].reset_index(drop=True)

    return approach_data, dwell_data, retract_data


def add_phase_timings(metadata: pd.DataFrame) -> pd.DataFrame:

    for i, row in metadata.iterrows():
        start_times_str = str(row["Step absolute start times (s)"])
        start_times = [float(t) for t in start_times_str.split(",")]

        assert (
            len(start_times) == 4,
            f"Expected 4 start times, got {len(start_times)}: {start_times_str}",
        )

        sequences = {
            "approach": {
                "start (s)": start_times[0],
                "end (s)": start_times[1],
                "duration (s)": start_times[1] - start_times[0],
                "displacement (um)": row["D[Z1] (nm)"] * 0.001,
            },
            "dwell": {
                "start (s)": start_times[1],
                "end (s)": start_times[2],
                "duration (s)": start_times[2] - start_times[1],
                "displacement (um)": 0,
            },
            "retract": {
                "start (s)": start_times[2],
                "end (s)": start_times[3],
                "duration (s)": start_times[3] - start_times[2],
                "displacement (um)": (row["D[Z3] (nm)"] - row["D[Z2] (nm)"]) * 0.001,
            },
        }

        # Calculate speed for each sequence
        for seq in sequences.values():
            seq["speed (um/s)"] = (
                seq["displacement (um)"] / seq["duration (s)"]
                if seq["duration (s)"] > 0
                else 0
            )

        # Collapse sequences into a 1D dictionary
        collapsed_sequences = {
            f"{seq_name} {key}": seq_info[key]
            for seq_name, seq_info in sequences.items()
            for key in seq_info
        }

        # Update the row with the collapsed sequences
        for key, value in collapsed_sequences.items():
            metadata.at[i, key] = value

    return metadata


def map_experimental_conditions(condition_str: str) -> Optional[Dict[str, float]]:

    # Default conditions
    conditions = {"depth_um": 2.0, "dwell_time_s": 1.0, "retract_speed_ums": 2.0}

    condition_str = str(condition_str)

    # Depths: 1um, 2um, 3um
    if "depths_1um" in condition_str:
        conditions["depth_um"] = 1.0
    elif "depths_2um" in condition_str:
        conditions["depth_um"] = 2.0
    elif "depths_3um" in condition_str:
        conditions["depth_um"] = 3.0

    # Dwell Times: 1s, 10s, 100s
    elif "dwell_1s" in condition_str:
        conditions["dwell_time_s"] = 1.0
    elif "dwell_10s" in condition_str:
        conditions["dwell_time_s"] = 10.0
    elif "dwell_100s" in condition_str:
        conditions["dwell_time_s"] = 100.0

    # Retraction Speeds: 0.2, 2, 20
    elif "retractSpeed_0p2" in condition_str:
        conditions["retract_speed_ums"] = 0.2
    elif "retractSpeed_2" in condition_str:
        conditions["retract_speed_ums"] = 2.0
    elif "retractSpeed_20" in condition_str:
        conditions["retract_speed_ums"] = 20.0
    else:
        print(f"Unknown experimental condition: {condition_str}")
        return None

    return conditions


def add_experimental_conditions(
    data: pd.DataFrame, filepath_column="output_filepath"
) -> pd.DataFrame:
    """
    Adds experimental condition columns to an existing dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with filepath column
        filepath_column (str): Name of column containing filepaths

    Returns:
        pd.DataFrame: Dataframe with added condition columns
    """
    data = data.copy()

    # Extract conditions for each row
    condition_data = data[filepath_column].apply(map_experimental_conditions)

    # Convert to dataframe and concatenate
    condition_df = pd.DataFrame(condition_data.tolist())
    print(condition_df)
    print("condition_df.columns:", condition_df.columns)
    print("Data columns:", data.columns)

    # Add to original dataframe
    for col in condition_df.columns:
        data[col] = condition_df[col]

    print("Data columns after adding conditions:", data.columns)
    # quit()

    return data


def filter_experiments(
    data: pd.DataFrame,
    sample_ids: List[str],
    depth_selections: List[float],
    dwell_selections: List[float],
    retract_speed_selections: List[float],
) -> pd.DataFrame:
    """
    Filter experiments based on selected conditions.

    Args:
        data (pd.DataFrame): DataFrame containing experiment data
        sample_ids (List[str]): List of sample IDs to filter
        depth_selections (List[float]): Selected depths
        dwell_selections (List[float]): Selected dwell times
        retract_speed_selections (List[float]): Selected retraction speeds

    Returns:
        pd.DataFrame: Filtered DataFrame
    """

    # Ensure all selections are provided
    if (
        not sample_ids
        or not depth_selections
        or not dwell_selections
        or not retract_speed_selections
    ):
        return pd.DataFrame()

    filtered_data = data[
        (data["sample_id"].isin(sample_ids))
        & (data["depth_um"].isin(depth_selections))
        & (data["dwell_time_s"].isin(dwell_selections))
        & (data["retract_speed_ums"].isin(retract_speed_selections))
    ]

    return filtered_data.reset_index(drop=True)
