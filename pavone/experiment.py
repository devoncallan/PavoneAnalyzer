import pandas as pd
from typing import Tuple, Optional, Dict, List, Any

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
    elif "retractSpeed_20" in condition_str:
        conditions["retract_speed_ums"] = 20.0
    elif "retractSpeed_2" in condition_str:
        conditions["retract_speed_ums"] = 2.0

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


# def filter_experiments(
#     data: pd.DataFrame,
#     sample_ids: List[str],
#     depth_selections: List[float],
#     dwell_selections: List[float],
#     retract_speed_selections: List[float],
# ) -> pd.DataFrame:
#     """
#     Filter experiments based on selected conditions.

#     Args:
#         data (pd.DataFrame): DataFrame containing experiment data
#         sample_ids (List[str]): List of sample IDs to filter
#         depth_selections (List[float]): Selected depths
#         dwell_selections (List[float]): Selected dwell times
#         retract_speed_selections (List[float]): Selected retraction speeds

#     Returns:
#         pd.DataFrame: Filtered DataFrame
#     """

#     # Ensure all selections are provided
#     if (
#         not sample_ids
#         or not depth_selections
#         or not dwell_selections
#         or not retract_speed_selections
#     ):
#         return pd.DataFrame()

#     filtered_data = data[
#         (data["sample_id"].isin(sample_ids))
#         & (data["depth_um"].isin(depth_selections))
#         & (data["dwell_time_s"].isin(dwell_selections))
#         & (data["retract_speed_ums"].isin(retract_speed_selections))
#     ]

#     return filtered_data.reset_index(drop=True)


def filter_experiments(
    data: pd.DataFrame,
    sample_ids: List[str],
    depth_selections: List[float],
    dwell_selections: List[float],
    retract_speed_selections: List[float],
) -> pd.DataFrame:
    """
    Filter experiments based on selected conditions.
    """

    # Debug: Print what we received
    print("=== DEBUG filter_experiments ===")
    print(f"Data shape: {data.shape}")
    print(f"Sample IDs: {sample_ids} (type: {type(sample_ids)})")
    print(f"Depth selections: {depth_selections} (type: {type(depth_selections)})")
    print(f"Dwell selections: {dwell_selections} (type: {type(dwell_selections)})")
    print(
        f"Retract speed selections: {retract_speed_selections} (type: {type(retract_speed_selections)})"
    )

    # Check if any selections are empty
    if (
        not sample_ids
        or not depth_selections
        or not dwell_selections
        or not retract_speed_selections
    ):
        print("One or more selections are empty, returning empty DataFrame")
        return pd.DataFrame()

    # Debug: Check data columns and sample values
    print(f"Available columns: {list(data.columns)}")
    if "sample_id" in data.columns:
        print(f"Unique sample_ids in data: {data['sample_id'].unique()}")
    if "depth_um" in data.columns:
        print(f"Unique depth_um in data: {data['depth_um'].unique()}")
    if "dwell_time_s" in data.columns:
        print(f"Unique dwell_time_s in data: {data['dwell_time_s'].unique()}")
    if "retract_speed_ums" in data.columns:
        print(f"Unique retract_speed_ums in data: {data['retract_speed_ums'].unique()}")

    try:
        # Apply filters one by one to see which one fails
        print("Applying sample_id filter...")
        mask1 = data["sample_id"].isin(sample_ids)
        print(f"Sample filter result: {mask1.sum()} matches")

        print("Applying depth_um filter...")
        mask2 = data["depth_um"].isin(depth_selections)
        print(f"Depth filter result: {mask2.sum()} matches")

        print("Applying dwell_time_s filter...")
        mask3 = data["dwell_time_s"].isin(dwell_selections)
        print(f"Dwell filter result: {mask3.sum()} matches")

        print("Applying retract_speed_ums filter...")
        mask4 = data["retract_speed_ums"].isin(retract_speed_selections)
        print(f"Retract speed filter result: {mask4.sum()} matches")

        print("Combining all filters...")
        combined_mask = mask1 & mask2 & mask3 & mask4
        print(f"Combined filter result: {combined_mask.sum()} matches")

        filtered_data = data[combined_mask]

    except Exception as e:
        print(f"Error in filtering: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    print(f"Returning filtered data with shape: {filtered_data.shape}")
    return filtered_data.reset_index(drop=True)


def label_experiments(
    data: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.DataFrame]:

    labeled_data = {}
    for _, row in data.iterrows():
        sample_id = row["sample_id"]
        depth = row["depth_um"]
        dwell_time = row["dwell_time_s"]
        retract_speed = row["retract_speed_ums"]

        key = (sample_id, f"{depth}um_{dwell_time}s_{retract_speed}um/s")
        if key not in labeled_data:
            labeled_data[key] = pd.DataFrame(columns=data.columns)

        labeled_data[key] = pd.concat(
            [labeled_data[key], row.to_frame().T], ignore_index=True
        )
    return labeled_data


def extract_data_for_plotting(
    experiments_data: Dict[str, List[Any]], group_by: Optional[str] = None
) -> Dict[str, List[pd.DataFrame]]:
    """
    Extract the main data from experiments_data for plotting.
    Convert from experiment tuples to simple data dict for plot_exp_overlay.
    Args:
        experiments_data (Dict): Dictionary containing experiment data
        group_by (Optional[str]): Grouping key, can be 'sample_id' or 'protocol'

    """
    data_dict = {}
    for (sample_id, protocol), bundled_data_list in experiments_data.items():

        if group_by is None:
            label = f"{sample_id}_{protocol}"
        elif group_by == "sample_id":
            label = sample_id
        elif group_by == "protocol":
            label = protocol

        # if label not in data_dict:
            # data_dict[label] = []

        data_dict[label] = bundled_data_list

    return data_dict
