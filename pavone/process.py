from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


from pavone.types import PavoneKey, reduced_columns
from pavone.parse import read_pavone_data, parse_pavone_filepath, locate_pavone_data


def process_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:

    F_normal_uN = raw_data[PavoneKey.load].to_numpy()  # Normal force [μN]
    d_indent_um = raw_data[PavoneKey.indent].to_numpy() / 1000  # Indentation [μm]
    d_cantilever_um = (
        raw_data[PavoneKey.cantilever].to_numpy() / 1000
    )  # Cantilever deflection [μm]
    d_piezo_um = raw_data[PavoneKey.piezo].to_numpy() / 1000  # Piezo deflection [μm]

    z_stage = d_piezo_um - d_cantilever_um  # Absolute probe position of the Pavone

    data = pd.DataFrame(columns=reduced_columns)
    data[PavoneKey.time] = raw_data[PavoneKey.time]
    data[PavoneKey.displacement] = z_stage
    data[PavoneKey.force] = F_normal_uN

    return data


def reduce_data(data: pd.DataFrame, N: int = 1, window_size: int = 1) -> pd.DataFrame:

    reduced_data = data.copy()

    # reduced_data[PavoneKey.force] = savgol_filter(
    #     data[PavoneKey.force], window_length=N, polyorder=3
    # )

    if N > 1:
        reduced_data = (
            reduced_data.groupby(reduced_data.index // N).agg(
                {
                    PavoneKey.time: "first",
                    PavoneKey.displacement: "mean",
                    PavoneKey.force: lambda x: x.loc[x.abs().idxmax()],
                    # PavoneKey.force: lambda x: np.percentile(x, 99),
                }
            )
        ).reset_index(drop=True)

    if window_size > 1:
        reduced_data[PavoneKey.force] = (
            reduced_data[PavoneKey.force]
            .rolling(window=window_size, center=True)
            .mean()
        )
        # reduced_data[PavoneKey.displacement] = (
        #     reduced_data[PavoneKey.displacement]
        #     .rolling(window=window_size, center=True)
        #     .mean()
        # )

    return reduced_data


def process_all_pavone_data(data_dir: str, output_dir: str):
    pavone_filepaths = locate_pavone_data(data_dir)
    num_files = len(pavone_filepaths)
    print(f"Found {num_files} Pavone files in {data_dir}")

    # List to store all metadata dictionaries
    all_metadata = []

    for filepath in pavone_filepaths:
        filepath_metadata = parse_pavone_filepath(filepath)
        exp_metadata, raw_data = read_pavone_data(filepath)
        processed_data = process_raw_data(raw_data)

        output_filename = (
            "_".join([filepath_metadata["date"], filepath_metadata["filename"]])
            + ".csv"
        )

        output_filepath = Path(output_dir) / output_filename
        processed_data.to_csv(output_filepath, index=False)

        # Combine all metadata into a single dictionary
        combined_metadata = {
            **filepath_metadata,
            **exp_metadata,
            "output_filepath": str(output_filepath),
        }

        all_metadata.append(combined_metadata)

    # Create DataFrame from all metadata
    metadata_df = pd.DataFrame(all_metadata)
    # print(metadata_df.columns)

    metadata_df = calculate_timings(metadata_df)

    metadata_df = add_conditions_to_dataframe(metadata_df)

    # print("After Adding Conditions:", metadata_df.columns)

    return metadata_df


def calculate_timings(metadata: pd.DataFrame) -> pd.DataFrame:

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


def split_by_timings(
    data: pd.DataFrame, metadata: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into approach, dwell, and retract based on the metadata timings.
    """
    approach_start = metadata["approach start (s)"]
    approach_end = metadata["approach end (s)"]
    dwell_start = metadata["dwell start (s)"]
    dwell_end = metadata["dwell end (s)"]
    retract_start = metadata["retract start (s)"]
    retract_end = metadata["retract end (s)"]

    approach_data = data[
        (data[PavoneKey.time] >= approach_start)
        & (data[PavoneKey.time] <= approach_end)
    ].reset_index(drop=True)
    dwell_data = data[
        (data[PavoneKey.time] >= dwell_start) & (data[PavoneKey.time] <= dwell_end)
    ].reset_index(drop=True)
    retract_data = data[
        (data[PavoneKey.time] >= retract_start) & (data[PavoneKey.time] <= retract_end)
    ].reset_index(drop=True)

    return approach_data, dwell_data, retract_data


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


def get_point_at_time(data: pd.DataFrame, time: float) -> pd.Series:
    """
    Get the data point at a specific time.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        time (float): The time at which to get the data point.

    Returns:
        pd.Series: The data point at the specified time, or None if not found.
    """
    # point = data[data[PavoneKey.time] == time]
    idx = (data[PavoneKey.time] - time).abs().idxmin()
    point = data.iloc[[idx]]
    return point

    # if not point.empty:
    #     return point.iloc[0]

    # # point = data[data[PavoneKey.time] == time]
    # else:
    #     return None


def baseline_correct(data: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:

    approach_data, dwell_data, retract_data = split_by_timings(data, metadata)

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


# def baseline_correct(data: pd.DataFrame) -> pd.DataFrame:

#     time = data[PavoneKey.time]
#     force = data[PavoneKey.force]

#     approach_mask = time <= time.min() + 2.0
#     retract_mask = time >= time.max() - 0.5
#     baseline_mask = approach_mask | retract_mask

#     baseline_time = time[baseline_mask].values.reshape(-1, 1)
#     baseline_force = force[baseline_mask].values

#     reg = LinearRegression().fit(baseline_time, baseline_force)

#     baseline_pred = reg.predict(baseline_time)
#     residuals = baseline_force - baseline_pred
#     std_dev = np.std(residuals)

#     baseline_line = reg.predict(time.values.reshape(-1, 1))
#     force_corrected = force - baseline_line

#     corrected_data = data.copy()
#     corrected_data[PavoneKey.force] = force_corrected

#     return corrected_data


def savgol_smoothing(
    data: ArrayLike, window_len: int = 251, polyorder: int = 3
) -> ArrayLike:
    if window_len % 2 == 0:
        window_len += 1  # Ensure window length is odd

    return savgol_filter(data, window_length=window_len, polyorder=polyorder)
    # smoothed_data = data.copy()
    # smoothed_data[PavoneKey.force] = savgol_filter(
    #     data[PavoneKey.force], window_length=window_len, polyorder=polyorder
    # )
    # return smoothed_data


# def split_approach_retract(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

#     assert (
#         list(data.columns) == reduced_columns
#     ), "Data does not contain the expected columns."

#     # Get the index of the maximum force
#     max_force_idx = data[PavoneKey.force].idxmax()
#     max_disp_idx = data[PavoneKey.displacement].idxmax()

#     split_idx = min(max_force_idx, max_disp_idx)

#     approach_data = data.iloc[: split_idx + 1].copy().reset_index()
#     retract_data = data.iloc[split_idx + 1 :].copy().reset_index()

#     return approach_data, retract_data


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


def add_conditions_to_dataframe(
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

    # print("Zeroed:")
    # print("Data: ", data.shape)
    # print(f"Metadata: {metadata.shape}")
    # print(f"Contact point: {contact_point}")

    return data, metadata, contact_point


def process_data(raw_data: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:

    data = raw_data.copy()

    # 1. Initial baseline correction (conservative)
    data = data[data[PavoneKey.time] >= 2]
    data = baseline_correct(data, metadata)

    force_gradient = np.gradient(data[PavoneKey.force], data[PavoneKey.time])
    data[PavoneKey.force_gradient] = savgol_smoothing(force_gradient, window_len=501)

    # 2. Estimate contact point and pull-off point
    approach_data, _, _ = split_by_timings(data, metadata)
    contact_point = get_contact_point(approach_data)

    data, metadata, contact_point = zero_contact_point(data, metadata, contact_point)
    # print(f"Contact point: {contact_point}")

    _, _, retract_data = split_by_timings(data, metadata)
    pull_off_point = get_pull_off_point(retract_data)

    if pull_off_point is not None:
        pull_off_point.force = 0

    # print("Data: ", data.shape)
    # print(f"Contact point: {contact_point}")
    # print(f"Pull-off point: {pull_off_point}")

    return data, contact_point, pull_off_point

    # # 3. More educated baseline correction
    # data = baseline_correct(data, metadata, contact_point)

    # # 3. Re-estimate contact point and pull-off point
    # approach_data, dwell_data, retract_data = split_by_timings(data, metadata)
    # contact_point = get_contact_point(approach_data)
    # pull_off_point = get_pull_off_point(retract_data)

    # pass


# data_test = data.copy()
# # data_test = data_test.set_index(PavoneKey.time)

# N = 50

# data_test = (
#     data_test.groupby(data_test.index // N)
#     .agg(
#         {
#             PavoneKey.time: "first",
#             PavoneKey.displacement: "mean",
#             PavoneKey.force: "mean",
#         }
#     )
#     .reset_index(drop=True)
# )
# window = 2
# data_test[PavoneKey.force] = (
#     data_test[PavoneKey.force].rolling(window=window, center=True).mean()
# )
# data_test[PavoneKey.displacement] = (
#     data_test[PavoneKey.displacement].rolling(window=window, center=True).mean()
# )
# plt.plot(data[PavoneKey.displacement], data[PavoneKey.force], alpha=0.2)
# plt.plot(data_test[PavoneKey.displacement], data_test[PavoneKey.force])


# pavone_data["Approach speed (um/s)"] = (
#     0.001 * pavone_data["D[Z1] (nm)"] / pavone_data["t[1] (s)"]
# )
# pavone_data["Retract speed (um/s)"] = (
#     -0.001
#     * (pavone_data["D[Z3] (nm)"] - pavone_data["D[Z2] (nm)"])
#     / pavone_data["t[3] (s)"]
# )
# pavone_data["Retract speed (um/s)"].values


# plt.figure(figsize=(4, 4))
# for path in processed_paths:

#     plt.figure(figsize=(4, 4))
#     data = pd.read_csv(path)

#     time = data[PavoneKey.time]
#     speed = np.diff(data[PavoneKey.displacement], prepend=0) / np.diff(
#         data[PavoneKey.time], prepend=1
#     )
#     force = data[PavoneKey.force]
#     plt.plot(time, force, "-", alpha=0.05)
#     # plt.plot(time, speed, "-", alpha=1)

#     data["speed"] = speed
#     data["displacement"] = (
#         data[PavoneKey.displacement].rolling(window=20, center=True).mean()
#     )

#     dt = np.diff(data[PavoneKey.time], prepend=1)
#     ddis = np.diff(data[PavoneKey.displacement], prepend=0)
#     dforce = np.diff(data[PavoneKey.force], prepend=0)

#     # data["speed"].rolling(window=20, center=True).mean()

#     # plt.plot(data[PavoneKey.time], data[PavoneKey.displacement], "-", alpha=0.2)
#     # plt.plot(data[PavoneKey.time], data["displacement"], "-", alpha=1)

#     plt.plot(time, force, "-", alpha=0.2)
