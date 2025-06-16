from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

from pavone.types import PavoneKey, reduced_columns


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


def load_pavone_data(data_dir: str, output_dir: str):
    from pavone.parse import read_pavone_data, parse_pavone_filepath, locate_pavone_data
    from pavone.experiment import add_phase_timings, add_experimental_conditions

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

    metadata_df = add_phase_timings(metadata_df)

    metadata_df = add_experimental_conditions(metadata_df)

    return metadata_df


def process_data(raw_data: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:

    from pavone.signal import baseline_correct, savgol_smoothing
    from pavone.experiment import split_by_phase
    from pavone.contact import (
        get_contact_point,
        get_pull_off_point,
        zero_contact_point,
    )

    data = raw_data.copy()

    # 1. Initial baseline correction (conservative)
    data = data[data[PavoneKey.time] >= 2]
    data = baseline_correct(data, metadata)

    force_gradient = np.gradient(data[PavoneKey.force], data[PavoneKey.time])
    data[PavoneKey.force_gradient] = savgol_smoothing(force_gradient, window_len=501)

    # 2. Estimate contact point and pull-off point
    approach_data, _, _ = split_by_phase(data, metadata)
    contact_point = get_contact_point(approach_data)

    data, metadata, contact_point = zero_contact_point(data, metadata, contact_point)

    _, _, retract_data = split_by_phase(data, metadata)
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
