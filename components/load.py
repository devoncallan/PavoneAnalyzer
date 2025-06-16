import os
from typing import Tuple, Optional

import streamlit as st
import pandas as pd


@st.cache_data
def load_pavone_data(data_dir: str, processed_dir: str) -> Optional[pd.DataFrame]:
    """
    Load and process all Pavone data files from the specified directories.
    Caches the result to avoid reloading unless the directories change.
    """
    from pavone.process import load_pavone_data

    try:
        with st.spinner("Loading Pavone data..."):
            pavone_data = load_pavone_data(data_dir, processed_dir)

        with st.sidebar:
            st.success(f"Loaded {len(pavone_data)} files")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    if len(pavone_data) == 0:
        st.warning("No data files found. Check your directory paths.")
        return

    return pavone_data


@st.cache_data
def load_experiment_data(
    filepath: str, row_dict: dict
) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Cache the loading and processing of individual files.
    Uses filepath as the cache key so each file is only processed once.
    """
    from pavone.process import process_data
    from pavone.experiment import split_by_phase

    # Convert row_dict back to Series for compatibility
    row = pd.Series(row_dict)

    # Load the raw data
    data = pd.read_csv(filepath)

    # Downsample the data
    data = data.iloc[::10, :]  # Downsample by taking every 10th row

    # Process the data to get contact and pull-off points
    data, contact_point, pull_off_point = process_data(data, row)

    # Split into phases
    approach_data, dwell_data, retract_data = split_by_phase(data, row)

    return data, contact_point, pull_off_point, approach_data, dwell_data, retract_data


def directory_input(
    data_dir: str = "/Users/devoncallan/Documents/GitHub/PavoneAnalyzer/test_data",
) -> Optional[Tuple[str, str]]:
    """
    Create input fields for data and processed directories.
    Returns the selected directories.
    """
    st.sidebar.header("Data Source")
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=data_dir,
        help="Directory containing raw Pavone data files",
    )
    processed_dir = os.path.join(data_dir, "processed")

    # Load data button
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        # Clear cache to force reload
        st.cache_data.clear()
        st.rerun()
        return

    return data_dir, processed_dir
