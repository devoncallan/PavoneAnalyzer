"""

Features:
- Reaction conditions: f1, f2, f3
- Experiment options: dwell time, retraction speed, depth
    - Dwell time: low, medium, high
    - Retraction speed: low, medium, high
    - Depth: low, medium, high

Outputs:
- Work of adhesion (Wad)
-




Adding columns
- Potentially classification for reaction conditions
- Batch number
- Experiment number



Visualization:
- Plotting the Wad or pull-off force vs.

"""

from typing import Dict, Tuple, Optional
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pavone.plot import plot_force_vs_displacement_plotly
from pavone.process import load_pavone_data, process_data
from pavone.experiment import split_by_phase


@st.cache_data
def load_all_pavone_data(data_dir: str, processed_dir: str) -> pd.DataFrame:
    """
    Cache the initial data loading and processing.
    This will only run once unless the directories change.
    """
    return load_pavone_data(data_dir, processed_dir)


@st.cache_data
def load_and_process_single_file(
    filepath: str, row_dict: dict
) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Cache the loading and processing of individual files.
    Uses filepath as the cache key so each file is only processed once.
    """
    # Convert row_dict back to Series for compatibility
    row = pd.Series(row_dict)

    # Load the raw data
    data = pd.read_csv(filepath)

    # Process the data to get contact and pull-off points
    data, contact_point, pull_off_point = process_data(data, row)

    # Split into phases
    approach_data, dwell_data, retract_data = split_by_phase(data, row)

    return data, contact_point, pull_off_point, approach_data, dwell_data, retract_data


# @st.cache_data
def create_plotly_figure(
    approach_data: Optional[pd.DataFrame],
    dwell_data: Optional[pd.DataFrame],
    retract_data: Optional[pd.DataFrame],
    contact_point: Optional[pd.Series],
    pull_off_point: Optional[pd.Series],
    title: str,
) -> go.Figure:
    """
    Cache the figure creation. This prevents recreating the same plot multiple times.
    """
    return plot_force_vs_displacement_plotly(
        approach_data=approach_data,
        dwell_data=dwell_data,
        retract_data=retract_data,
        contact_point=contact_point,
        pull_off_point=pull_off_point,
        title=title,
    )


def main():
    st.set_page_config(page_title="Pavone Data Viewer", page_icon="🔬", layout="wide")

    st.title("🔬 Pavone Indentation Data Viewer")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Directory inputs
        data_dir = st.text_input(
            "Data Directory",
            value="/Users/devoncallan/Documents/GitHub/PavoneAnalyzer/test_data",
            help="Path to your raw Pavone data directory",
        )

        processed_dir = st.text_input(
            "Processed Directory",
            value="/Users/devoncallan/Documents/GitHub/PavoneAnalyzer/test_data/processed",
            help="Path to store/find processed data",
        )

        # Load data button
        if st.button("🔄 Load/Refresh Data", type="primary"):
            # Clear cache to force reload
            st.cache_data.clear()
            st.rerun()

    # Check if directories exist
    if not os.path.exists(data_dir):
        st.error(f"Data directory does not exist: {data_dir}")
        return

    # Load all pavone data (cached)
    try:
        with st.spinner("Loading Pavone data..."):
            pavone_data = load_all_pavone_data(data_dir, processed_dir)

        st.success(f"Loaded {len(pavone_data)} files")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    if len(pavone_data) == 0:
        st.warning("No data files found. Check your directory paths.")
        return

    # Create file selection interface
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create display names for the selectbox (you can customize this)
        display_names = []
        for i, row in pavone_data.iterrows():
            filename = Path(row["output_filepath"]).stem
            # You can add more info here based on your metadata
            display_name = f"{i:03d}: {filename}"
            display_names.append(display_name)

        selected_display = st.selectbox(
            "Select a file to view:",
            options=display_names,
            index=0,
            help="Choose a Pavone data file to analyze",
        )

        # Extract the index from the selected display name
        selected_index = int(selected_display.split(":")[0])

    with col2:
        # Display file info
        selected_row = pavone_data.iloc[selected_index]
        st.info(f"**File:** {Path(selected_row['output_filepath']).name}")

        # Add navigation buttons
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("⬅️ Previous") and selected_index > 0:
                st.session_state.selected_index = selected_index - 1
                st.rerun()
        with col2b:
            if st.button("➡️ Next") and selected_index < len(pavone_data) - 1:
                st.session_state.selected_index = selected_index + 1
                st.rerun()

    # Load and process the selected file (cached)
    try:
        with st.spinner("Processing selected file..."):
            # Convert row to dict for caching (Series aren't hashable)
            row_dict = selected_row.to_dict()

            (
                data,
                contact_point,
                pull_off_point,
                approach_data,
                dwell_data,
                retract_data,
            ) = load_and_process_single_file(selected_row["output_filepath"], row_dict)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["📊 Force vs Displacement", "📈 All Plots", "📋 Data Summary"]
    )

    with tab1:
        st.subheader("Force vs Displacement")

        # Options for the plot
        col1, col2, col3 = st.columns(3)
        with col1:
            show_approach = st.checkbox("Show Approach", value=True)
        with col2:
            show_dwell = st.checkbox("Show Dwell", value=dwell_data is not None)
        with col3:
            show_retract = st.checkbox("Show Retract", value=True)

        # Create the figure (cached)
        title = f"Force vs Displacement - {Path(selected_row['output_filepath']).stem}"
        fig = create_plotly_figure(
            approach_data=approach_data if show_approach else None,
            dwell_data=dwell_data if show_dwell else None,
            retract_data=retract_data if show_retract else None,
            contact_point=contact_point,
            pull_off_point=pull_off_point,
            title=title,
        )

        # Display the plot
        st.plotly_chart(fig, key=title)

        # Add download button for the plot
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Plot as HTML"):
                html_str = fig.to_html()
                st.download_button(
                    label="Download HTML",
                    data=html_str,
                    file_name=f"{Path(selected_row['output_filepath']).stem}_plot.html",
                    mime="text/html",
                )
    st.write(selected_row)


if __name__ == "__main__":
    main()
