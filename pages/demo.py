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

from ..components.load import load_pavone_data, load_experiment_data, directory_input


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

    from pavone.plot import plot_split_phase

    return plot_split_phase(
        approach_data=approach_data,
        dwell_data=dwell_data,
        retract_data=retract_data,
        contact_point=contact_point,
        pull_off_point=pull_off_point,
        plot_type="fvd",  # Force vs Displacement
        title=title,
        height=600,
        width=800,
    )


def main():
    st.set_page_config(page_title="Pavone Data Viewer", page_icon="🔬", layout="wide")

    st.title("🔬 Pavone Indentation Data Viewer")

    data_dir, processed_dir = directory_input(
        data_dir="/Users/devoncallan/Documents/GitHub/PavoneAnalyzer/test_data"
    )

    # Check if directories exist
    if not os.path.exists(data_dir):
        st.error(f"Data directory does not exist: {data_dir}")
        return

    # Load all pavone data (cached)
    pavone_data = load_pavone_data(data_dir, processed_dir)

    if pavone_data is None:
        st.error("Failed to load Pavone data.")
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
            ) = load_experiment_data(selected_row["output_filepath"], row_dict)

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
