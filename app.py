import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit.delta_generator import DeltaGenerator
from typing import List, Optional
import os

from components.select import selection_panel
from components.load import (
    load_pavone_data,
    load_experiments_data,
    directory_input,
)
from pavone.experiment import extract_data_for_plotting
from pavone.plot import plot_exp_overlay


def main():
    st.set_page_config(page_title="Pavone Analysis", layout="wide")

    st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")
    st.markdown(
        """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Pavone Data Visualizer 🔬 ")

    # Get directories
    data_dir, processed_dir = directory_input(
        # data_dir="/Users/devoncallan/Documents/GitHub/PavoneAnalyzer/test_data"
        data_dir="./test_data",
    )

    # Check if directories exist
    if not os.path.exists(data_dir):
        st.error(f"Data directory does not exist: {data_dir}")
        return

    # Load all pavone data (cached)
    summary_data = load_pavone_data(data_dir, processed_dir)

    if summary_data is None:
        st.error("Failed to load Pavone data.")
        return

    # Main layout: plot on left, controls on right
    col_plot, col_controls = st.columns([2, 1])

    with col_controls:
        # st.subheader("Controls")

        # Use selection panel to get filtered data
        analysis_mode, filtered_summary_data = selection_panel(
            st.container(),
            summary_data,
        )

        st.divider()

    with col_plot:
        # st.subheader("Results")

        if not filtered_summary_data.empty:
            try:
                with st.spinner("Loading experiment data..."):
                    # Load individual experiments
                    experiments_data = load_experiments_data(filtered_summary_data)

                    if experiments_data:
                        # Extract data for plotting based on analysis mode
                        if analysis_mode == "Vary Samples":
                            data_dict = extract_data_for_plotting(
                                experiments_data, group_by="sample_id"
                            )
                        else:  # Vary Protocols
                            data_dict = extract_data_for_plotting(
                                experiments_data, group_by="protocol"
                            )

                        # Create overlay plot
                        if data_dict:
                            fig = plot_exp_overlay(
                                data_dict,
                                plot_type="fvd",
                                title=f"Force vs Displacement - {analysis_mode}",
                                # height=500,
                                # width=800,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data available for plotting.")
                    else:
                        st.warning("Failed to load experiment data.")

            except Exception as e:
                st.error(f"Error processing experiments: {str(e)}")
                # Show stack trace in expander for debugging
                with st.expander("Error details"):
                    st.exception(e)

        else:
            st.warning(
                "No experiments match your selection criteria. Please adjust your filters."
            )
            return


if __name__ == "__main__":
    main()
