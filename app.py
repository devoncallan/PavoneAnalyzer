import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit.delta_generator import DeltaGenerator
from typing import List, Optional

from components.select import selection_panel

# Mock data for demonstration
def create_mock_data():
    """Create some mock data to demonstrate the UI"""
    samples = ["Sample_A", "Sample_B", "Sample_C", "Sample_D", "Sample_E"]
    protocols = [
        "depths_1um",
        "depths_2um",
        "depths_3um",
        "dwell_10s",
        "dwell_100s",
        "retractSpeed_0p2",
        "retractSpeed_20",
    ]

    mock_experiments = []
    for sample in samples:
        for protocol in protocols:
            # 2-3 replicates per sample+protocol
            for rep in range(2):
                mock_experiments.append(
                    {
                        "sample_id": sample,
                        "protocol": protocol,
                        "replicate": rep,
                        "depth_um": (
                            1.0
                            if "depths_1um" in protocol
                            else (
                                2.0
                                if "depths_2um" in protocol
                                else 3.0 if "depths_3um" in protocol else 2.0
                            )
                        ),
                        "dwell_s": (
                            10.0
                            if "dwell_10s" in protocol
                            else 100.0 if "dwell_100s" in protocol else 1.0
                        ),
                        "retract_speed_um_s": (
                            0.2
                            if "retractSpeed_0p2" in protocol
                            else 20.0 if "retractSpeed_20" in protocol else 2.0
                        ),
                    }
                )

    return pd.DataFrame(mock_experiments)


def create_mock_plot():
    """Create a simple mock plot"""
    fig = go.Figure()

    # Add some mock traces
    import numpy as np

    x = np.linspace(-3, 3, 100)

    for i, name in enumerate(["Condition 1", "Condition 2", "Condition 3"]):
        y = np.sin(x + i) * np.exp(-(x**2) / 5) * (i + 1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2)))

    fig.update_layout(
        title="Force vs Displacement",
        xaxis_title="Displacement (μm)",
        yaxis_title="Force (μN)",
        height=500,
        template="plotly_white",
    )

    return fig


def main():
    st.set_page_config(page_title="Pavone Analysis", layout="wide")

    # Load mock data
    df = create_mock_data()

    st.title("🔬 Pavone Analysis Tool")

    # Mode selection
    analysis_mode = st.radio(
        "Analysis Mode:",
        ["Single Protocol, Multiple Samples", "Single Sample, Multiple Protocols"],
        horizontal=True,
    )

    # st.divider()

    # Main layout: plot on left, controls on right
    col_plot, col_controls = st.columns([2, 1])

    with col_plot:
        st.subheader("Results")

        # Mock plot for now
        fig = create_mock_plot()
        st.plotly_chart(fig, use_container_width=True)

        # Simple summary
        st.info(f"📊 Showing 3 conditions • Mode: {analysis_mode}")

    with col_controls:
        st.subheader("Controls")

        # Determine multi-selection behavior based on mode
        if analysis_mode == "Single Protocol, Multiple Samples":
            sample_multi = True
            protocol_multi = False
        else:  # Single Sample, Multiple Protocols
            sample_multi = False
            protocol_multi = True

        # Sample selection using component function
        sample_options = df["sample_id"].unique().tolist()
        selected_samples = exp_id_select(st, exp_ids=sample_options, multi=sample_multi)

        st.divider()

        # Protocol selection using component function
        st.markdown("**Select Protocols:**")
        depth_selection, dwell_selection, retract_selection = exp_condition_select(
            st, multi=protocol_multi
        )

        st.divider()

        # Show current selections
        st.markdown("**Current Selection:**")
        st.write(f"Samples: {len(selected_samples)} selected")

        # Count active protocols based on selections
        active_protocols = []
        active_protocols.extend([f"{d}μm depth" for d in depth_selection])
        active_protocols.extend([f"{d}s dwell" for d in dwell_selection])
        active_protocols.extend([f"{r}μm/s retract" for r in retract_selection])

        if not active_protocols:
            st.write("Protocol: Baseline (2μm, 1s, 2μm/s)")
        else:
            st.write(f"Protocols: {len(active_protocols)} selected")
            for protocol in active_protocols:
                st.write(f"  • {protocol}")


if __name__ == "__main__":
    main()
