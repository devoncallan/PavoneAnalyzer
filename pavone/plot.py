from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from pavone.types import PavoneKey

colors = {
    "approach": "#1f77b4",  # Blue
    "dwell": "#ff7f0e",  # Orange
    "retract": "#2ca02c",  # Green
    "contact": "#d62728",  # Red
    "pulloff": "#9467bd",  # Purple
}

STYLES = {
    "Approach": dict(
        mode="lines+markers",
        name="Approach",
        line=dict(color=colors["approach"], width=2),
        marker=dict(size=3),
        hovertemplate="<b>Approach</b><br>"
        + "Displacement: %{x:.3f} μm<br>"
        + "Force: %{y:.3f} μN<br>"
        + "<extra></extra>",
    ),
    "Dwell": dict(
        mode="lines+markers",
        name="Dwell",
        line=dict(color=colors["dwell"], width=2),
        marker=dict(size=3),
        hovertemplate="<b>Dwell</b><br>"
        + "Displacement: %{x:.3f} μm<br>"
        + "Force: %{y:.3f} μN<br>"
        + "<extra></extra>",
    ),
    "Retract": dict(
        mode="lines+markers",
        name="Retract",
        line=dict(color=colors["retract"], width=2),
        marker=dict(size=3),
        hovertemplate="<b>Retract</b><br>"
        + "Displacement: %{x:.3f} μm<br>"
        + "Force: %{y:.3f} μN<br>"
        + "<extra></extra>",
    ),
    "OverlayPrimary": dict(
        mode="lines",
        line=dict(
            color="black",
            width=1,
        ),
        opacity=0.9,
    ),
    "OverlaySecondary": dict(
        mode="lines",
        line=dict(
            color="gray",
            width=1,
        ),
        opacity=0.15,
    ),
    "Contact Point": dict(
        mode="markers",
        name="Contact Point",
        marker=dict(
            color=colors["contact"],
            size=10,
            symbol="circle",
            line=dict(width=2, color="white"),
        ),
        hovertemplate="<b>Contact Point</b><br>"
        + "Displacement: %{x:.3f} μm<br>"
        + "Force: %{y:.3f} μN<br>"
        + "<extra></extra>",
    ),
    "Pull-off Point": dict(
        mode="markers",
        name="Pull-off Point",
        marker=dict(
            color=colors["pulloff"],
            size=10,
            symbol="diamond",
            line=dict(width=2, color="white"),
        ),
        hovertemplate="<b>Pull-off Point</b><br>"
        + "Displacement: %{x:.3f} μm<br>"
        + "Force: %{y:.3f} μN<br>"
        + "<extra></extra>",
    ),
}


PLOT_STYLES = {
    "fvd": dict(
        xaxis=dict(title="Displacement (μm)", range=[-3, None]),
        yaxis=dict(title="Force (μN)"),
        title="Force vs Displacement",
    ),
    "fvt": dict(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Force (μN)"),
        title="Force vs Time",
    ),
    "dvt": dict(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Displacement (μm)"),
        title="Displacement vs Time",
    ),
}


def create_figure(plot_type: str = "fvd", **kwargs) -> go.Figure:

    # Common defaults across all plot types
    base_layout = dict(
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=80, r=120, t=80, b=80),
        dragmode="zoom",
    )
    if plot_type in PLOT_STYLES:
        type_layout = PLOT_STYLES[plot_type]
    else:
        print(
            f"Unsupported plot type: {plot_type}. Available types: {list(PLOT_STYLES.keys())}"
        )
        type_layout = {}

    # Merge: base_layout < type_layout < kwargs
    final_layout = {**base_layout, **type_layout, **kwargs}
    return go.Figure(layout=go.Layout(**final_layout))


def plot(
    fig: go.Figure,
    data: pd.DataFrame,
    plot_type: str = "fvd",
    style_key: Optional[str] = None,
    **kwargs,
):
    # Determine the x and y data based on the plot type
    if plot_type == "fvd":
        x = data[PavoneKey.displacement]
        y = data[PavoneKey.force]
    elif plot_type == "fvt":
        x = data[PavoneKey.time]
        y = data[PavoneKey.force]
    elif plot_type == "dvt":
        x = data[PavoneKey.time]
        y = data[PavoneKey.displacement]
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    # Start with empty base style
    base_style = {}

    # Get style from STYLES if provided
    style_dict = {}
    if style_key is not None:
        if style_key in STYLES:
            style_dict = STYLES[style_key]
        else:
            raise ValueError(
                f"Style key '{style_key}' not found in STYLES. Available keys: {list(STYLES.keys())}"
            )

    # Merge: base_style < style_dict < kwargs
    final_style = {**base_style, **style_dict, **kwargs}

    fig.add_trace(go.Scatter(x=np.array(x), y=np.array(y), **final_style))


def add_zero_lines(fig: go.Figure, x_zero: bool = True, y_zero: bool = True, **kwargs):
    """
    Add horizontal and/or vertical zero lines to the figure.

    """
    default_kwargs = {
        "line_dash": "dash",
        "line_color": "black",
        "line_width": 1,
        "opacity": 0.7,
    }
    plot_kwargs = {
        **default_kwargs,
        **kwargs,
    }  # Merge default with user-provided kwargs

    if x_zero:
        fig.add_vline(x=0, **plot_kwargs)
    if y_zero:
        fig.add_hline(y=0, **plot_kwargs)


#################################
### Figure Creation Functions ###
#################################


def plot_split_phase(
    approach_data: Optional[pd.DataFrame] = None,
    dwell_data: Optional[pd.DataFrame] = None,
    retract_data: Optional[pd.DataFrame] = None,
    contact_point: Optional[pd.Series] = None,
    pull_off_point: Optional[pd.Series] = None,
    plot_type: str = "fvd",
    **kwargs,
) -> go.Figure:
    """
    Create an interactive Plotly force vs displacement chart with phase separation.

    Parameters:
    -----------
    approach_data : pd.DataFrame, optional
        DataFrame containing approach phase data with 'displacement' and 'force' columns
    dwell_data : pd.DataFrame, optional
        DataFrame containing dwell phase data
    retract_data : pd.DataFrame, optional
        DataFrame containing retract phase data
    contact_point : pd.Series, optional
        Single point data for contact point
    pull_off_point : pd.Series, optional
        Single point data for pull-off point
    show_zero_lines : bool
        Whether to show horizontal and vertical zero reference lines

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """

    fig = create_figure(plot_type=plot_type, **kwargs)

    # Plot approach data
    if approach_data is not None:
        plot(fig, approach_data, plot_type=plot_type, style_key="Approach")

    if dwell_data is not None:
        plot(fig, dwell_data, plot_type=plot_type, style_key="Dwell")

    if retract_data is not None:
        plot(fig, retract_data, plot_type=plot_type, style_key="Retract")

    if contact_point is not None:
        plot(fig, contact_point, plot_type=plot_type, style_key="Contact Point")

    if pull_off_point is not None:
        plot(fig, pull_off_point, plot_type=plot_type, style_key="Pull-off Point")

    # Add zero reference lines
    if plot_type == "fvd":
        add_zero_lines(fig)

    return fig


def plot_exp_overlay(
    data_dict: Dict[str, List[Tuple[pd.DataFrame, ...]]],
    plot_type: str = "fvd",
    **kwargs,
) -> go.Figure:

    fig = create_figure(plot_type=plot_type, **kwargs)

    # colors = ["blue", "orange", "green", "red", "purple"]
    colors = ["#2E86AB", "#F18F01", "#2ca02c", "#A23B72", "#5D737E"]

    for i, (label, bundled_data_list) in enumerate(data_dict.items()):
        for trial_num, bundled_data in enumerate(bundled_data_list):

            data = bundled_data[0]
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for '{label}' must be a DataFrame.")

            # Plot the data with overlay style
            plot(
                fig,
                data,
                plot_type=plot_type,
                style_key="OverlayPrimary" if trial_num == 0 else "OverlaySecondary",
                name=f"{label} {trial_num}",
                line=dict(
                    color=colors[i % len(colors)],
                    width=1,
                ),
                # line_color=colors[i % len(colors)],
            )

    # Add zero reference lines
    if plot_type == "fvd":
        add_zero_lines(fig)

    return fig
