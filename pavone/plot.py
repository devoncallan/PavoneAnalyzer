from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from pavone.types import PavoneKey


def plot_force_vs_time(ax: Axes, data: pd.DataFrame, label: str = "", **kwargs):
    ax.plot(data[PavoneKey.time], data[PavoneKey.force], label=label, **kwargs)
    ax.plot(
        data[PavoneKey.time],
        np.zeros_like(data[PavoneKey.time]),
        color="black",
        linestyle="--",
    )
    ax.set_ylim(-0.01, 0.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (μN)")
    ax.legend()


def plot_force_gradient_vs_time(
    ax: Axes,
    data: pd.DataFrame,
    label: str = "",
    **kwargs,
):
    # force_gradient = np.gradient(data[PavoneKey.force], data[PavoneKey.time])
    # force_gradient = savgol_smoothing(force_gradient)
    ax.plot(data[PavoneKey.time], data[PavoneKey.force_gradient], label=label, **kwargs)
    ax.plot(
        data[PavoneKey.time],
        np.zeros_like(data[PavoneKey.time]),
        color="black",
        linestyle="--",
    )
    # ax.set_ylim(-0.01, 0.01)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Gradient (μN/s)")
    ax.legend()


def plot_displacement_vs_time(ax: Axes, data: pd.DataFrame, label: str = "", **kwargs):
    ax.plot(data[PavoneKey.time], data[PavoneKey.displacement], label=label, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (μm)")
    ax.legend()


def plot_force_vs_displacement(ax: Axes, data: pd.DataFrame, label: str = "", **kwargs):
    ax.plot(data[PavoneKey.displacement], data[PavoneKey.force], label=label, **kwargs)
    ax.plot(
        data[PavoneKey.displacement],
        np.zeros_like(data[PavoneKey.displacement]),
        color="black",
        linestyle="--",
    )
    ax.set_xlim(left=-5, right=5)
    ax.set_xlabel("Displacement (μm)")
    ax.set_ylabel("Force (μN)")
    ax.legend()


def plot_data_summary(
    approach_data: Optional[pd.DataFrame] = None,
    dwell_data: Optional[pd.DataFrame] = None,
    retract_data: Optional[pd.DataFrame] = None,
    contact_point: Optional[pd.Series] = None,
    pull_off_point: Optional[pd.Series] = None,
):

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    if approach_data is not None:

        plot_force_vs_time(axs[0], approach_data, label="Approach")
        plot_force_gradient_vs_time(axs[1], approach_data, label="Approach")
        plot_displacement_vs_time(axs[2], approach_data, label="Approach")
        plot_force_vs_displacement(axs[3], approach_data, label="Approach")

        if contact_point is not None:
            plot_force_vs_time(
                axs[0],
                contact_point,
                label="Contact Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_force_gradient_vs_time(
                axs[1],
                contact_point,
                label="Contact Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_displacement_vs_time(
                axs[2],
                contact_point,
                label="Contact Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_force_vs_displacement(
                axs[3],
                contact_point,
                label="Contact Point",
                color="red",
                markersize=5,
                marker="o",
            )

    if dwell_data is not None:
        plot_force_vs_time(axs[0], dwell_data, label="Dwell")
        plot_force_gradient_vs_time(axs[1], dwell_data, label="Dwell")
        plot_displacement_vs_time(axs[2], dwell_data, label="Dwell")
        plot_force_vs_displacement(axs[3], dwell_data, label="Dwell")
    if retract_data is not None:

        plot_force_vs_time(axs[0], retract_data, label="Retract")
        plot_force_gradient_vs_time(axs[1], retract_data, label="Retract")
        plot_displacement_vs_time(axs[2], retract_data, label="Retract")
        plot_force_vs_displacement(axs[3], retract_data, label="Retract")

        if pull_off_point is not None:
            print(pull_off_point)
            plot_force_vs_time(
                axs[0],
                pull_off_point,
                label="Pull-Off Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_force_gradient_vs_time(
                axs[1],
                pull_off_point,
                label="Pull-Off Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_displacement_vs_time(
                axs[2],
                pull_off_point,
                label="Pull-Off Point",
                color="red",
                markersize=5,
                marker="o",
            )
            plot_force_vs_displacement(
                axs[3],
                pull_off_point,
                label="Pull-Off Point",
                color="red",
                markersize=5,
                marker="o",
            )

    plt.tight_layout()
    plt.show()


def plot_force_vs_displacement_plotly(
    approach_data: Optional[pd.DataFrame] = None,
    dwell_data: Optional[pd.DataFrame] = None,
    retract_data: Optional[pd.DataFrame] = None,
    contact_point: Optional[pd.Series] = None,
    pull_off_point: Optional[pd.Series] = None,
    title: str = "Force vs Displacement",
    show_zero_lines: bool = True,
    height: int = 600,
    width: int = 800,
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
    title : str
        Chart title
    show_zero_lines : bool
        Whether to show horizontal and vertical zero reference lines
    height : int
        Chart height in pixels
    width : int
        Chart width in pixels

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """

    fig = go.Figure()

    # Color scheme for different phases
    colors = {
        "approach": "#1f77b4",  # Blue
        "dwell": "#ff7f0e",  # Orange
        "retract": "#2ca02c",  # Green
        "contact": "#d62728",  # Red
        "pulloff": "#9467bd",  # Purple
    }

    n = 10

    # Plot approach data
    if approach_data is not None:
        print("Plotting approach data")
        fig.add_trace(
            go.Scatter(
                x=approach_data[PavoneKey.displacement][::n],
                y=approach_data[PavoneKey.force][::n],
                mode="lines+markers",
                name="Approach",
                line=dict(color=colors["approach"], width=2),
                marker=dict(size=3),
                hovertemplate="<b>Approach</b><br>"
                + "Displacement: %{x:.3f} μm<br>"
                + "Force: %{y:.3f} μN<br>"
                + "<extra></extra>",
            )
        )

    # Plot dwell data
    if dwell_data is not None:
        print("Plotting dwell data")
        fig.add_trace(
            go.Scatter(
                x=dwell_data[PavoneKey.displacement][::n],
                y=dwell_data[PavoneKey.force][::n],
                mode="lines+markers",
                name="Dwell",
                line=dict(color=colors["dwell"], width=2),
                marker=dict(size=3),
                hovertemplate="<b>Dwell</b><br>"
                + "Displacement: %{x:.3f} μm<br>"
                + "Force: %{y:.3f} μN<br>"
                + "<extra></extra>",
            )
        )

    # Plot retract data
    if retract_data is not None:
        fig.add_trace(
            go.Scatter(
                x=retract_data[PavoneKey.displacement][::n],
                y=retract_data[PavoneKey.force][::n],
                mode="lines+markers",
                name="Retract",
                line=dict(color=colors["retract"], width=2),
                marker=dict(size=3),
                hovertemplate="<b>Retract</b><br>"
                + "Displacement: %{x:.3f} μm<br>"
                + "Force: %{y:.3f} μN<br>"
                + "<extra></extra>",
            )
        )

    # Add contact point
    if contact_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[contact_point[PavoneKey.displacement]],
                y=[contact_point[PavoneKey.force]],
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
            )
        )

    # Add pull-off point
    if pull_off_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[pull_off_point[PavoneKey.displacement]],
                y=[pull_off_point[PavoneKey.force]],
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
            )
        )

    # Add zero reference lines
    if show_zero_lines:
        # Horizontal line at y=0
        fig.add_hline(
            y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7
        )

        # Vertical line at x=0
        fig.add_vline(
            x=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title="Displacement (μm)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            range=[-5, 5],  # Force the x-axis range
            autorange=False,  # Disable auto range to keep the specified range
        ),
        yaxis=dict(
            title="Force (μN)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="closest",
        height=height,
        width=width,
        template="plotly_white",
        margin=dict(l=80, r=120, t=80, b=80),
    )

    # Add pan and zoom tools
    fig.update_layout(dragmode="zoom", showlegend=True)

    return fig
