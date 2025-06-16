from typing import Optional

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

STYLES = {}


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
