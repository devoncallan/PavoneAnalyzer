from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd
from streamlit.delta_generator import DeltaGenerator


def protocol_select(c: DeltaGenerator, multi: bool = False):

    c1, c2 = c.columns([1, 1])
    selection_mode = "multi" if multi else "single"
    depth_selections = c1.pills(
        "Depth (μm)",
        options=[1.0, 2.0, 3.0],
        selection_mode=selection_mode,
        default=[2.0],
    )
    if not depth_selections:
        c2.error("Please select depth.")
    dwell_selections = c.pills(
        "Dwell time (s)",
        options=[1, 10, 100],
        selection_mode=selection_mode,
        default=[1],
    )
    if not dwell_selections:
        c2.error("Please select dwell time.")
    retract_speed_selections = c1.pills(
        "Retraction speed (μm/s)",
        options=[0.2, 2.0, 20.0],
        selection_mode=selection_mode,
        default=[2.0],
    )
    if not retract_speed_selections:
        c2.error("Please select retraction speed.")

    if not multi:
        return [depth_selections], [dwell_selections], [retract_speed_selections]

    return depth_selections, dwell_selections, retract_speed_selections


def sample_id_select(
    c: DeltaGenerator, sample_ids: Optional[List[str]] = None, multi: bool = False
) -> List[str]:
    max_selections = 1 if not multi else None
    sample_ids = c.multiselect(
        "Sample ID", options=sample_ids, max_selections=max_selections
    )
    if not sample_ids:
        c.error("Please select at least one sample ID.")
        return []

    if not multi:
        return [sample_ids]

    return sample_ids


def selection_panel(
    c: DeltaGenerator,
    summary_data: pd.DataFrame,
    multi: bool = False,
) -> pd.DataFrame:
    """
    Create a selection panel for experimental conditions and experiment IDs.

    Parameters:
        c (DeltaGenerator): Streamlit component container.
        exp_ids (Optional[List[str]]): List of experiment IDs to select from.
        multi (bool): Whether to allow multiple selections.

    Returns:
        Dict[str, Any]: Dictionary containing selected conditions and experiment IDs.
    """

    sample_ids = summary_data["sample_id"].unique()

    depth_select, dwell_select, retract_speed_select = protocol_select(c, multi=multi)
    sample_id = sample_id_select(c, sample_ids=sample_ids, multi=not multi)

    from pavone.experiment import filter_experiments

    filtered_summary_data = filter_experiments(
        summary_data, sample_ids, depth_select, dwell_select, retract_speed_select
    )

    return filtered_summary_data
