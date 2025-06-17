from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd
from streamlit.delta_generator import DeltaGenerator


def protocol_select(c: DeltaGenerator, multi: bool = False, **kwargs):

    # c1, c2 = c.columns([1, 1])
    c1 = c.container()
    c2 = c.container()
    selection_mode = "multi" if multi else "single"
    depth_selections = c1.pills(
        "Depth (μm)",
        options=[1.0, 2.0, 3.0],
        selection_mode=selection_mode,
        default=[2.0],
        **kwargs,
    )
    if not depth_selections:
        c2.error("Please select depth.")

    dwell_selections = c.pills(
        "Dwell time (s)",
        options=[1, 10, 100],
        selection_mode=selection_mode,
        default=[1],
        **kwargs,
    )
    if not dwell_selections:
        c2.error("Please select dwell time.")

    retract_speed_selections = c1.pills(
        "Retraction speed (μm/s)",
        options=[0.2, 2.0, 20.0],
        selection_mode=selection_mode,
        default=[2.0],
        **kwargs,
    )
    if not retract_speed_selections:
        c2.error("Please select retraction speed.")

    if not multi:
        return [depth_selections], [dwell_selections], [retract_speed_selections]

    return depth_selections, dwell_selections, retract_speed_selections


def sample_id_select(
    c: DeltaGenerator,
    sample_ids: Optional[List[str]] = None,
    multi: bool = False,
    **kwargs
) -> List[str]:
    
    
    max_selections = 1 if not multi else None
    default_sample_id = sample_ids[0] if sample_ids is not None else None
    selected_sample_ids = c.multiselect(
        "Sample ID",
        options=sample_ids,
        default=default_sample_id,
        max_selections=max_selections,
        label_visibility="collapsed",
        **kwargs,
    )
    if not selected_sample_ids:
        c.error("Please select at least one sample ID.")
        return []

    selected_sample_ids = list(selected_sample_ids)

    if not multi:
        return selected_sample_ids

    return selected_sample_ids


def selection_panel(
    c: DeltaGenerator,
    summary_data: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """
    Create a selection panel for experimental conditions and experiment IDs.

    Parameters:
        c (DeltaGenerator): Streamlit component container.
        exp_ids (Optional[List[str]]): List of experiment IDs to select from.
        multi (bool): Whether to allow multiple selections.

    Returns:
        Dict[str, Any]: Dictionary containing selected conditions and experiment IDs.
    """

    # analysis_mode = c.radio(
    #     "Analysis Mode:",
    #     ["Single Protocol, Multiple Samples", "Single Sample, Multiple Protocols"],
    #     horizontal=True,
    # )
    c1, c2 = c.columns([1, 2])
    c1.markdown("##### Analysis Mode")

    analysis_mode = c2.segmented_control(
        "Analysis Mode:",
        ["Vary Samples", "Vary Protocols"],
        label_visibility="collapsed",
        # horizontal=True,
        selection_mode="single",
        default="Vary Samples",
    )

    # Determine multi-selection behavior based on mode
    if analysis_mode == "Vary Samples":
        protocol_multi = False
        sample_multi = True
        sample_label = "Select Multiple Samples"
        protocol_label = "Select One Protocol"
    else:  # Vary Protocols
        protocol_multi = True
        sample_multi = False
        sample_label = "Select One Sample"
        protocol_label = "Select Multiple Protocols"
    # Get unique sample IDs from summary dat
    
    c = c.container()
    

    sample_ids = summary_data["sample_id"].unique()

    # sample_select = sample_id_select(c, sample_ids=sample_ids, multi=sample_multi)

    c.markdown("")
    c.markdown(
        f"##### {sample_label}",
    )
    sample_select = sample_id_select(c, sample_ids=sample_ids, multi=sample_multi)

    c.markdown("")
    c.markdown(f"##### {protocol_label}")
    depth_select, dwell_select, retract_speed_select = protocol_select(
        c, multi=protocol_multi
    )

    from pavone.experiment import filter_experiments

    filtered_summary_data = filter_experiments(
        summary_data, sample_select, depth_select, dwell_select, retract_speed_select
    )

    return analysis_mode, filtered_summary_data
