import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from typing import Dict, Any, Optional, List


def exp_condition_select(c: DeltaGenerator):

    depth_selections = c.pills(
        "Depth (μm)", options=[1.0, 2.0, 3.0], selection_mode="multi"
    )
    dwell_selections = c.pills(
        "Dwell time (s)", options=[0.1, 0.2, 0.3], selection_mode="multi"
    )
    retract_speed_selections = c.pills(
        "Retraction speed (μm/s)", options=[0.2, 2.0, 20.0], selection_mode="multi"
    )
    return depth_selections, dwell_selections, retract_speed_selections
