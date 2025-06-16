# def create_plotly_figure(
#     approach_data: Optional[pd.DataFrame],
#     dwell_data: Optional[pd.DataFrame],
#     retract_data: Optional[pd.DataFrame],
#     contact_point: Optional[pd.Series],
#     pull_off_point: Optional[pd.Series],
#     title: str,
# ) -> go.Figure:
#     """
#     Cache the figure creation. This prevents recreating the same plot multiple times.
#     """

#     from pavone.plot import plot_split_phase

#     return plot_split_phase(
#         approach_data=approach_data,
#         dwell_data=dwell_data,
#         retract_data=retract_data,
#         contact_point=contact_point,
#         pull_off_point=pull_off_point,
#         plot_type="fvd",  # Force vs Displacement
#         title=title,
#         height=600,
#         width=800,
#     )


# def plot_overlay()