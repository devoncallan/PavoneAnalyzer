def calculate_Wad():
    pass


def shift_origin():

    min_depth_um = 1


def process_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:

    F_normal_uN = raw_data[PavoneKey.load].to_numpy()  # Normal force [μN]
    d_indent_um = raw_data[PavoneKey.indent].to_numpy() / 1000  # Indentation [μm]
    d_cantilever_um = (
        raw_data[PavoneKey.cantilever].to_numpy() / 1000
    )  # Cantilever deflection [μm]
    d_piezo_um = raw_data[PavoneKey.piezo].to_numpy() / 1000  # Piezo deflection [μm]

    z_stage = d_piezo_um - d_cantilever_um  # Absolute probe position of the Pavone

    data = pd.DataFrame(columns=reduced_columns)
    data[PavoneKey.time] = raw_data[PavoneKey.time]
    data[PavoneKey.displacement] = z_stage
    data[PavoneKey.force] = F_normal_uN

    return data


def split_approach_retract(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Get the index of the maximum force
    max_force_idx = data[PavoneKey.force].idxmax()
    max_disp_idx = data[PavoneKey.displacement].idxmax()

    split_idx = min(max_force_idx, max_disp_idx)

    approach_data = data.iloc[: split_idx + 1].copy().reset_index()
    retract_data = data.iloc[split_idx + 1 :].copy().reset_index()

    return approach_data, retract_data


def find_contact_and_pull_off_point_idx(
    approach_data: pd.DataFrame, retract_data: pd.DataFrame
) -> Tuple[int, int]:
    pass


# def calculate(approach):


def process_raw_pavone_dataset(raw_data_dir: str, output_dir: str):

    # Locate all Pavone data files
    # Process all data files
    # Output processed data as csv to a new directory

    pass
