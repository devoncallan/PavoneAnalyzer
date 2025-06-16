from typing import Tuple, List
import io
import pandas as pd
import matplotlib.pyplot as plt
import os


class PavoneKey:

    # Keys for raw Pavone data
    time = "Time (s)"
    load = "Load (uN)"
    indent = "Indentation (nm)"
    cantilever = "Cantilever (nm)"
    piezo = "Piezo (nm)"
    auxiliary = "Auxiliary"

    # Keys for processed Pavone data
    displacement = "Displacement (um)"
    force = "Force (uN)"


reduced_columns = [PavoneKey.time, PavoneKey.displacement, PavoneKey.force]


def find_all_files(directory, type="txt", exclude: List[str] = []) -> List[str]:
    all_files: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file = str(file)
            in_exclude_list = any([x in file.lower() for x in exclude])
            if file.endswith(f".{type}") and not in_exclude_list:
                relative_path = os.path.relpath(
                    os.path.join(root, file), start=directory
                )
                all_files.append(os.path.join(directory, relative_path))
    return all_files


def read_pavone_data(file_path: str) -> Tuple[dict, pd.DataFrame]:
    metadata = {}
    data_started = False
    data_lines = []
    data_header = []
    with open(file_path, "r") as file:
        for line in file:
            # line = line.decode('utf-8')  # Decode the line if necessary
            # Check if the data section has started
            if line.startswith(PavoneKey.time):
                data_started = True
                data_header = line.strip().split("\t")
                continue

            # If in the data section, append line to data_lines
            if data_started:
                data_lines.append(line.strip().split("\t"))
            else:
                # Process metadata
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]

    # Create DataFrame from data_lines
    data_df = pd.DataFrame(data_lines, columns=data_header)

    # Convert numeric columns to appropriate types
    for col in data_df.columns:
        try:
            data_df[col] = pd.to_numeric(data_df[col])
        except ValueError:
            pass  # Keep as string if conversion fails

    return metadata, data_df


def parse_pavone_filepath(file_path):
    print(file_path)
    parts = file_path.split("/")
    base_directory = parts[0]
    experiment_info = parts[1]
    plate_info = parts[2]
    scan_info = parts[3]
    filename = parts[4]

    experiment_details = experiment_info.split("_")
    date = experiment_details[0]
    experiment_code = "_".join(experiment_details[1:])

    plate_details = plate_info.split("_")
    plate_number = plate_details[0]
    well_number = plate_details[1]

    filename_details = filename.split("_")
    coordinates_info = filename.split(" ")[-4:]

    S = coordinates_info[0]
    X = coordinates_info[1]
    Y = coordinates_info[2]
    I = coordinates_info[3].split(".")[0]

    return {
        "filepath": file_path,
        "date": str(date),
        "experiment_code": experiment_code,
        "plate_number": plate_number,
        "well_number": well_number,
        "scan_info": scan_info,
        "S": S,
        "X": X,
        "Y": Y,
        "I": I,
        "filename": filename,
        "classification": int(-1),
    }


def create_new_classification_file(file_paths, output_filepath="classification.csv"):
    file_data = [parse_pavone_filepath(file_path) for file_path in file_paths]
    df = pd.DataFrame(file_data)
    df = df.sort_values(by=["experiment_code", "well_number"])
    return df


def get_classification_file(file_paths, class_file_path="classification.csv"):
    if os.path.exists(class_file_path):
        return pd.read_csv(class_file_path)

    df = create_new_classification_file(file_paths, class_file_path)
    df.to_csv(class_file_path, index=False)
    return df


def update_classification(df, index, classification=-1):
    if index in df.index:
        df.at[index, "classification"] = classification
    else:
        print(f"No row found at index {index}")
    return df


def locate_pavone_data(data_dir: str):
    all_files = find_all_files(data_dir, type="txt", exclude=["position", "test"])
    return all_files


##########################
### Plotting functions ###
##########################


def plot_pavone_data(df: pd.DataFrame, ax=None, savefig=True):
    if ax is None:
        fig, ax = plt.subplots(dpi=150)

    ax.plot(df[PavoneKey.time], df[PavoneKey.load], "-")
    ax.set_xlabel(PavoneKey.time)
    ax.set_ylabel(PavoneKey.load)
    plt.tight_layout()

    if savefig:
        img = io.BytesIO()
        plt.savefig(img, format="png")
        return fig, ax, img
    else:
        return fig, ax
