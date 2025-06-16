from typing import Tuple, Dict, List
from pathlib import Path
import pandas as pd

from pavone.types import PavoneKey, reduced_columns


def read_pavone_data(filepath: str) -> Tuple[dict, pd.DataFrame]:
    metadata = {}
    data_started = False
    data_lines = []
    data_header = []

    with open(filepath, "r") as file:
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
                continue

            # Process metadata
            parts = line.strip().split("\t")
            if len(parts) % 2 != 0:
                continue

            for i in range(0, len(parts), 2):
                key = parts[i]
                value = parts[i + 1]
                if not key or not value:
                    continue
                metadata[key] = value

    # Create DataFrame from data_lines
    raw_data = pd.DataFrame(data_lines, columns=data_header)

    # Convert numeric columns to appropriate types
    for col in raw_data.columns:
        try:
            raw_data[col] = pd.to_numeric(raw_data[col])
        except ValueError:
            pass  # Keep as string if conversion fails

    for key, value in metadata.items():
        try:
            metadata[key] = pd.to_numeric(value)
        except ValueError:
            pass

    return metadata, raw_data


def parse_pavone_filepath(filepath: str) -> Dict[str, str]:
    """
    Parse the file path of a Pavone data file to extract metadata.

    Args:
        filepath (str): The file path of the Pavone data file.

    Returns:
        dict: A dictionary containing parsed metadata.
    """

    filepath: Path = Path(filepath)

    filename = filepath.stem
    scan_info = filepath.parent.name
    plate_info = filepath.parent.parent.name
    exp_info = filepath.parent.parent.parent.name

    filename_details, coordinates_str = filename.split(" ", 1)
    S, X, Y, I = coordinates_str.split(" ")
    plate_num, well_num = plate_info.split("_", 1)
    date, exp_code = exp_info.split("_", 1)

    return {
        "filepath": str(filepath),
        "date": date,
        "experiment_code": exp_code,
        "plate_number": plate_num,
        "well_number": well_num,
        "sample_id": "_".join([plate_num, well_num]),
        "scan_info": scan_info,
        "S": S,
        "X": X,
        "Y": Y,
        "I": I,
        "filename": filename,
    }


def parse_pavone_files(filepaths: List[str]) -> pd.DataFrame:
    """
    Parse multiple Pavone data files and return a DataFrame with metadata.

    Args:
        filepaths (list of str): List of file paths to the Pavone data files.

    Returns:
        pd.DataFrame: DataFrame containing parsed metadata for each file.
    """

    data = [parse_pavone_filepath(fp) for fp in filepaths]
    return pd.DataFrame(data)


def find_all_files(
    data_dir: str, type: str = "txt", exclude: List[str] = []
) -> List[str]:
    """
    Find all files in the given directory with the specified type, excluding certain patterns.

    Args:
        data_dir (str): Directory to search for files.
        type (str): File type to search for (default is "txt").
        exclude (list of str): List of substrings to exclude from file names.

    Returns:
        list of str: List of file paths that match the criteria.
    """

    data_dir_path = Path(data_dir)
    filepaths = list(data_dir_path.rglob(f"*.{type}"))
    print(f"Found {len(filepaths)} files of type '{type}' in '{data_dir}'")
    print(f"Excluding files containing: {exclude}")

    filtered_filepaths = [
        str(filepath)
        for filepath in filepaths
        if not any(ex.lower() in str(filepath.stem).lower() for ex in exclude)
    ]
    print(f"Filtered down to {len(filtered_filepaths)} files after exclusion")

    return filtered_filepaths


def locate_pavone_data(data_dir: str) -> List[str]:
    return find_all_files(
        data_dir, type="txt", exclude=["position", "test", "Screening"]
    )
