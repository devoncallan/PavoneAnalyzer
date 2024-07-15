import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# Function to parse the file
def parse_file(file_path:str="", uploaded_file=None):
    metadata = {}
    data_started = False
    data_lines = []

    if file_path != "":
        with open(file_path, 'r') as file:
            for line in file:
                # Check if the data section has started
                if line.startswith('Time (s)'):
                    data_started = True
                    data_header = line.strip().split('\t')
                    continue

                # If in the data section, append line to data_lines
                if data_started:
                    data_lines.append(line.strip().split('\t'))
                else:
                    # Process metadata
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
    elif uploaded_file is not None:
        for line in uploaded_file:
            line = line.decode('utf-8')  # Decode the line if necessary
            # Check if the data section has started
            if line.startswith('Time (s)'):
                data_started = True
                data_header = line.strip().split('\t')
                continue

            # If in the data section, append line to data_lines
            if data_started:
                data_lines.append(line.strip().split('\t'))
            else:
                # Process metadata
                parts = line.strip().split('\t')
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

def extract_contact_points_from_data(data: pd.DataFrame) -> (int, int):
    
    y = data['Indentation (nm)']
    peaks, _ = find_peaks(y, prominence=1)  # Prominence > 0 to ensure we are finding a significant peak

    # Assuming the data is zero before and after the peak, we find where the signal "starts" and "ends".

    start = np.where(y[:peaks[0]] == 0)[0][-1] + 1  # Last zero before the peak

    end = peaks[0] + np.where(y[peaks[0]:] == 0)[0][0]  # First zero after the peak
    return start, end