
import streamlit as st
from Pavone_Indent_Analysis import parse_file, extract_contact_points_from_data
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os
st.set_page_config(layout="wide")


# Initialize session state for current index if it doesn't already exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

def get_classification_file(file_paths, output_filepath='classification.csv'):

    if os.path.exists(output_filepath):
        return pd.read_csv(output_filepath)

    file_data = [parse_file_path(file_path) for file_path in file_paths]

    # Convert to DataFrame
    df = pd.DataFrame(file_data)
    df.to_csv(output_filepath, index=False)
    return df

    # print(f"Classification file '{output_filepath}' has been created/updated.")

def find_text_files(directory):
    print('hello')
    # List to hold file paths
    text_files = []

    # Walk through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a text file
            if file.endswith('.txt') and 'position' not in file.lower() and 'test' not in file.lower():
                # Create a relative path from the base directory
                relative_path = os.path.relpath(os.path.join(root, file), start=directory)
                # Add to list
                text_files.append(relative_path)

    return text_files

def parse_file_path(file_path):
    # Split the path into its main components based on slashes
    print(file_path)
    parts = file_path.split('/')
    
    # Extract individual components
    experiment_info = parts[0]
    plate_info = parts[1]
    scan_info = parts[2]
    filename = parts[3]
    
    # Further split to extract detailed parts
    experiment_details = experiment_info.split('_')
    date = experiment_details[0]
    experiment_code = '_'.join(experiment_details[1:])  # Combine back the remaining parts

    plate_details = plate_info.split('_')
    plate_number = plate_details[0]
    well_number = plate_details[1]

    # Filename can be further split if more details needed
    filename_details = filename.split('_')
    coordinates_info = filename.split(' ')[-4:]  # Splits and takes the last four elements
    
    # Extracting coordinates and identifier
    S = coordinates_info[0]
    X = coordinates_info[1]
    Y = coordinates_info[2]
    I = coordinates_info[3].split('.')[0]  # Removing file extension

    return {
        'filepath': file_path,
        'date': date,
        'experiment_code': experiment_code,
        'plate_number': plate_number,
        'well_number': well_number,
        'scan_info': scan_info,
        'S': S,
        'X': X,
        'Y': Y,
        'I': I,
        'filename': filename,
        'classification': -1 # Initial classification (unclassified)
    }

    parts = file_path.split('/')
# Directory containing the text files
base_directory = '/Users/devoncallan/Downloads/2024_07_08_ChemspeedSamples'
exp_name = os.path.basename(base_directory)
class_file_path = f'{exp_name}.csv'

# Get a list of all text files and their relative paths
files = find_text_files(base_directory)
df = get_classification_file(files, class_file_path)
# st.write(df)

# results = pd.DataFrame(columns=['File Path', 'Classification'])

def load_data(file_path):
    # Load and return data from the file
    file_path = os.path.join(base_directory, file_path)
    metadata, data_df = parse_file(file_path)
    # data = pd.read_csv(file_path)
    return metadata, data_df

def plot_data(c, data_df):
    fig, ax = plt.subplots()
    plt.plot(data_df['Time (s)'], data_df['Load (uN)'], '-')
    # start_idx, end_idx = extract_contact_points_from_data(data_df)
    # plt.axvline(x=0.240, color='k')
    # plt.axvline(x=57.471, color='k')
    # plt.axvline(x=114.703, color='k')
    # plt.axvline(x=data_df['Time (s)'][start_idx], color='r')
    # plt.axvline(x=data_df['Time (s)'][end_idx], color='r')
    c.pyplot(fig)

def save_results(df, output_file_path):
    df.to_csv(output_file_path, index=False)

def update_classification(df, index, classification = -1):

    if index in df.index:
        df.at[index, 'classification'] = classification
    else:
        print(f'No row found at index {index}')
    return df

def increment_index(df):
    if st.session_state.current_index < len(df) - 1:
        st.session_state.current_index += 1
    else:
        st.session_state.current_index = 0
    st.experimental_rerun()

st.title('Data Classifier')

col1, col2 = st.columns([1, 1])
col1_left, col1_right = col1.columns([1, 1])
col2_left, col2_right = col2.columns([1, 1])

if col1_left.button('Previous', use_container_width=True) and st.session_state.current_index > 0:
    st.session_state.current_index -= 1
    st.experimental_rerun()

if col1_right.button('Next', use_container_width=True) and st.session_state.current_index < len(files) - 1:
    st.session_state.current_index += 1
    st.experimental_rerun()

if col2_left.button('Bad', type='secondary', use_container_width=True):
    update_classification(df, st.session_state.current_index, 0)
    save_results(df, class_file_path)
    increment_index(df)

if col2_right.button(label='Good', type='primary', use_container_width=True):
    update_classification(df, st.session_state.current_index, 1)
    save_results(df, class_file_path)
    increment_index(df)


file_path = files[st.session_state.current_index]
parts = parse_file_path(file_path)

# Display current file index and a selector for jumping to a file
jump_to = col1.number_input('Jump to index', min_value=0, max_value=len(df)-1, value=st.session_state.current_index)
if jump_to != st.session_state.current_index:
    st.session_state.current_index = jump_to
    st.experimental_rerun()

col1.markdown(f'#### Experiment {st.session_state.current_index} Info:')
col1.write(parts)

metadata, data_df = load_data(file_path)

class_idx = df.loc[st.session_state.current_index, 'classification']
col2.markdown(f'### Classification: {class_idx}')

plot_data(col2, data_df)

st.divider()
st.markdown('#### Download data:')
st.write(df)


