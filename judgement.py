import streamlit as st
import zipfile
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from Pavone_Indent_Analysis import parse_file, extract_contact_points_from_data

from parse_pavone import parse_pavone_filepath

print('howdy')
st.set_page_config(layout="wide")

# base_directory = 'data'

# Initialize session state for current index if it doesn't already exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    
def create_new_classification_file(file_paths, output_filepath='classification.csv'):
    file_data = [parse_pavone_filepath(file_path) for file_path in file_paths]
    df = pd.DataFrame(file_data)
    df.to_csv(output_filepath, index=False)
    return df

def get_classification_file(file_paths, output_filepath='classification.csv'):
    if os.path.exists(output_filepath):
        return pd.read_csv(output_filepath)

    return create_new_classification_file(file_paths, output_filepath)

def find_text_files(directory):
    text_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt') and 'position' not in file.lower() and 'test' not in file.lower():
                relative_path = os.path.relpath(os.path.join(root, file), start=directory)
                text_files.append(relative_path)
    return text_files

def load_data(file_path):
    file_path = os.path.join(st.session_state.extract_dir, file_path)
    metadata, data_df = parse_file(file_path)
    return metadata, data_df

# def plot_data(data_df):
#     fig, ax = plt.subplots(dpi=150)
#     plt.plot(data_df['Time (s)'], data_df['Load (uN)'], '-')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Load (μN)')
#     plt.tight_layout()
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     return fig, ax, img

def save_results(df, output_file_path):
    df.to_csv(output_file_path, index=False)

def update_classification(df, index, classification=-1):
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
    print('Increment rerun!')
    # st.rerun()

@st.cache_data
def extract_zip_file(uploaded_file):
    print('Extracting ZIP file...')
    local_zip_path = "uploaded_data.zip"
    with open(local_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_dir = "extracted_data"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir

# @st.cache_data
def load_classification_data(extract_dir, exp_name):
    print('Loading classification data...')
    base_directory = extract_dir
    class_file_path = f'{exp_name}.csv'
    files = find_text_files(base_directory)
    print(files)

    return files, class_file_path

st.title('Data Classifier')

uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")

if 'extract_dir' not in st.session_state:
    if uploaded_file is not None:
        extract_dir = extract_zip_file(uploaded_file)
        st.session_state.extract_dir = extract_dir
    else:
        st.stop()

# if st.session_state.extract_dir is not None:
print('hello')
# extract_dir = extract_zip_file(uploaded_file)

exp_name = os.path.basename(st.session_state.extract_dir)
files, class_file_path = load_classification_data(st.session_state.extract_dir, exp_name)
df = get_classification_file(files, class_file_path)

col1, col2 = st.columns([1, 1])
col1_left, col1_right = col1.columns([1, 1])
col2_left, col2_right = col2.columns([1, 1])

if col1_left.button('Previous', use_container_width=True) and st.session_state.current_index > 0:
    st.session_state.current_index -= 1
    print('Previous rerun!')
    # st.rerun()

if col1_right.button('Next', use_container_width=True) and st.session_state.current_index < len(files) - 1:
    st.session_state.current_index += 1
    print('Next rerun!')
    # st.rerun()

if col2_left.button('Bad', type='secondary', use_container_width=True):
    update_classification(df, st.session_state.current_index, 0)
    save_results(df, class_file_path)
    increment_index(df)

if col2_right.button(label='Good', type='primary', use_container_width=True):
    update_classification(df, st.session_state.current_index, 1)
    save_results(df, class_file_path)
    increment_index(df)

file_path = files[st.session_state.current_index]
parts = dict(df.loc[st.session_state.current_index])
# parts = parse_file_path(file_path)

jump_to = col1.number_input('Jump to index', min_value=0, max_value=len(df)-1, value=st.session_state.current_index)
if jump_to != st.session_state.current_index:
    st.session_state.current_index = jump_to
    print('Jump rerun!')
    # st.rerun()

col1.markdown(f'#### Experiment {st.session_state.current_index} Info:')
col1.write(parts)

metadata, data_df = load_data(file_path)
class_idx = df.loc[st.session_state.current_index, 'classification']
class_str = 'Good' if class_idx == 1 else 'Bad' if class_idx == 0 else 'Unclassified'
col2_left.markdown(f'### Classification: {class_str}')

fig, ax, img = plot_data(data_df)

image_filename = parts['filename'].replace('.txt', '.png')

btn = col2_right.download_button(
   label="Download figure",
   data=img,
   file_name=image_filename,
   mime="image/png"
)
col2.pyplot(fig)

st.divider()
st.markdown('#### Download data:')
if st.button('Reset classifications:'):
    create_new_classification_file(files, class_file_path)
    st.rerun()

st.write(df)
    