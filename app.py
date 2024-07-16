import streamlit as st
import zipfile
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from Pavone_Indent_Analysis import parse_file, extract_contact_points_from_data

from parse_pavone import parse_pavone_filepath, locate_pavone_data, create_new_classification_file, get_classification_file, update_classification, plot_pavone_data, read_pavone_data

st.set_page_config(layout="wide")

# Initialize session state for current index if it doesn't already exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    
# def create_new_classification_file(file_paths, output_filepath='classification.csv'):
#     file_data = [parse_file_path(file_path) for file_path in file_paths]
#     df = pd.DataFrame(file_data)
#     df.to_csv(output_filepath, index=False)
#     return df

# def get_classification_file(file_paths, output_filepath='classification.csv'):
#     if os.path.exists(output_filepath):
#         return pd.read_csv(output_filepath)

#     return create_new_classification_file(file_paths, output_filepath)

# def find_text_files(directory):
#     text_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.txt') and 'position' not in file.lower() and 'test' not in file.lower():
#                 relative_path = os.path.relpath(os.path.join(root, file), start=directory)
#                 text_files.append(os.path.join(directory, relative_path))
#     return text_files

# def find_all_files(directory, type='txt', exclude=[]):
#     all_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(f'.{type}') and not any([x in file for x in exclude]):
#                 relative_path = os.path.relpath(os.path.join(root, file), start=directory)
#                 all_files.append(os.path.join(directory, relative_path))
#     return all_files

# def parse_file_path(file_path):
#     # print(file_path)
#     parts = file_path.split('/')
#     base_dir = parts[0]
#     experiment_info = parts[1]
#     plate_info = parts[2]
#     scan_info = parts[3]
#     filename = parts[4]

#     experiment_details = experiment_info.split('_')
#     date = experiment_details[0]
#     experiment_code = '_'.join(experiment_details[1:])

#     plate_details = plate_info.split('_')
#     plate_number = plate_details[0]
#     well_number = plate_details[1]

#     filename_details = filename.split('_')
#     coordinates_info = filename.split(' ')[-4:]

#     S = coordinates_info[0]
#     X = coordinates_info[1]
#     Y = coordinates_info[2]
#     I = coordinates_info[3].split('.')[0]

#     return {
#         'filepath': file_path,
#         'date': date,
#         'experiment_code': experiment_code,
#         'plate_number': plate_number,
#         'well_number': well_number,
#         'scan_info': scan_info,
#         'S': S,
#         'X': X,
#         'Y': Y,
#         'I': I,
#         'filename': filename,
#         'classification': -1
#     }

# def load_data(file_path):
#     metadata, data_df = parse_file(file_path)
#     return metadata, data_df

# def plot_data(data_df):
#     fig, ax = plt.subplots(dpi=150)
#     plt.plot(data_df['Time (s)'], data_df['Load (uN)'], '-')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Load (μN)')
#     plt.tight_layout()
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     return fig, ax, img

# def save_results(df, output_file_path):
#     df.to_csv(output_file_path, index=False)


def increment_index(max_index):
    if st.session_state.current_index < max_index - 1:
        st.session_state.current_index += 1
    else:
        st.session_state.current_index = 0




# Locate Pavone data from the specified directory
data_dir = '2024_07_08_ChemspeedSamples'
pavone_files = locate_pavone_data(data_dir)
num_files = len(pavone_files)

# Get the current classification file
class_file_path = f'{data_dir}.csv'
if os.path.exists(class_file_path):
    class_df = get_classification_file(pavone_files, class_file_path)
else:
    class_df = create_new_classification_file(pavone_files, class_file_path)
    
num_classified = num_files - int((class_df['classification'] == -1).sum())
    
c1, c2 = st.columns([1, 1])
c1.title('Pavone Data Classifier:')
c2.title(f'{num_classified} / {num_files} classified')

if num_classified >= num_files:
    st.balloons()

# Define streamlit panels
c1, c2 = st.columns([1, 1])
c1L, c1C, c1R= c1.columns([1, 1, 1])
c2L, c2R = c2.columns([1, 1])


c1L.markdown('##### Jump to index:')
new_index = c1C.number_input('Jump to index', min_value=0, max_value=num_files-1, value=st.session_state.current_index, label_visibility='collapsed')
if new_index != st.session_state.current_index:
    st.session_state.current_index = new_index
    st.rerun()
    
if c1R.button('Next unclassified ➡️', use_container_width=True):
    next_unclassified = class_df[class_df['classification'] == -1].index[0]
    st.session_state.current_index = next_unclassified
    st.rerun()
    
# c1L, c1R = c1.columns([1, 1])
    

    


# Define bad and good classification buttons
if c2L.button(':thumbsdown:', type='secondary', use_container_width=True):
    update_classification(class_df, st.session_state.current_index, 0)
    class_df.to_csv(class_file_path, index=False)
    increment_index(max_index=num_files)
    st.rerun()

if c2R.button(label=':thumbsup:', type='primary', use_container_width=True):
    update_classification(class_df, st.session_state.current_index, 1)
    class_df.to_csv(class_file_path, index=False)
    increment_index(max_index=num_files)
    st.rerun()


file_path = pavone_files[st.session_state.current_index]
file_info = dict(class_df.loc[st.session_state.current_index])
metadata, data_df = read_pavone_data(file_path)

c1.markdown('')
c1.markdown(f'#### Experiment {st.session_state.current_index} Info:')
c1.write(file_info)

# Define string and color based on classification
class_val = class_df.loc[st.session_state.current_index, 'classification']
class_str = 'Good' if class_val == 1 else 'Bad' if class_val == 0 else 'Unclassified'
class_color = 'green' if class_str == 'Good' else 'red' if class_str == 'Bad' else 'gray'

# Use the color in the markdown with inline CSS
c2L, c2R = c2.columns([6, 1])
c2L.markdown(f'#### Classification: <span style="color: {class_color};">{class_str}</span>', unsafe_allow_html=True)


# Plot Pavone data
fig, ax, img = plot_pavone_data(data_df, savefig=True)

image_filename = file_info['filename'].replace('.txt', '.png')

btn = c2R.download_button(
   label=":floppy_disk:",
   data=img,
   file_name=image_filename,
   mime="image/png",
   use_container_width=True
)
c2.pyplot(fig)


### Display
c = st.container()
c.divider()
c1, c2 = c.columns([1, 1])
c1.markdown('#### Download data:')
c.write(class_df)
if c.button('Reset classifications'):
    class_df = create_new_classification_file(pavone_files, class_file_path)
    class_df.to_csv(class_file_path, index=False)
    st.rerun()
# st.write(class_df)
    