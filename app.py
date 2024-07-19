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

pavone_files = class_df['filepath'].tolist()
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
