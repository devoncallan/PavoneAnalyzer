import streamlit as st
from Pavone_Indent_Analysis import parse_file, extract_contact_points_from_data
import matplotlib.pyplot as plt



print('hi')

st.title('Pavone Analyzer')

file = st.file_uploader('Upload files')

if not file:
    st.stop()
    
metadata, data_df = parse_file(uploaded_file=file)

# Data cleaning
data_df['Z-stage (nm)'] = data_df['Piezo (nm)'] - data_df['Cantilever (nm)']

# Potentially reduce number of data points (quite excessive)
# Add columns or change for alternative units (um)
data_df

start_idx, end_idx = extract_contact_points_from_data(data_df)

fig, ax = plt.subplots()
plt.plot(data_df['Time (s)'], data_df['Load (uN)'], 'o')
# plt.axvline(x=0.240, color='k')
# plt.axvline(x=57.471, color='k')
# plt.axvline(x=114.703, color='k')
plt.axvline(x=data_df['Time (s)'][start_idx], color='r')
plt.axvline(x=data_df['Time (s)'][end_idx], color='r')
st.pyplot(fig)

