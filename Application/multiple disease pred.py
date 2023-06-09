# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: siddhardhan
""" 

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

# diabetes_model = pickle.load(open('C:/Users/lanka/Downloads/Disease_prediction/saved models/diabetes_model.sav', 'rb'))

# heart_disease_model = pickle.load(open('C:/Users/lanka/Downloads/Disease_prediction/saved models/heart_disease_model.sav','rb'))

# parkinsons_model = pickle.load(open('C:/Users/lanka/Downloads/Disease_prediction/saved models/parkinsons_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Intel Medical Image Processing',
                          
                          ['Brain Tumour Segmentation',
                           'Pneumonia Detection',
                           'Kidney Tumour & Stone detection'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Brain Tumour Segmentation'):
    
    # page title
    st.title('Brain Tumour Segmentation using Intel SYCL / DPC++')
    
    

    # File Upload
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    
    # Display uploaded files
    if uploaded_files:
        st.subheader("Uploaded Files:")
        for file in uploaded_files:
            file_details = {"FileName": file.name, "FileType": file.type, "FileSize": file.size}
            st.write(file_details)



    if st.button('Detect Brain Tumour'):
        diab_diagnosis = "Scrutinize the DICOM file viewer" 
        st.write(diab_diagnosis)

        st.title("Brain tumour viewer")
    
        # Display OHIF Viewer using an iframe
        st.write('<iframe src="https://cloud.app.box.com/dicom_viewer/12345?toolbar=true" width="100%" height="800"></iframe>', unsafe_allow_html=True)







# Heart Disease Prediction Page
if (selected == 'Pneumonia Detection'):
    
    # page title
    st.title('Pneumonia Detection using Intel SYCL / DPC++')
    
    # File Upload
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    
    # Display uploaded files
    if uploaded_files:
        st.subheader("Uploaded Files:")
        for file in uploaded_files:
            file_details = {"FileName": file.name, "FileType": file.type, "FileSize": file.size}
            st.write(file_details)



    if st.button('Detect Pneumonia'):
        diab_diagnosis = "Scrutinize the DICOM file viewer" 
        st.write(diab_diagnosis)

        st.title("Pneumonia viewer")
    
        # Display OHIF Viewer using an iframe
        st.write('<iframe src="https://cloud.app.box.com/dicom_viewer/12345?toolbar=true" width="100%" height="800"></iframe>', unsafe_allow_html=True)

    

# Parkinson's Prediction Page
if (selected == "Kidney Tumour & Stone detection"):
    
    # page title
    st.title("Kidney Tumour & Stone detection using Intel SYCL / DPC++")
      # File Upload
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    
    # Display uploaded files
    if uploaded_files:
        st.subheader("Uploaded Files:")
        for file in uploaded_files:
            file_details = {"FileName": file.name, "FileType": file.type, "FileSize": file.size}
            st.write(file_details)



    if st.button('Detect Kidney Tumour'):
        diab_diagnosis = "Scrutinize the DICOM file viewer" 
        st.write(diab_diagnosis)
        st.title("Kidney tumour viewer")
    
        # Display OHIF Viewer using an iframe
        st.write('<iframe src="https://cloud.app.box.com/dicom_viewer/12345?toolbar=true" width="100%" height="800"></iframe>', unsafe_allow_html=True)


