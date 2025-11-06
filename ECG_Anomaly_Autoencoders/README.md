## ECG anomaly data
Compile and compare 2 Keras autoencoders for labeling anomolous ECG data. \
ECG dataset of 4998 subject over 140 timepoints. Target labels in last column indicate whether an anomaly is present. The Keras models used are: 
1. Functional API and 
2. Model Subclassing (using Sequential API layers).
   
**Script/main.ipynb** presents data preprocessing and autoencoder application and comparison.\
**Script/ECG_plotting.m** is a MatLab script that creates a graph overlaying all raw ECG data.


### Credits
Based on original Model Subclassing autoencoder project by [Devavrata Tripathy](https://www.kaggle.com/datasets/devavratatripathy/ecg-dataset).
