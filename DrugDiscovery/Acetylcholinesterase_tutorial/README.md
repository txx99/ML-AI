Build an ML model for drug discovery using the ChEMBL database (https://www.ebi.ac.uk/chembl/) of bioactive molecules with drug-inducing properties.
## Acetylcholinesterase [CHEMBL220]
### **Query:** 
'acetylcholinesterase' *Human Acetylcholinesterase*
### **Preprocess:** 
*script:* DD_ML_Part_1_Bioactivity_Preprocessing.ipynb \
*outputs:* bioactivity_data.csv (raw); bioactivity_preprocessed_data.csv (cleaned) 
### **Exploration:** 
*input:* bioactivity_preprocessed_data.csv \
*script:* DD_ML_Part_2_Exploratory_Data_Analysis.ipynb \
*output:* Part_2_Results
### **Fingerprinting + Feature Selection:** 
*input:* acetylcholinesterase_bioactivity_3class_data.csv \
*script:* DD_ML_Part_3_Descriptor_Dataset_Preparation.ipynb \
*output:* ./PaDEL acetylcholinesterase_bioactivity_3class_pubchem_fp.csv 
### **ML (RandomForestRegressor):** 
*input:* acetylcholinesterase_bioactivity_3class_pubchem_fp.csv \
*script:* DD_ML_Part_4_ML_Models.ipynb \
*output:* regression_model_scatter_plot.pdf


Based on tutorial and template Jupyter notebook by Chanin Nantasenamat, 
at the 'Data Professor' YouTube channel http://youtube.com/dataprofessor.
