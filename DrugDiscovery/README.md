# Drug Discovery ML Project
Build an ML model for drug discovery using the [ChEMBL database](https://www.ebi.ac.uk/chembl/) of bioactive molecules with drug-inducing properties.

### **Query:** 
protein/organism of interest.
### **Preprocess:**
remove duplicate smiles, drop NAs, select features (chembl_id, canonical_smiles, IC50), categorise bioactivity by IC50 values (<1k active; >10k inactive; 1k-10k intermediate).
### **Exploration:** 
calculate Lipinski Rule descriptors for oral drugs, convert IC50 to pIC50, visualise descriptors vs bioactivity (with MannWhitney-U significance testing):
- Molecular weight
- Octanol-water partition coefficient (LogP)
- Hydrogen bond donors
- Hydrogen bond acceptors 
  
### **Fingerprinting + Feature Selection:** 
usie PaDEL-Descriptor + fingerprinting database of choice (Pubchem) to build df of chemical fingerprints, concatenate with target feature of choice. 
### **ML:** 
remove low variance features, split X and Y data, lazypredict ML performances, train + test best model.

## Credits
Based on tutorial and template Jupyter notebook by Chanin Nantasenamat, 
at the ['Data Professor'](http://youtube.com/dataprofessor) YouTube channel.
