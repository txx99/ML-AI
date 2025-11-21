# %% part 1 

# ! pip install chembl_webresource_client
# Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client 

# Target search 
target = new_client.target
target_query = target.search('acetylcholinesterase') 
targets = pd.DataFrame.from_dict(target_query) # query results --> dictionary format --> into df  
targets # display 

selected_target = targets.target_chembl_id[1]
# new query based on activity (as opposed to 'target')
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)
df.standard_type.unique() # ours only has IC50 due to filtering, but could also be EC50 or % activity
df.to_csv('./data/bioactivity_data.csv', index=False) # dont save index numbers into csv

df=pd.read_csv('./data/bioactivity_data.csv')
df.head()
# remove rows where SMILES == NA
df=df[df['canonical_smiles'].notna()]
# duplicate molecule but slighlty different SMILES --> use Key 
from rdkit import Chem

def id_duplicates(df):
    def calculate_inchi_key(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles) 
            if mol: # if molecule
                return Chem.MolToInchiKey(mol) # create its inchi key
        except:
            return None

    df_analysis = df.copy()
    # calculate inchi keys 
    df_analysis['inchi_key'] = df_analysis['canonical_smiles'].apply(calculate_inchi_key)
    # group duplicates
    df_analysis['duplicate_group'] = df_analysis.groupby('inchi_key').ngroup()
    # count occurrences/group 
    duplicate_count = df_analysis['inchi_key'].value_counts()
    df_analysis['occurrence_count'] = df_analysis['inchi_key'].map(duplicate_count)

    return df_analysis



def process_duplicates(df, inchi_key_col='inchi_key', pchem_col='pchembl_value'):
    df1 = df.copy()

    # create new stats df:
    # per inchi_key group, calculate pchem stats
    group_stats = df1.groupby(inchi_key_col).agg({
        pchem_col: ['count', 'mean', 'std']
    }).round(4) # round to 4 decimals 
    # rename cols
    group_stats.columns = ['count', 'mean', 'std']
    group_stats = group_stats.reset_index()

    processed_rows = []
    duped_compounds = []
    # process each group
    for inchi_key in group_stats['inchi_key']:
        # df of all group members
        group_data = df1[df1[inchi_key_col]==inchi_key].copy()
        # collect stats row of group 
        group_stats_row = group_stats[group_stats['inchi_key']==inchi_key].iloc[0]

        if len(group_data) > 1: #multiple entries
            # add stats to df of all group members 
            group_data['group_mean'] = group_stats_row['mean']
            group_data['group_std'] = group_stats_row['std']
            duped_compounds.append(group_data)

            if group_stats_row['std'] == 0: # low variance bw entries
                # keep only first entry
                processed_row = group_data.iloc[[0]].copy() #first entry
                processed_row['group_type'] = 'zero_sd'
            else: # variation in group , sd != 0 
                # apply mean pchem of the group to the first entry 
                processed_row = group_data.iloc[[0]].copy() 
                processed_row[pchem_col] = group_stats_row['mean']
                processed_row['group_type'] = 'nonzero_sd'

            processed_rows.append(processed_row) #list of lists

        else: # unique group
            group_data['group_type'] = 'single_entry'
            processed_rows.append(group_data)
            
    # combine all processed rows
    final_df = pd.concat(processed_rows)
    duped_compounds_df = pd.concat(duped_compounds) if duped_compounds else pd.DataFrame()

    # sort final_df by ichi keys
    final_df = final_df.sort_values(by=[inchi_key_col])

    return final_df, duped_compounds_df

df_analysed = id_duplicates(df)
final_df, duped_compounds_df = process_duplicates(df_analysed)
final_df['group_type'].unique()
# feature selection for columns we care about
selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df2 = final_df[selection]
# remove rows with any NAs
colNames = list(df2.columns.values)
for col in colNames:
    df2=df2[df2[col].notna()]
df2 = df2.reset_index(drop=True)
# list data content for bioactivity_class
bioactivity_class = []
for i in df2.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("intermediate")
# concatenate new column into df
df2 = pd.concat([df2,pd.Series(bioactivity_class)], axis=1) # since bioactivity_class is list[], must make Series or df before concatenating 
list(df2)
df2 = df2.rename(columns={0:"bioactivity_class"})
list(df2)

df2.to_csv('./data/bioactivity_preprocessed_data.csv', index=False)



#%% Part 2

# ! pip install rdkit
# import sys
# sys.path.append('/usr/local/lib/python3.7/site-packages/')

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import seaborn as sns
sns.set_theme(style='ticks')
import matplotlib.pyplot as plt


df_preprocessed = pd.read_csv('./data/bioactivity_preprocessed_data.csv')
clean_smiles = []
for i in df_preprocessed.canonical_smiles.tolist():
    compounds = str(i).split('.')
    longest_compound= max(compounds, key=len) # max length element after splitting compounds 
    clean_smiles.append(longest_compound)
clean_smiles = pd.Series(clean_smiles, name = 'canonical_smiles') # series = 1D array w label
df_clean_smiles = pd.concat([df_preprocessed.drop(columns='canonical_smiles'), clean_smiles], axis=1)

# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) # returns RDKit molecule object based on SMILES string
        moldata.append(mol)
       
    baseData= np.arange(1,1) # create empty numpy array (based on range (1,1) = [])
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        # store molecule's descriptors together as 1D array
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row]) # vertical stacking of new array element (row) to priors (baseData) --> 2D array object 
        i=i+1      
    
    #save lipinski rule array into a df 
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
df_combined = pd.concat([df_clean_smiles,df_lipinski], axis=1)
df_combined.standard_value.describe()

def norm_value(input_df):
  input_df = input_df[input_df['standard_value'] != 0]
  norm = []

  for ic50 in input_df['standard_value']:
    if ic50 > 100000000:
      ic50 = 100000000
    norm.append(ic50)

  input_df['standard_value_norm'] = norm
  input_df['standard_value']
  x = input_df.drop(columns='standard_value', axis=1)

  return x.reset_index(drop=True)

df_norm = norm_value(df_combined)

# https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb
def pIC50(input_df):
    pIC50 = []

    for i in input_df['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input_df['pIC50'] = pIC50
    x = input_df.drop(columns = 'standard_value_norm', axis = 1) # drop normalised standard_value column    
    x = x[x['pIC50'] != 0] #ensure pIC50 never 0 bc issues ds... ?    
    return x

df_final = pIC50(df_norm)
df_final.to_csv('./data/acetylcholinesterase_bioactivity_3class_data.csv', index=False)

df_2class = df_final[df_final['bioactivity_class'] != 'intermediate']
df_2class.to_csv('./data/acetylcholinesterase_bioactivity_2class_data.csv')

# freq of 2 bioactivity classes
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='black')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_bioactivity_class.pdf')

#MW vs logP 
plt.figure(figsize=(5.5, 5.5))
sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# plt.show()
plt.savefig('./data/Part_2_Results/plot_MW_vs_LogP.pdf')

# pIC50
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_ic50.pdf')


# statistical analysis 
def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# active and inactive
  selection = [descriptor, 'bioactivity_class'] # descriptor vs bioactivity_class
  df = df_2class[selection]

  active = df[df.bioactivity_class == 'active']
  active = active[descriptor]

  inactive = df[df.bioactivity_class == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'
  
  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv('./data/Part_2_Results/'+filename)

  return results

mannwhitney('pIC50')

# bioactivity vs MW
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_MW.pdf')

mannwhitney('MW')

# bioactivity vs logP
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_LogP.pdf')

mannwhitney('LogP')

# numH donors
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_NumHDonors.pdf')

mannwhitney('NumHDonors')

# numH Acceptors
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')
plt.savefig('./data/Part_2_Results/plot_NumHAcceptors.pdf')

mannwhitney('NumHAcceptors')



#%% part 3

# !!! DOWNLOAD PaDEL !!!

# wget https://github.com/dataprofessor/padel/raw/main/fingerprints_xml.zip --directory-prefix ./PaDEL/
# ! pip install padelpy
# ! powershell Expand-Archive -Path ./PaDEL/fingerprints_xml.zip -DestinationPath ./PaDEL/fingerprints_xml
from padelpy import padeldescriptor

df3 = pd.read_csv('./data/acetylcholinesterase_bioactivity_3class_data.csv')

selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('./PaDEL/molecule.smi', sep='\t', index=False, header=False)

# validation: 
# cat ./PaDEL/molecule.smi | head -5
# cat ./PaDEL/molecule.smi | wc -l

# may hang after completing execution
fp_descriptortype = './PaDEL/fingerprints_xml/PubchemFingerprinter.xml'

padeldescriptor(mol_dir='./PaDEL/molecule.smi', d_file='./PaDEL/fp_descriptor_output.csv', descriptortypes=fp_descriptortype, 
        standardizenitro=True, threads=2, removesalt=True, fingerprints=True, detectaromaticity=True, standardizetautomers=True)

df3_X = pd.read_csv('./PaDEL/fp_descriptor_output.csv')
df3_X = df3_X.drop(columns=['Name']) # fp table no names
df3_Y = df3['pIC50']

dataset3 = pd.concat([df3_X,df3_Y], axis=1) 

dataset3.to_csv('./data/acetylcholinesterase_bioactivity_3class_pubchem_fp.csv', index=False)



# %% part 4

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('./data/acetylcholinesterase_bioactivity_3class_pubchem_fp.csv')
# for X (features), only want fingerprints
X = df.drop('pIC50', axis=1)
Y = df.pIC50

selection = VarianceThreshold(threshold=.1) 
X = selection.fit_transform(X)
X.shape

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# verify dimensions 
X_train.shape, Y_train.shape
X_test.shape, Y_test.shape

# building 42 models with default parameters
lazy_regs = LazyRegressor(ignore_warnings=True, verbose=0)
lz_model, lz_predict = lazy_regs.fit(X_train, X_test, Y_train, Y_test)
lz_model
np.random.seed(22)

# instantiate, fit, predict
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2_RForest = model.score(X_test, Y_test)
r2_RForest

Y_pred = model.predict(X_test)

sns.set_theme(color_codes=True)
sns.set_style("white")

ax = sns.regplot(x=Y_test, y=Y_pred)
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 13)
ax.set_ylim(0, 13)
plt.savefig('regression_model_scatter_plot.pdf')
plt.show()