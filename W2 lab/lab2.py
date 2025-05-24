import functionals as f
# ^ original py script by prof w/ functions for this lab
import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the dataset
path_to_file = 'messy_data.csv'
data = pd.read_csv(path_to_file)

# one hot encoding
data_onehot = pd.get_dummies(data, drop_first=True)

# Remove duplicates
data_no_duplicates = data.copy().drop_duplicates()

# Fix inconsistent entries
data_format_fixed = data.copy() 
# want a COPY of the data to manipulate, not a variable that links back to the OG data
data_format_fixed['sex'] = data_format_fixed['sex'].apply(lambda x: 'female' if 'F' in x or 'f' in x else 'male')

# Check for OUTLIERS - if they exist, remove them or consider imputing them
# outliers = np.abs(stats.zscore(data_no_duplicates.select_dtypes(include=[np.number]))) > 3
outliers = np.abs(stats.zscore(data_no_duplicates.select_dtypes(include=[np.number])))> 3
# looking w/i 3 standard deviations

#    we can see that tprc column has some outliers
#       --> must decide if we're dropping those samples or assigning mean to those samples
#    tprc outliers represent 2% of the data
outliers = pd.DataFrame(outliers)
# data_no_duplicates.loc[[outliers[4], ['tprc']]] = np.mean(data_no_duplicates["tprc"])
# df.loc[df[col]condition, outputInfo]
#         rows of Interest, col to view
# in datanodups$tprc, locate matches to outliers$tprc exist (ie are TRUE) = replace them with mean of datanodups$tprc

for i in range(0, len(outliers[4])):
    if outliers[4][i]==True:
        data_no_duplicates[i, 'trpc'] = np.mean(data_no_duplicates["tprc"])


# now have dataset with no duplicates, fixed format (didnt really need fixing), and outlier removed (if they exist)
cleaner_data = data_no_duplicates.copy()
# looking at missing values, see deck has a lot of missing values, so lets remove it completely
# also okay to remove since its kind of a duplication of the 'class' data
cleaner_data.drop(columns=['deck'],inplace=True)

# # impute missing values for numeric columns only
for col in cleaner_data.select_dtypes(include=[np.number]).columns:
    cleaner_data[col].fillna(cleaner_data[col].mean(), inplace=True)

# # f.simple_model(cleaner_data)
