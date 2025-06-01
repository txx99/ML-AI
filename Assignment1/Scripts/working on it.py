import pandas as pd
import numpy as np
import data_preprocessor as dp
import statistics as stats
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('messy_data.csv')
clean_data = data.copy()

# IMPUTING DATA =================================================
# 2. Preprocess the data
cleaned_data = dp.impute_missing_values(data=clean_data, strategy='mean')
# print(cleaned_data)
# print(cleaned_data['j'])
# print(cleaned_data['z'])
# cleaned_data = impute(clean_data, "mean")

def impute(data, strategy):
    data2=data.copy()
    numericCols=[]
    factoredCols=[]
    # separate targets from data
    colNames=list(data2.columns.values)
    for col in colNames:
        #delete cols missing 50% of data
        if pd.isnull(data2[col]).sum()>=(len(data2[col]))*.5:
            print(col, len(data2[col]))
            data2=data2.drop(columns=col)
        elif data2[col].dtype == 'object':
            factoredCols.append(col)
        else:
            numericCols.append(col)
#     for col in numericCols:
#         nans=np.where(pd.isnull(data2[col]))[0]
#         for j in nans:
#             if strategy == 'mean':
#                 data2.loc[j, col]=np.mean(data2[col])
#                 # data2.loc[j, col]= data2[col].mean()
#             #     for i in numericCols:
#             # if strategy == 'mean':
#             #     data2['i'].fillna(data2['i'].mean(), inplace=True)
#             elif strategy=='mode':
#                 data2.loc[j, col]=stats.mode(data2[col])
#             elif strategy =='median':
#                 data2.loc[j, col]=np.median(data2[col])
#     for col in factoredCols:
#         data2[col]=data2[col].fillna(stats.mode(data2[col]))
#         # data2.loc[j, col]=stats.mode(data2[col])

#     return data2

# # missing_data = data.isnull().sum()
# # print(missing_data)
# # print(data['j'])
# imp=impute(data, 'mean')
# # print(imp['j'])
# # missing_data = imp.isnull().sum()
# # print(missing_data)

# print(list(imp.columns.values))


# REMOVING DUPLICATES ================================================
# clean_data = dp.remove_duplicates(clean_data)
# dups=cleaned_data.duplicated()
dups=data.duplicated().sum()
print('dups =', dups)
dups=cleaned_data.copy().drop_duplicates()

print('dups = ', dups)


#NORMALISE ================================================
# scaling = adjust the 'scale' between two columns that moght have different units and drastically different scales
# dont want ML thrown off by this, so we 'scale' or 'standardise' these columns 
# MinMaxScaler = x-min / (max-min) <-- max + min of column
# StdScaler = x-mean/std <-- of column
# in ML, would apply scaler then apply transform on the data
# training data = .fit_transform() , testing//novel data = .transform()
# which to use? depends on data but StdScaler usually better/common
# print("GOING OGING", dups)
# data2 = pd.get_dummies(dups, drop_first=True)
# print("Gone", data2)

def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    data = pd.get_dummies(data, drop_first=True)
    print(data)
    # scalers
    # data=np.array(data)
    mmScaler=MinMaxScaler()
    stdScaler=StandardScaler()

    #list of cols to scale
    colNames=list(data.columns.values)
    numericCols=[]
    for col in colNames:
        if data[col].dtype != 'object':
            numericCols.append(col)

    #fit + transform by scaler
    if method == 'minmax':
        mmScaler.fit(data[numericCols])
        data[numericCols]=mmScaler.transform(data[numericCols])
    elif method =='standard':
        stdScaler.fit(data[numericCols])
        data[numericCols]=stdScaler.transform(data[numericCols])


    return data

normalData = normalize_data(dups)
print(normalData)



# ====== REMOVE REDUNDANT FEATURES ==========================================
# clean_data = dp.remove_redundant_features(clean_data)

def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    
    data2=data.copy()
    # make correlation matrix (liek heatmap)
    corrMatrix= data2.corr(numeric_only=True)
    print(corrMatrix)
    # highly correlated features = are giving us the same information --> can drop one
    #list and remove highly correlated fetaures
    removable_array = np.where(abs(corrMatrix)>threshold)
    # removable_array = 
    #^ returns numeric positions in corrMatrix, including 1.000 for all
    # removable_indices=np.array()
    # removable_indices=[]
    # for [i, j] in removable_array:
    #     print(i, j)
    #     if i!=j:
    #         removable_indices.append(i)
    # print(removable_indices)

    removable_indices=removable_array[0].tolist()    
    # removable_indices=list(set(removable_indices))
    removables=Counter(removable_indices)
    print(removables)
    removables = [k for k, v in removables.items() if v>1]
    colNames=list(data2.columns.values)
    print(colNames)
    for i in removables:
        # print(colNames[i])
        remove=colNames[i]
        data2=data2.drop(columns=remove)
    print(data2)


    return data2

# no_dups = remove_redundant_features(normalData)



# log regression model ========================
def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            onehot = pd.get_dummies(features[col], prefix=col).astype(float)
            # get_dummies() = one-hot encoding as boolean --> .astype() to convert if needed
            # but usually dont need to convert booleans, the program will know how ot interpret 
            features = pd.concat([features, onehot], axis=1)
            # concatenating the extra columns into the OG df by the horizontal (1) axis
            features.drop(col, axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)

    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None


# simple_model(data, scale_data=True)
# getting error of 'classifier' innfo in the data. we did one-hot encoding for the categroical data 
# but currently it's not being scaled...