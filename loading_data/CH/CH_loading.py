# load the original data from source and do initial pre-processing according to discussion in thesis. 
# then save the pre-processed data to disk, without NA. 

import pandas as pd
import numpy as np

# Script for initial load and pre-processing of the Adult data. 

# Load the data and use the first column (Row-Number) as the index of the dataframe. 
df = pd.read_csv("original_data/Churn_Modelling.csv", index_col = 0)

print(f"Unique CustomerIds: {len(np.unique(df.iloc[:,0]))}")

# Remove the first column because the unique CustomerIds are not interesting to us. 
df = df.drop("CustomerId", axis = 1)

# We use the default column names, except that we change the response "Exited" to "y".
df.columns = df.columns.tolist()[:-1] + ["y"]

categorical_features = ["Surname","Geography", "Gender", "HasCrCard", "IsActiveMember"] 
numerical_features = ["CreditScore","Age","Tenure","Balance","NumOfProducts", "EstimatedSalary"]

# Check if there are any NA values. 
print(df.shape)
print(f"Any missing values: {df.isnull().values.any()}") # There are no NA-values. 

# Print the default data types of the dataset. 
print(df.dtypes)

# Set the categorical features to categories and leave the numerical columns as they are (either floats or ints).
df[categorical_features + ["y"]] = df[categorical_features + ["y"]].astype("category")
print(df.dtypes)

# Change y such that "Retained/0"=1 and "Exited/1"=0
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: 1 if x == 0 else 0)

# Get some more info about the levels in the categorical features below. 
summer = 0
for feat in categorical_features:
    unq = len(df[feat].value_counts().keys().unique())
    print(f"Feature '{feat}' has {unq} unique levels")
    summer += unq
print(f"The sum of all levels is {summer}. This will be the number of cat-columns after one-hot encoding (non-full rank)")

# We save the complete data set as a csv for use in other scripts.
df.to_csv("loading_data/CH/CH_no_NA.csv") # Save this to csv.
