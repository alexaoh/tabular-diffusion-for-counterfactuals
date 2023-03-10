# load the original data from source and do initial pre-processing according to discussion in thesis. 
# then save the pre-processed data to disk, without NA. 

from scipy.io import arff
import pandas as pd
import numpy as np

# Script for initial load and pre-processing of the Adult data. 

data = arff.loadarff("original_data/dataset_37_diabetes.arff")
df = pd.DataFrame(data[0])
print(df.head())
df.columns = ["num_pregnant", "plasma", "dbp", "skin", "insulin", "bmi", "pedi", "age", "y"]

categorical_features = [] 
numerical_features = df.columns.tolist()

# Check if there are any NA values. 
print(df.shape)
print(f"Any missing values: {df.isnull().values.any()}") # There are no NA-values. 

# Print the default data types of the dataset. 
print(df.dtypes)

# Set output to category.
df[categorical_features + ["y"]] = df[categorical_features + ["y"]].astype("category")

# Set age, insulin, skin, dbp, plasma and num_pregnant to int64.
df.loc[:,["age", "insulin", "skin", "dbp", "plasma", "num_pregnant"]] = \
    df.loc[:,["age", "insulin", "skin", "dbp", "plasma", "num_pregnant"]].astype("int64")

print(df.dtypes)

# Change y such that "tested_negative"=1 and "tested_positive"=0.
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: 1 if x.decode("UTF-8") == "tested_negative" else 0)

# We save the complete data set as a csv for use in other scripts.
df.to_csv("loading_data/DI/DI_no_NA.csv") # Save this to csv.
