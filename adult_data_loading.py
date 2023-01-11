import pandas as pd

# Script for initial load and pre-processing of the Adult data. 

df1 = pd.read_csv("original_data/adult.data", header = None, na_values = " ?")
df2 = pd.read_csv("original_data/adult.test", header = None, na_values = " ?")
df1.columns = df2.columns = ["age","workclass","fnlwgt","education","education_num",
                      "marital_status","occupation","relationship","race","sex",
                      "capital_gain","capital_loss","hours_per_week","native_country", "y"]
adult_data = pd.concat([df1,df2])

categorical_features = ["workclass","marital_status","occupation","relationship", \
                        "race","sex","native_country"]
numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

# Remove "education" column.
adult_data = adult_data.drop(columns = ["education"])

# Check if there are any NA values. 
print(adult_data.shape)
print(adult_data.isnull().values.any())
adult_data = adult_data.dropna() # Drop the NA values since we know they are few for this data set. 
print(adult_data.shape)

# Select covariates and response. 
X = adult_data.loc[:, adult_data.columns != "y"]
y1 = adult_data.loc[:,"y"] # Temporary data frame. 
y = y1.copy()

# Change y such that " <=50K"=0 and " >50K"=1
y.loc[y == " <=50K"] = 0
y.loc[y == " <=50K."] = 0
y.loc[y == " >50K"] = 1
y.loc[y == " >50K."] = 1


# Get some more info about the levels in the categorical features below. 
summer = 0
for feat in categorical_features:
    unq = len(X[feat].value_counts().keys().unique())
    print(f"Feature '{feat}'' has {unq} unique levels")
    summer += unq
print(f"The sum of all levels is {summer}. This will be the number of cat-columns after one-hot encoding (non-full rank)")

# We save the complete data set as a csv for use in other scripts.
adult_for_saving = X.copy()
adult_for_saving["y"] = y
adult_for_saving.to_csv("adult_data_no_NA.csv") # Save this to csv.

