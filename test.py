# Testing if the data getters of Data are working as expected!

from Data import Data
import pandas as pd
import numpy as np

# Set some seeds.
seed = 0
np.random.seed(seed)

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split 

def train_test_valid_split(X, y):
        """Split data into training/testing/validation, where validation is optional at instantiation."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
        X_test, X_valid, y_test, y_valid = train_test_split( \
                                    X_test, y_test, test_size=1/3, random_state=42)
        return (X_train, y_train, X_test, y_test, X_valid, y_valid)

# Read the data. 
adult_data = pd.read_csv("adult_data_no_NA.csv", index_col = 0)
print(adult_data.shape) # Looks good!

categorical_features = ["workclass","marital_status","occupation","relationship", \
                        "race","sex","native_country"]
numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

X = adult_data.loc[:, adult_data.columns != "y"]
y = adult_data.loc[:,"y"] 

# SPlit the data.
X_train, y_train, X_test, y_test, X_valid, y_valid = train_test_valid_split(X,y)

# We want to compare this to the other file, which I will save and load below. 
X_train_check = pd.read_csv("train_data_checking.csv", index_col=0) 
print(X_train_check.shape)
print(X_train.equals(X_train_check))

X_test_check = pd.read_csv("test_data_checking.csv", index_col=0) 
print(X_test_check.shape)
print(X_test.equals(X_test_check))

X_valid_check = pd.read_csv("valid_data_checking.csv", index_col=0) 
print(X_valid_check.shape)
print(X_valid.equals(X_valid_check))

# Thus the getters seem to work perfectly!
