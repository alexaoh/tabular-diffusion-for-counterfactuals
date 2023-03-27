# Author: Alexander J Ohrt.
# In this file we use the trees from MCCE to generate data for Experiment 1 in my master's thesis. 

# We follow the methodology shown in the README (quite "dirty" implementation, but that should be OK for our purposes).

import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pandas as pd
import numpy as np

from mcce.mcce import MCCE # For working with mcce.
from mcce.data import Data # For making data sets to work with mcce. 

import Data # Import my class for scaling/encoding, etc.

# load the data. 
training = pd.read_csv("splitted_data/AD/AD_train.csv", index_col = 0)
test = pd.read_csv("splitted_data/AD/AD_test.csv", index_col = 0)
valid = pd.read_csv("splitted_data/AD/AD_valid.csv", index_col = 0)
data = {"Train":training, "Test":test, "Valid":valid}

# Specify column-names in the data sets.
categorical_features = ["workclass","marital_status","occupation","relationship", \
                        "race","sex","native_country"]
numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
features = numerical_features + categorical_features

target = ["y"]
immutable_features = target # Set immutable features to "target", such that we can sample conditionally from the trees. 

data_object = Data.Data(data, categorical_features, numerical_features, already_splitted_data=True, scale_version="quantile", valid = True)
X_train, y_train = data_object.get_training_data_preprocessed()
X_test, y_test = data_object.get_test_data_preprocessed()
X_valid, y_valid = data_object.get_validation_data_preprocessed()

# Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
X_valid = pd.concat((X_test, X_valid))
y_valid = pd.concat((y_test, y_valid))

class random_model():
    """Make random model to suppress error in MCCE. We do not need a prediction model when generating data anyway."""
    def __init__(self):
        pass
    
    def predict(self):
        return None

# This is the dataframe we will use to fit the MCCE object. 
training_df = X_train.copy()
training_df["y"] = y_train

# Create data object according to mccepy README that can be passed to MCCE later. 
class Dataset():
    def __init__(self, 
                 continuous,
                 categorical,
                 categorical_encoded, 
                 immutables
                 ):

        self.continuous = continuous
        self.categorical = categorical
        self.categorical_encoded = categorical_encoded # Add this to work with MCCE class constructor.
        self.immutables = immutables         

continuous = numerical_features
categorical = categorical_features
categorical_encoded = data_object.encoder.get_feature_names(categorical_features).tolist()
immutables = ["y"] # Set the label as immutable. The rest are not. 

# Make the data object.
dataset = Dataset(continuous, categorical, categorical_encoded, immutables)

# Make the MCCE object.
model = random_model()
mcce = MCCE(dataset, model)

# Fit the trees in the MCCE object to the data (only conditional on the response ["y"]).
dtypes = dict([(x, "float") for x in continuous])
for x in categorical_encoded:
    dtypes[x] = "category"
training_df2 = training_df.copy()
dtypes["y"] = "category"
training_df2 = (training_df2).astype(dtypes)

mcce.fit(training_df2, dtypes)
generated_data = mcce.generate(training_df2, k = 1) # Sample one time for each data point in the training data. 

# Reverse transform the data. 
generated_data = data_object.descale(generated_data)
generated_data = data_object.decode(generated_data)

# Save to disk.
generated_data.to_csv("synthetic_data/AD_from_trees.csv")