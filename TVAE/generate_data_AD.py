# Author: Alexander J Ohrt.
# In this file we use TVAE via the SDV Python API to synthesize data. 

import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pickle
import torch
import random
import pandas as pd
import numpy as np
from ctgan import TVAE
import Data # Import my class for scaling/encoding, etc.

# Load data etc here. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using '{device}' device.")

# Set seeds for reproducibility. 
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Argument for when running the script. 
train = False

# Load the data. Load as csv here, since TVAE wants the categorical features are strings (objects) anyway. 
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

# We return the data before pre-processing, since the pre-processing in TVAE is done according to the method developed by Xu et. al.
data_object = Data.Data(data, categorical_features, numerical_features, already_splitted_data=True, scale_version="quantile", valid = True)
X_train, y_train = data_object.get_training_data()
X_test, y_test = data_object.get_test_data()
X_valid, y_valid = data_object.get_validation_data()

# Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
X_valid = pd.concat((X_test, X_valid))
y_valid = pd.concat((y_test, y_valid))

# This is the dataframe we will use to fit the MCCE object. 
training_df = X_train.copy()
training_df["y"] = y_train

if train: 
        # Build a TVAE-object and fit it to the training data. 
        tvae = TVAE()

        print("\n Began fitting.\n")
        tvae.fit(train_data = training_df, discrete_columns = categorical_features + target)
        print("\n Ended fitting. \n")

        # Save fitted model to disk.
        with open("pytorch_models/AD_TVAE.obj", "wb") as f:
                pickle.dump(tvae, f)

if not train:
        # Load fitted model from disk.
        with open("pytorch_models/AD_TVAE.obj", 'rb')  as f:
                tvae = pickle.load(f)
                tvae.decoder = tvae.decoder.to(device)

# Sample data.
print("\n Began sampling.\n")

d1 = pd.concat((X_train, X_valid))
d2 = pd.concat((y_train, y_valid))
d1["y"] = d2

generated_data = tvae.sample(samples = d1.shape[0]) # Generate "adult_data"-size of synthetic data. 
#generated_data = tvae.sample(samples = training_df.shape[0])
print("\n Ended sampling.\n")

# Save to disk.
generated_data.to_csv("synthetic_data/AD_TVAE.csv")
