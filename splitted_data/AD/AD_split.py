# Here we split the adult data as discussed in the thesis and save the splitted data sets to disk. 
import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pandas as pd
from Data import Data

adult_data = pd.read_csv("loading_data/AD/AD_no_NA.csv", index_col = 0)
print(f"Total: {adult_data.shape}")
categorical_features = ["workclass","marital_status","occupation","relationship", \
                        "race","sex","native_country"]
numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

# We don't care about the preprocessing attributes, as we are only interested in the train/validation/split 
# for the data BEFORE it is pre-processed. 

Adult = Data(adult_data, categorical_features, numerical_features, already_splitted_data=False)
X_train, y_train = Adult.get_training_data()
X_test, y_test = Adult.get_test_data()
X_valid, y_valid = Adult.get_validation_data()

# Print the shapes of each data set. 
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Valid: {X_valid.shape}")

# Save the three data sets to disk for later use. 
training = X_train.copy()
training["y"] = y_train
training.to_csv("splitted_data/AD/AD_train.csv") # Save this to csv.

test = X_test.copy()
test["y"] = y_test
test.to_csv("splitted_data/AD/AD_test.csv") # Save this to csv.

valid = X_valid.copy()
valid["y"] = y_valid
valid.to_csv("splitted_data/AD/AD_valid.csv") # Save this to csv.
