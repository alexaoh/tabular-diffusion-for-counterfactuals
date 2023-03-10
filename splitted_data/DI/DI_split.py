# Here we split the data as discussed in the thesis and save the splitted data sets to disk. 
import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pandas as pd
from Data import Data

#data = pd.read_csv("loading_data/DI/DI_no_NA.csv", index_col = 0)
data = pd.read_pickle("loading_data/DI/DI_no_NA.pkl")

print(f"Total: {data.shape}")
categorical_features = []
numerical_features = data.columns.tolist()

# We don't care about the preprocessing attributes, as we are only interested in the train/validation/split 
# for the data BEFORE it is pre-processed. 

Data_object = Data(data, categorical_features, numerical_features, already_splitted_data=False)
X_train, y_train = Data_object.get_training_data()
X_test, y_test = Data_object.get_test_data()
X_valid, y_valid = Data_object.get_validation_data()

# Print the shapes of each data set. 
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Valid: {X_valid.shape}")

# Save the three data sets to disk for later use. 
training = X_train.copy()
training["y"] = y_train
training.to_csv("splitted_data/DI/DI_train.csv") # Save this to csv.
training.to_pickle("splitted_data/DI/DI_train.pkl") # Save as serialized object. 

test = X_test.copy()
test["y"] = y_test
test.to_csv("splitted_data/DI/DI_test.csv") # Save this to csv.
test.to_pickle("splitted_data/DI/DI_test.pkl") # Save as serialized object. 

valid = X_valid.copy()
valid["y"] = y_valid
valid.to_csv("splitted_data/DI/DI_valid.csv") # Save this to csv.
valid.to_pickle("splitted_data/DI/DI_valid.pkl") # Save as serialized object.
