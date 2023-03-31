# Author: Alexander J Ohrt.
# In this file we use MCCE to generate counterfactuals for Experiment 2 in my master's thesis. 
# We generate one counterfactual per factual, that we have found previously, based on previously fitted CatBoost predictor.

import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import argparse
import pandas as pd
import numpy as np
import random
import catboost as ctb

from mcce.mcce import MCCE # For working with mcce.
from mcce.data import Data # For making data sets to work with mcce. 

import Data # Import my class for scaling/encoding, etc.

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "generate_counterfactuals_AD.py", 
                                     description = "Generate counterfactuals for factuals in AD with MCCE.")
    parser.add_argument("-s", "--seed", help="Seed for random number generators. Default is 1234.", 
                        type=int, default = 1234, required = False)
    args = parser.parse_args()
    return args

def main(args):
    # Set seeds for reproducibility. 
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the original data. Load as csv here, since we change the dtypes according to method in README later anyway. 
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
    immutable_features = ["age", "sex"]

    data_object = Data.Data(data, categorical_features, numerical_features, already_splitted_data=True, scale_version="quantile", valid = True)
    X_train, y_train = data_object.get_training_data_preprocessed()
    X_test, y_test = data_object.get_test_data_preprocessed()
    X_valid, y_valid = data_object.get_validation_data_preprocessed()

    # Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
    X_valid = pd.concat((X_test, X_valid))
    y_valid = pd.concat((y_test, y_valid))

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
    categorical_encoded = data_object.encoder.get_feature_names_out(categorical_features).tolist()
    immutables = immutable_features

    # "Fix" the data types.
    dtypes = dict([(x, "float") for x in continuous])
    for x in categorical_encoded:
        dtypes[x] = "category"
    training_df2 = training_df.copy()
    dtypes["y"] = "category"
    training_df2 = (training_df2).astype(dtypes)

    # Make the data object.
    dataset = Dataset(continuous, categorical, categorical_encoded, immutables)

    # Load the classifier we used to find the factuals. 
    model = ctb.CatBoostClassifier()
    model.load_model("../predictors/cat_boost_AD"+str(seed)+".dump")

    # Make the MCCE object.
    mcce = MCCE(dataset, model, seed = seed)

    # Fit the MCCE model. 
    mcce.fit(training_df2.drop(target, axis = 1), dtypes) # Do not include the target. 

    # Load the factuals we want to explain. 
    factuals = pd.read_csv("../factuals_AD_catboost"+str(seed)+".csv", index_col = 0)

    # Encode the factuals. 
    factuals_enc = data_object.encode(factuals)
    factuals_enc = data_object.scale(factuals_enc)

    # Generate k = 10000 possible counterfactuals per factual. 
    cfs = mcce.generate(factuals_enc.drop(("y_true", "y_pred"), axis=1), k=10000)

    # Postprocess the samples, such that we are left with one counterfactual per factual.
    mcce.postprocess(cfs, factuals_enc, cutoff=0.5) # predicted >= 0.5 is considered positive; < 0.5 is negative
    # I would prefer if it returned the processed counterfactuals. We could do this for the two other methods.
    # Understand what it does and apply the same methodology to the two other generative models!!
    cfs = mcce.results_sparse # Instead, I believe this method saves the postprocessed data in "results_sparse".

    # Decode the counterfactuals.
    cfs_dec = data_object.decode(cfs)
    cfs_dec = data_object.decode(cfs_dec)

    # Compare the factuals and the counterfactuals (visually).
    print(factuals.iloc[:5, :])
    print(cfs_dec.iloc[:5, :])

if __name__ == "__main__":
    args = take_args()
    main(args = args)
