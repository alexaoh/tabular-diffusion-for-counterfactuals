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

from mccepy.mcce.mcce import MCCE # For working with mcce.
from mccepy.mcce.data import Data # For making data sets to work with mcce. 

from ModifiedMCCE import ModifiedMCCE, Dataset # Import ModifiedMCCE class and Dataset class. 

import Data # Import my class for scaling/encoding, etc.

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "CH_MCCE_generate_counterfactuals.py", 
                                     description = "Generate counterfactuals for factuals in CH with MCCE.")
    parser.add_argument("-s", "--seed", help="Seed for random number generators. Default is 1234.", 
                        type=int, default = 1234, required = False)
    parser.add_argument("-K", help = "Number of observations to generated per factual. Default is K = 10000.",
                        type = int, default = 10000, required = False)
    parser.add_argument("-g", "--generate", help = "If we should generate possible counterfactuals. Default is 'True' (bool).",
                        type = bool, default = True, required = False)

    args = parser.parse_args()
    return args

def main(args):
    # Set seeds for reproducibility. 
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the original data. Load as csv here, since we change the dtypes according to method in README later anyway. 
    training = pd.read_csv("splitted_data/CH/CH_train.csv", index_col = 0)
    test = pd.read_csv("splitted_data/CH/CH_test.csv", index_col = 0)
    valid = pd.read_csv("splitted_data/CH/CH_valid.csv", index_col = 0)
    data = {"Train":training, "Test":test, "Valid":valid}

    # Specify column-names in the data sets.
    categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
    numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    features = numerical_features + categorical_features
    target = ["y"]
    immutable_features = ["Age", "Gender"]

    data_object = Data.Data(data, categorical_features, numerical_features, 
                            seed = seed, already_splitted_data=True, scale_version="quantile", valid = True)
    X_train, y_train = data_object.get_training_data_preprocessed()
    X_test, y_test = data_object.get_test_data_preprocessed()
    X_valid, y_valid = data_object.get_validation_data_preprocessed()

    # Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
    X_valid = pd.concat((X_test, X_valid))
    y_valid = pd.concat((y_test, y_valid))

    # This is the dataframe we will use to fit the MCCE object. 
    training_df = X_train.copy()
    training_df["y"] = y_train   

    continuous = numerical_features
    categorical = categorical_features
    categorical_encoded = data_object.encoder.get_feature_names_out(categorical_features).tolist()
    immutables = immutable_features
    label_encode = data_object.label_encode # Function for label encoding the categorical features of a dataframe. 
    X_train = data_object.get_training_data()[0]

    # "Fix" the data types.
    dtypes = dict([(x, "float") for x in continuous])
    for x in categorical_encoded:
        dtypes[x] = "category"
    training_df2 = training_df.copy()
    dtypes["y"] = "category"
    training_df2 = (training_df2).astype(dtypes)

    # Make the data object.
    dataset = Dataset(continuous, categorical, categorical_encoded, immutables, label_encode, X_train)

    # Load the classifier we used to find the factuals. 
    model = ctb.CatBoostClassifier()
    model.load_model("predictors/cat_boost_CH"+str(seed)+".dump")

    # Make the MCCE object.
    mcce = MCCE(dataset, model, seed = seed)

    # Fit the MCCE model. 
    mcce.fit(training_df2.drop(target, axis = 1), dtypes) # Do not include the target. 

    # Load the factuals we want to explain. 
    factuals = pd.read_csv("factuals/factuals_CH_catboost"+str(seed)+".csv", index_col = 0)

    # Encode the factuals. 
    factuals_enc = data_object.encode(factuals)
    factuals_enc = data_object.scale(factuals_enc)

    if args.generate:
        # Generate k = args.K possible counterfactuals per factual. 
        cfs = mcce.generate(factuals_enc.drop(["y_true", "y_pred"], axis=1), k=args.K)

        # Decode and descale the generated data, since our CatBoost model works with unscaled data. 
        cfs = data_object.decode(cfs)
        cfs = data_object.descale(cfs)

        # Check if there are NaNs (which might appear after decoding).
        print(f"Number of NaNs: {len(np.where(pd.isnull(cfs).any(axis = 1))[0])}")
        cfs = cfs.dropna() # We simply drop rows with NaNs, instead of imputing. 

        # Save the generated possible counterfactuals, Dh, to disk.
        cfs.to_csv("counterfactuals/CH_MCCE_Dh_K"+str(args.K)+"_"+str(seed)+".csv")
    else: 
        # Load the generated possible counterfactuals, Dh, from disk. 
        cfs = pd.read_csv("counterfactuals/CH_MCCE_Dh_K"+str(args.K)+"_"+str(seed)+".csv", index_col = 0)

    # Make ModifiedMCCE object for post-processing the generated samples. 
    modified_mcce = ModifiedMCCE(dataset, model, generative_model = "MCCE")

    # Postprocess the samples, such that we are left with one counterfactual per factual.
    cfs = modified_mcce.postprocess(cfs, factuals.drop(["y_true", "y_pred"], axis=1), cutoff=0.5) # predicted >= 0.5 is considered positive; < 0.5 is negative.

    # Sort the counterfactuals according to the index of the factual dataframe. 
    cfs = cfs.reindex(factuals.index)

    # Make predictions on the counterfactuals to show that they now lead to a positive prediction.
    print(f"Number of NaNs in post-processed dataframe, i.e. number of missing counterfactuals: {len(np.where(pd.isnull(cfs).any(axis = 1))[0])}")
    cfs2 = cfs.copy().dropna() # Drop NA in case they exist, i.e. some factuals are missing counterfactuals. 
    cfs["new_preds"] = np.nan
    cfs.loc[cfs2.index, "new_preds"] = model.predict(cfs2) 

    # Compare the factuals and the counterfactuals (visually).
    print(factuals.iloc[:5, :])
    print(cfs.iloc[:5, :])

    # Save the counterfactuals to disk. 
    cfs.to_csv("counterfactuals/CH_MCCE_final_K"+str(args.K)+"_"+str(seed)+".csv")

if __name__ == "__main__":
    args = take_args()
    main(args = args)
