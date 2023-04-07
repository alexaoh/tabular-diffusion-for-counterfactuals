# Author: Alexander J Ohrt.
# In this file we use TVAE to generate counterfactuals for Experiment 2 in my master's thesis. 
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

from ModifiedMCCE import ModifiedMCCE, Dataset # Import ModifiedMCCE class and Dataset class. 

import Data # Import my class for scaling/encoding, etc.

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "DI_TVAE_generate_counterfactuals.py", 
                                     description = "Generate counterfactuals for factuals in DI with TVAE.")
    parser.add_argument("-s", "--seed", help="Seed for random number generators. Default is 1234.", 
                        type=int, default = 1234, required = False)
    parser.add_argument("-K", help = "N/A: We generate K = 10000 possible counterfactuals straight after training the model.",
                        type = int, default = 10000, required = False)
    parser.add_argument("-g", "--generate", help = "N/A: We generate K = 10000 possible counterfactuals straight after training the model.",
                        type = bool, default = False, required = False)

    args = parser.parse_args()
    return args

def main(args):
    # Set seeds for reproducibility. 
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the original data. Load as csv here, since we change the dtypes according to method in README later anyway. 
    training = pd.read_csv("splitted_data/DI/DI_train.csv", index_col = 0)
    test = pd.read_csv("splitted_data/DI/DI_test.csv", index_col = 0)
    valid = pd.read_csv("splitted_data/DI/DI_valid.csv", index_col = 0)
    data = {"Train":training, "Test":test, "Valid":valid}

    # Specify column-names in the data sets.
    categorical_features = []
    numerical_features = ["num_pregnant", "plasma", "dbp", "skin", "insulin", "bmi", "pedi", "age"]
    features = numerical_features + categorical_features
    target = ["y"]
    immutable_features = ["age"]

    data_object = Data.Data(data, categorical_features, numerical_features, 
                            seed = seed, already_splitted_data=True, scale_version="quantile", valid = True)

    continuous = numerical_features
    categorical = categorical_features
    categorical_encoded = categorical
    immutables = immutable_features
    label_encode = data_object.label_encode # Function for label encoding the categorical features of a dataframe. 
    X_train = data_object.get_training_data()[0]

    # Make the data object.
    dataset = Dataset(continuous, categorical, categorical_encoded, immutables, label_encode, X_train)

    # Load the classifier we used to find the factuals. 
    model = ctb.CatBoostClassifier()
    model.load_model("predictors/cat_boost_DI"+str(seed)+".dump")

    # Load the factuals we want to explain. 
    factuals = pd.read_csv("factuals/factuals_DI_catboost"+str(seed)+".csv", index_col = 0)

    # WE ASSUME the following is already done. 
    # These are generated separately, and loaded from 'synthetic_data'-directory. 
    # It is important that the hyperparameters match the hyperparameters of the pre-trained model. 
    # Check this is in other scripts or in final thesis report. In order to satisfy this requirement, we use the same pipeline for generating this data as earlier. 
    # The CLI line below is used to run the file, then save the generated samples:
    # python TVAE/generate_data_DI.py -s 1234 -d DI -t True -g True -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance "None" --num-samples 10000 --savename "TVAE_K10000_"
        
    # Load the generated possible counterfactuals, Dh, from disk. 
    cfs = pd.read_csv("synthetic_data/DI_TVAE_K"+str(args.K)+"_"+str(args.seed)+".csv", index_col = 0)
    # Check if there are NaNs (which might appear after decoding).
    print(f"Number of NaNs: {len(np.where(pd.isnull(cfs).any(1))[0])}")
    cfs = cfs.dropna() # Drop NaNs just in case there are any. 

    # Drop the "y"-column (lable) that is generated from TVAE from the cfs. This is generated since we are using response-conditional neural networks. 
    cfs = cfs.drop("y", axis = 1)

    # Make ModifiedMCCE object for post-processing the generated samples. 
    modified_mcce = ModifiedMCCE(dataset, model, generative_model = "TVAE")

    # Postprocess the samples, such that we are left with one counterfactual per factual.
    cfs = modified_mcce.postprocess(cfs, factuals.drop(["y_true", "y_pred"], axis=1), cutoff=0.5) # predicted >= 0.5 is considered positive; < 0.5 is negative.

    # Sort the counterfactuals according to the index of the factual dataframe. 
    # This also assures that: 
    # If any of the indices of the original factuals are missing, add row with this index to results_sparse with all NA values. 
    # This cannot happen for MCCE because of the conditional fixed sampling, but may happen with other models. 
    cfs = cfs.reindex(factuals.index)

    # Make predictions on the counterfactuals to show that they now lead to a positive prediction.
    print(f"Number of NaNs in post-processed dataframe, i.e. number of missing counterfactuals: {len(np.where(pd.isnull(cfs).any(1))[0])}")
    cfs2 = cfs.copy().dropna() # Drop NA in case they exist, i.e. some factuals are missing counterfactuals. 
    cfs["new_preds"] = np.nan 
    cfs.loc[cfs2.index, "new_preds"] = model.predict(cfs2) 

    # Compare the factuals and the counterfactuals (visually).
    print(factuals.iloc[:5, :])
    print(cfs.iloc[:5, :])

    # Save the counterfactuals to disk. 
    cfs.to_csv("counterfactuals/DI_TVAE_final_K"+str(args.K)+".csv")

if __name__ == "__main__":
    args = take_args()
    main(args = args)
