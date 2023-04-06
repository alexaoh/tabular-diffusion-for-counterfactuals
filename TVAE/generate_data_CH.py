# Author: Alexander J Ohrt.
# In this file we use TVAE via the SDV Python API to synthesize data. 

import sys
import os
# Tricks for loading data and libraries from parent directories. 
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import argparse
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

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "generate_data_CH.py", 
                                     description = "Generate synthetic data for CH with TVAE.")
    parser.add_argument("-s", "--seed", help="Seed for random number generators. Default is 1234.", 
                        type=int, default = 1234, required = False)
    parser.add_argument("--train", help = "The model should be trained.",
                        action = "store_true")
    parser.add_argument("--savename", 
                        help = "Name for saving synthetic samples.",
                        required = False)
    parser.add_argument("--num-samples", 
                        help = "Number of samples to generate. Default is the number of observations in the real dataset.",
                        required = False)
    
    # Hyperparameters.
    hyperparams = parser.add_argument_group("Hyperparameters")
    hyperparams.add_argument("--compress-dims", help = "Layers in the encoder. Default is [128,128].",
                             type = int, nargs = "*", default = [128, 128], required = False)
    hyperparams.add_argument("--decompress-dims", help = "Layers in the decoder. Default is [128,128].",
                             type = int, nargs = "*", default = [128, 128], required = False)
    hyperparams.add_argument("-b", "--batch-size", help = "Batch size during training. Default is 512.",
                             type = int, default = 512, required = False)
    hyperparams.add_argument("-e", "--epochs", help = "Epochs during training. Default is 200.",
                             type = int, default = 200, required = False)
    hyperparams.add_argument("--loss-factor", help = "Reconstruction error loss factor. Default is 2.",
                             type = int, default = 2, required = False)
    hyperparams.add_argument("--embedding-dim", help = "Latent dimension. Default is 128.",
                             type = int, default = 128, required = False)
    args = parser.parse_args()
    return args

def main(args):
        # Set seeds for reproducibility. 
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Argument for when running the script. 
        train = args.train

        # Load the data. Load as csv here, since TVAE wants the categorical features are strings (objects) anyway. 
        training = pd.read_csv("splitted_data/CH/CH_train.csv", index_col = 0)
        test = pd.read_csv("splitted_data/CH/CH_test.csv", index_col = 0)
        valid = pd.read_csv("splitted_data/CH/CH_valid.csv", index_col = 0)
        data = {"Train":training, "Test":test, "Valid":valid}

        # Specify column-names in the data sets.
        categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
        numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
        features = numerical_features + categorical_features
        target = ["y"]

        # We return the data before pre-processing, since the pre-processing in TVAE is done according to the method developed by Xu et. al.
        data_object = Data.Data(data, categorical_features, numerical_features, 
                                seed = seed, already_splitted_data=True, scale_version="quantile", valid = True)
        X_train, y_train = data_object.get_training_data()
        X_test, y_test = data_object.get_test_data()
        X_valid, y_valid = data_object.get_validation_data()

        # Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
        X_valid = pd.concat((X_test, X_valid))
        y_valid = pd.concat((y_test, y_valid))

        # This is the dataframe we will use to fit the MCCE object. 
        training_df = X_train.copy()
        training_df["y"] = y_train

        if args.train: 
                # Build a TVAE-object and fit it to the training data. 
                tvae = TVAE(compress_dims = args.compress_dims, decompress_dims = args.decompress_dims, 
                            batch_size = args.batch_size, epochs = args.epochs, loss_factor = args.loss_factor, 
                            embedding_dim = args.embedding_dim)

                print("\n Began fitting.\n")
                tvae.fit(train_data = training_df, discrete_columns = categorical_features + target)
                print("\n Ended fitting. \n")

                # Save fitted model to disk.
                with open("pytorch_models/CH_TVAE"+str(seed)+".obj", "wb") as f:
                        pickle.dump(tvae, f)

        if not args.train:
                # Load fitted model from disk.
                with open("pytorch_models/CH_TVAE"+str(seed)+".obj", 'rb')  as f:
                        tvae = pickle.load(f)
                        tvae.decoder = tvae.decoder.to(device)

        # Sample data.
        print("\n Began sampling.\n")

        d1 = pd.concat((X_train, X_valid))
        d2 = pd.concat((y_train, y_valid))
        d1["y"] = d2

        if args.num_samples is None:
               generated_data = tvae.sample(samples = d1.shape[0]) # Generate CH-size of synthetic data. 
        else:
               generated_data = tvae.sample(samples = int(args.num_samples))
        
        print("\n Ended sampling.\n")

        # Save to disk.
        if args.savename is None:
               generated_data.to_csv("synthetic_data/CH_TVAE"+str(seed)+".csv")
        else:
               generated_data.to_csv("synthetic_data/CH_TVAE_"+str(args.savename)+"_"+str(seed)+".csv")

if __name__ == "__main__":
    args = take_args()
    print(args)
    main(args = args)
