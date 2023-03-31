# We check ML efficacy of synthetic data with CatBoost as predictor. 

import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Data import Data

import catboost as ctb

from prediction_model_utils import make_confusion_matrix, make_confusion_matrix_v2, calculate_auc_f1_acc, calculate_auc_f1_v2

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "ML_efficacy_catBoost_AD.py", 
                                     description = "Calculate metrics for ML Efficacy on AD data.")
    parser.add_argument("-s", "--seed", help="Seed for initializing CatBoostClassifier.", 
                        type=int, default = 1234, required = False)
    args = parser.parse_args()
    return args

def main(args):
    """Main function to run if file is called directly."""
    # Set seeds for reproducibility. 
    seed = args.seed
    np.random.seed(seed)

    categorical_features = ["workclass","marital_status","occupation","relationship", \
                        "race","sex","native_country"]
    numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

    # Load the real data into the scope. 
    training = pd.read_csv("splitted_data/AD/AD_train.csv", index_col = 0)
    testing = pd.read_csv("splitted_data/AD/AD_test.csv", index_col = 0)
    valid = pd.read_csv("splitted_data/AD/AD_valid.csv", index_col = 0)
    data = {"Train":training, "Test":testing, "Valid":valid}

    Data_object = Data(data, cat_features = categorical_features, num_features = numerical_features,
                            already_splitted_data=True, scale_version="quantile", valid = True)
    X_train, y_train = Data_object.get_training_data()
    X_test, y_test = Data_object.get_test_data()
    X_valid, y_valid = Data_object.get_validation_data()
    print(f"X_train.shape: {X_train.shape}")

    adult_data = Data_object.get_original_data()
    print(f"adult_data.shape: {adult_data.shape}")

    # Load the synthetic data into the scope. We do this for TabDDPM, MCCE-trees and TVAE. 
    synth_tabddpm = pd.read_csv("synthetic_data/AD_Gaussian_multinomial_diffusion"+str(seed)+".csv", index_col = 0)
    synth_mcce = pd.read_csv("synthetic_data/AD_from_trees"+str(seed)+".csv", index_col = 0)
    synth_tvae = pd.read_csv("synthetic_data/AD_TVAE"+str(seed)+".csv", index_col = 0)
    print(f"synth_tabddpm.shape: {synth_tabddpm.shape}")
    print(f"synth_mcce.shape: {synth_mcce.shape}")
    print(f"synth_tvae.shape: {synth_tvae.shape}")

    Synth_tabddpm = Data(synth_tabddpm, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train_tabddpm, y_train_tabddpm = Synth_tabddpm.get_training_data()
    X_valid_tabddpm, y_valid_tabddpm = Synth_tabddpm.get_validation_data()
    print(f"X_train_tabddpm.shape: {X_train_tabddpm.shape}")

    Synth_mcce = Data(synth_mcce, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train_mcce, y_train_mcce = Synth_mcce.get_training_data()
    X_valid_mcce, y_valid_mcce = Synth_mcce.get_validation_data()
    print(f"X_train_mcce.shape: {X_train_mcce.shape}")

    Synth_tvae = Data(synth_tvae, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train_tvae, y_train_tvae = Synth_tvae.get_training_data()
    X_valid_tvae, y_valid_tvae = Synth_tvae.get_validation_data()
    print(f"X_train_tvae.shape: {X_train_tvae.shape}")

    # Find the indices of the categorical features (according to names) in X_train.
    # We assume this indices are the same in the true and synthetic data.
    categorical_indices = [X_train.columns.get_loc(c) for c in categorical_features]
    categorical_indices_tabddpm = [X_train_tabddpm.columns.get_loc(c) for c in categorical_features]
    categorical_indices_mcce = [X_train_mcce.columns.get_loc(c) for c in categorical_features]
    categorical_indices_tvae = [X_train_tvae.columns.get_loc(c) for c in categorical_features]

    # Define the CatBoost models.
    # Implement following this: https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb
    model_real = ctb.CatBoostClassifier(random_seed = seed)
    model_tabddpm = ctb.CatBoostClassifier(random_seed = seed)
    model_mcce = ctb.CatBoostClassifier(random_seed = seed)
    model_tvae = ctb.CatBoostClassifier(random_seed = seed)
    
    # Fit the models.
    model_real.fit(X_train, y_train, cat_features = categorical_indices, 
               eval_set = (X_valid, y_valid), logging_level = "Silent")
    print("Fitted real data model.")
    model_tabddpm.fit(X_train_tabddpm, y_train_tabddpm, cat_features = categorical_indices_tabddpm, 
               eval_set = (X_valid_tabddpm, y_valid_tabddpm), logging_level = "Silent")
    print("Fitted fake TabDDPM data model.")
    model_mcce.fit(X_train_mcce, y_train_mcce, cat_features = categorical_indices_mcce, 
               eval_set = (X_valid_mcce, y_valid_mcce), logging_level = "Silent")
    print("Fitted fake MCCE data model.")
    model_tvae.fit(X_train_tvae, y_train_tvae, cat_features = categorical_indices_tvae, 
               eval_set = (X_valid_tvae, y_valid_tvae), logging_level = "Silent")
    print("Fitted fake TVAE data model.")

    predicted_probs_real = model_real.predict_proba(X_test)
    predictions_real = model_real.predict(X_test)
    predicted_probs_tabddpm = model_tabddpm.predict_proba(X_test)
    predictions_tabddpm = model_tabddpm.predict(X_test)
    predicted_probs_mcce = model_mcce.predict_proba(X_test)
    predictions_mcce = model_mcce.predict(X_test)
    predicted_probs_tvae = model_tvae.predict_proba(X_test)
    predictions_tvae = model_tvae.predict(X_test)

    # Plot classification matrix and print some more stats.
    make_confusion_matrix(y_test, predictions_real, text = "on real data.")
    make_confusion_matrix(y_test, predictions_tabddpm, text = "on fake TabDDPM data.")
    make_confusion_matrix(y_test, predictions_mcce, text = "on fake MCCE data.")
    make_confusion_matrix(y_test, predictions_tvae, text = "on fake TVAE data.")
    plt.show()

    # Calculate f1 score, auc and accuracy. 
    f1_real, auc_real, acc_real = calculate_auc_f1_acc(y_test, predicted_probs_real[:,1])
    f1_tabddpm, auc_tabddpm, acc_tabddpm = calculate_auc_f1_acc(y_test, predicted_probs_tabddpm[:,1])
    f1_mcce, auc_mcce, acc_mcce = calculate_auc_f1_acc(y_test, predicted_probs_mcce[:,1])
    f1_tvae, auc_tvae, acc_tvae = calculate_auc_f1_acc(y_test, predicted_probs_tvae[:,1])

    # Save the scores as a csv for later aggregation into tables in report. 
    d = [[seed, f1_real, f1_tabddpm, f1_mcce, f1_tvae, 
          auc_real, auc_tabddpm, auc_mcce, auc_tvae,
          acc_real, acc_tabddpm, acc_mcce, acc_tvae]]
    df_columns = ["seed", "F1_real", "F1_tabddpm", "F1_mcce", "F1_tvae", 
                  "AUC_real", "AUC_tabddpm", "AUC_mcce", "AUC_tvae", 
                  "acc_real", "acc_tabddpm", "acc_mcce", "acc_tvae"]
    df = pd.DataFrame(data = d, columns = df_columns)

    filename = "ML_efficacy_catBoost_AD.csv"
    if os.path.isfile(filename):
        # Append to csv file if it exists. 
        df.to_csv(filename, mode = "a", index = False, header = False)
    else:
        # If the file does not exist, make it, with the correct column labels.
        df.to_csv(filename,index = False)

if __name__ == "__main__":
    args = take_args()
    main(args = args)
