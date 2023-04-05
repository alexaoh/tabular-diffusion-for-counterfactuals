# Define the prediction model for AD, predict the output (binary classification), 
# save the prediction model and save the factuals we want to explain later. 

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import catboost as ctb

from Data import Data
from prediction_model_utils import make_confusion_matrix, calculate_auc_f1_acc

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "find_factuals_DI.py", 
                                     description = "Find factuals from CatBoost predictor on DI.")
    parser.add_argument("-s", "--seed", help="Seed for initializing CatBoostClassifier. Default is 1234.", 
                        type=int, default = 1234, required = False)
    parser.add_argument("--num-factuals", help = "Number of factuals to select. Default is 10.",
                        type = int, default = 10, required = False)
    parser.add_argument("--train", help = "The classifier should be trained.",
                        action = "store_true")
    parser.add_argument("--save-factuals", help = "Factuals should be saved to disk.",
                        action = "store_true")
    args = parser.parse_args()
    return args

def main(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the data. 
    categorical_features = []
    numerical_features = ["num_pregnant", "plasma", "dbp", "skin", "insulin", "bmi", "pedi", "age"]

    # Load the real data into the scope. 
    training = pd.read_csv("splitted_data/DI/DI_train.csv", index_col = 0)
    testing = pd.read_csv("splitted_data/DI/DI_test.csv", index_col = 0)
    valid = pd.read_csv("splitted_data/DI/DI_valid.csv", index_col = 0)
    data = {"Train":training, "Test":testing, "Valid":valid}

    Data_object = Data(data, cat_features = categorical_features, num_features = numerical_features,
                            seed = seed, already_splitted_data=True, scale_version="quantile", valid = True)
    X_train, y_train = Data_object.get_training_data()
    X_test, y_test = Data_object.get_test_data()
    X_valid, y_valid = Data_object.get_validation_data()

    # Find the indices of the categorical features (according to names) in X_train.
    categorical_indices = [X_train.columns.get_loc(c) for c in categorical_features]

    if args.train:
        model = ctb.CatBoostClassifier(random_seed = seed) # We begin by using the default parameters. 
        model.fit(X_train, y_train, cat_features = None, 
               eval_set = (X_valid, y_valid), logging_level = "Verbose")
        model.save_model("predictors/cat_boost_DI"+str(seed)+".dump")

    if not args.train:
        model = ctb.CatBoostClassifier(random_seed = seed)
        model.load_model("predictors/cat_boost_DI"+str(seed)+".dump")
    
    # Make predictions.
    predicted_probs = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    # Evaluate the model — is it decent for our situation?
    make_confusion_matrix(y_test, predictions)
    f1, auc, acc = calculate_auc_f1_acc(y_test, predicted_probs[:,1])
    print(f"F1 score: {f1}")
    print(f"AUC: {auc}")
    print(f"Accuracy: {acc}")
    plt.show()
    # We assume this model is good enough for our purposes!

    # Select randomly 10 individuals who are negatively predicted — these will be our factuals. 
    test_data = X_test.copy()
    test_data["y_true"] = y_test
    test_data["y_pred"] = predictions
    
    negatively_predicted_individuals = test_data[test_data["y_pred"] == 0]
    factuals = negatively_predicted_individuals.sample(n = args.num_factuals, random_state = seed)
    
    # Save the factuals.
    if args.save_factuals:
        factuals.to_csv("factuals/factuals_DI_catboost"+str(args.seed)+".csv")

if __name__ == "__main__":
    args = take_args()
    main(args = args)
