# Define the prediction model for AD, predict the output (binary classification), 
# save the prediction model and save the factuals we want to explain later. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import catboost as ctb

from Data import Data
from prediction_model_utils import make_confusion_matrix, calculate_auc_f1

def main(train, save_factuals):
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    # Load the data. 
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

    # Find the indices of the categorical features (according to names) in X_train.
    categorical_indices = [X_train.columns.get_loc(c) for c in categorical_features]

    if train:
        model = ctb.CatBoostClassifier(random_seed = seed) # We begin by using the default parameters. 
        model.fit(X_train, y_train, cat_features = categorical_indices, 
               eval_set = (X_valid, y_valid), logging_level = "Verbose")
        model.save_model("predictors/cat_boost_AD.dump")

    if not train:
        model = ctb.CatBoostClassifier()
        model.load_model("predictors/cat_boost_AD.dump")
    
    # Make predictions.
    predicted_probs = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    # Evaluate the model — is it decent for our situation?
    make_confusion_matrix(y_test, predictions)
    f1, roc = calculate_auc_f1(y_test, predicted_probs[:,1])
    print(f"f1 score: {f1}")
    print(f"roc: {roc}")
    plt.show()
    # We assume this model is good enough for our purposes!

    # Select randomly 100 individuals who are negatively predicted — these will be our factuals. 
    test_data = X_test.copy()
    test_data["y_true"] = y_test
    test_data["y_pred"] = predictions
    
    negatively_predicted_individuals = test_data[test_data["y_pred"] == 0]
    factuals = negatively_predicted_individuals.sample(n = 100, random_state = seed)
    
    # Save the factuals.
    if save_factuals:
        factuals.to_csv("factuals/factuals_AD_catboost.csv")

if __name__ == "__main__":
    main(train = False, save_factuals = False)
