# We check ML efficacy of synthetic data with CatBoost as predictor. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics # plot_roc_curve.

from Data import Data, CustomDataset, ToTensor

import catboost as ctb

from prediction_model_utils import make_confusion_matrix_v2, calculate_auc_f1_v2

def main():
    """Main function to run if file is called directly."""
    # Set seeds for reproducibility. 
    seed = 1234
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

    # Load the synthetic data into the scope. 
    synthetic_samples = pd.read_csv("synthetic_data/AD_Gaussian_multinomial_diffusion.csv", index_col = 0)
    print(f"synthetic_samples.shape: {synthetic_samples.shape}")

    Synth = Data(synthetic_samples, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train_s, y_train_s = Synth.get_training_data()
    X_valid_s, y_valid_s = Synth.get_validation_data()
    print(f"X_train_synthetic.shape: {X_train_s.shape}")

    # Define the CatBoost models.
    # Implement following this: https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb
    # to optimize the implementation.
    model_real = ctb.CatBoostClassifier()
    model_synth = ctb.CatBoostClassifier()

    # Fit the models.
    model_real.fit(X_train, y_train, cat_features = categorical_features)
    model_synth.fit(X_train_s, y_train_s, cat_features = categorical_features)

    predicted_real = model_real.predict_proba(X_test)

    predicted_synth = model_synth.predict_proba(X_test)

    # Plot classification matrix and print some more stats.
    make_confusion_matrix_v2(y_test, predicted_real[:,1], predicted_synth[:,1])

    # Calculate f1 score and auc, and return these values.
    f1_true, auc_true, f1_synth, auc_synth = calculate_auc_f1_v2(y_test, predicted_real[:,1], predicted_synth[:,1])

    print(f"F1 score from real data: {f1_true}")
    print(f"F1 score from synthetic data: {f1_synth}")
    print(f"AUC from real data: {auc_true}")
    print(f"AUC from synthetic data {auc_synth}")

if __name__ == "__main__":
    main()
