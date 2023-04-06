# Make tables that appear in thesis. 

# Could use pd.to_latex
# or tabulate library.

import pandas as pd
import numpy as np

# Load our data.


# Make aggregate tables for ML efficacy for each dataset in our thesis. 

###### Make table for ML efficacy over 5 different seeds for TabDDPM, TVAE and MCCE in Experiment 1. 
def make_ML_efficacy():
    df = pd.read_csv("ML_efficacy_catBoost_AD.csv") # Read data. 
    #df = pd.read_csv("ML_efficacy_catBoost_CH.csv") # Read data. 
    #df = pd.read_csv("ML_efficacy_catBoost_DI.csv") # Read data. 

    decimal_rounding = 3

    # Find mean and standard error of each column. 
    means = df.mean()
    sds = df.std()

    # Make new dataframe of desired shape. 
    ml_efficacy = pd.DataFrame()

    # Add average +- standard error to each field. 
    ml_efficacy["F1"] = [f"${means['F1_real']:.{decimal_rounding}f} \pm {sds['F1_real']:.{decimal_rounding}}$", 
                        f"${means['F1_tabddpm']:.{decimal_rounding}f} \pm {sds['F1_tabddpm']:.{decimal_rounding}}$",
                        f"${means['F1_tvae']:.{decimal_rounding}f} \pm {sds['F1_tvae']:.{decimal_rounding}}$",
                        f"${means['F1_mcce']:.{decimal_rounding}f} \pm {sds['F1_mcce']:.{decimal_rounding}}$"]

    ml_efficacy["AUC"] = [f"${means['AUC_real']:.{decimal_rounding}f} \pm {sds['AUC_real']:.{decimal_rounding}}$", 
                        f"${means['AUC_tabddpm']:.{decimal_rounding}f} \pm {sds['AUC_tabddpm']:.{decimal_rounding}}$",
                        f"${means['AUC_tvae']:.{decimal_rounding}f} \pm {sds['AUC_tvae']:.{decimal_rounding}}$",
                        f"${means['AUC_mcce']:.{decimal_rounding}f} \pm {sds['AUC_mcce']:.{decimal_rounding}}$"]

    ml_efficacy["Accuracy"] = [f"${means['acc_real']:.{decimal_rounding}f} \pm {sds['acc_real']:.{decimal_rounding}}$", 
                        f"${means['acc_tabddpm']:.{decimal_rounding}f} \pm {sds['acc_tabddpm']:.{decimal_rounding}}$",
                        f"${means['acc_tvae']:.{decimal_rounding}f} \pm {sds['acc_tvae']:.{decimal_rounding}}$",
                        f"${means['acc_mcce']:.{decimal_rounding}f} \pm {sds['acc_mcce']:.{decimal_rounding}}$"]

    ml_efficacy = ml_efficacy.rename(index = {0:"Identity", 1: "TabDDPM", 2: "TVAE", 3: "MCCE"})
    print(ml_efficacy)

    # Print data frame as latex table that can be copied into latex file. 
    print(ml_efficacy.to_latex(escape = False, caption = "ML efficacy for TabDDPM, TVAE and MCCE for \\textbf{AD} data. The reported numbers are averages over five different random seeds, given as empirical mean $\pm$ standard error. The identity row shows results when using true training and true test data. These numbers are found using CatBoost as classifier. "))


###### Make tables for counterfactuals for TabDDPM, MCCE and TVAE in Experiment 2.
def make_counterfactual_average_tables(data_code):
    # need to make this work with data_code!
    df_mcce = pd.read_csv("counterfactuals/AD_MCCE_final_K10000.csv", index_col = 0) # Read data. 
    df_tvae = pd.read_csv("counterfactuals/AD_TVAE_final_K10000.csv", index_col = 0) # Read data. 
    df_tabddpm = pd.read_csv("counterfactuals/AD_TabDDPM_final_K10000.csv", index_col = 0) # Read data. 

    decimal_rounding = 3

    def calculate_mean_and_ne(df):
        # Check if there are NaNs (which might appear after decoding).
        print(f"Number of NaNs: {len(np.where(pd.isnull(df).any(1))[0])}")

        # Number of factuals wit counterfactuals. 
        ne = df.dropna().shape[0]

        # Before doing further calculations, we drop the NaNs. 
        df = df.dropna()  
        
        # Find mean of each numeric column. 
        means = df.mean(numeric_only=True)
        
        return means, ne

    means_mcce, ne_mcce = calculate_mean_and_ne(df_mcce)
    means_tvae, ne_tvae = calculate_mean_and_ne(df_tvae)
    means_tabddpm, ne_tabddpm = calculate_mean_and_ne(df_tabddpm)

    table = pd.DataFrame()

    table["L0"] = [means_tabddpm["L0"], means_tvae["L0"], means_mcce["L0"]]
    table["Gower"] = [means_tabddpm["L1"], means_tvae["L1"], means_mcce["L1"]]
    table["NCE"] = [ne_tabddpm, ne_tvae, ne_mcce]

    table = table.rename(index = {0:"TabDDPM", 1:"TVAE", 2:"MCCE"})
    print(table)

    # Print data frame as latex table that can be copied into latex file. 
    print(table.to_latex(escape = False, caption = "Average counterfactual data for all methods for AD"))

if __name__ == "__main__":
    #make_ML_efficacy()
    make_counterfactual_average_tables("AD")
