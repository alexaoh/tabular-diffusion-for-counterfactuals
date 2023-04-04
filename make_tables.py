# Make tables that appear in thesis. 

# Could use pd.to_latex
# or tabulate library.

import pandas as pd

# Load our data.


# Make aggregate tables for ML efficacy for each dataset in our thesis. 


###### Make table for ML efficacy over 5 different seeds for TabDDPM, TVAE and MCCE in Experiment 1. 
df = pd.read_csv("ML_efficacy_catBoost_AD.csv") # Read data. 
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
