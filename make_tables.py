# Make tables that appear in thesis. 

import pandas as pd
import numpy as np
import random

seed = 1234

np.random.seed(seed)
random.seed(seed)

###### Make table for ML efficacy over 5 different seeds for TabDDPM, TVAE and MCCE in Experiment 1. 
def make_ML_efficacy(data_code):
    if data_code == "AD":
        df = pd.read_csv("ML_efficacy_catBoost_AD.csv") 
    elif data_code == "CH":
        df = pd.read_csv("ML_efficacy_catBoost_CH.csv") 
    elif data_code == "DI":
        df = pd.read_csv("ML_efficacy_catBoost_DI.csv")

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
    print(ml_efficacy.to_latex(escape = False, caption = "ML efficacy for TabDDPM, TVAE and MCCE for \\textbf{data_code} data. The reported numbers are averages over five different random seeds, given as empirical mean $\pm$ standard error. The identity row shows results when using true training and true test data. These numbers are found using CatBoost as classifier. "))


###### Make tables for counterfactuals for TabDDPM, MCCE and TVAE in Experiment 2.
def make_counterfactual_average_tables(data_code):
    df_mcce = pd.read_csv("counterfactuals/"+data_code+"_MCCE_final_K10000.csv", index_col = 0)
    df_tvae = pd.read_csv("counterfactuals/"+data_code+"_TVAE_final_K10000.csv", index_col = 0)
    df_tabddpm = pd.read_csv("counterfactuals/"+data_code+"_TabDDPM_final_K10000.csv", index_col = 0)

    decimal_rounding = 2

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

    table["L0"] = [f"{means_tabddpm['L0']:.{decimal_rounding}f}", f"{means_tvae['L0']:.{decimal_rounding}f}", 
                   f"{means_mcce['L0']:.{decimal_rounding}f}"]
    table["Gower"] = [f"{means_tabddpm['L1']:.{decimal_rounding}f}", f"{means_tvae['L1']:.{decimal_rounding}f}", 
                   f"{means_mcce['L1']:.{decimal_rounding}f}"]
    table["NCE"] = [ne_tabddpm, ne_tvae, ne_mcce]

    table = table.rename(index = {0:"TabDDPM", 1:"TVAE", 2:"MCCE"})
    table = table.rename(columns = {"L0": f"$L_0\downarrow$", "Gower": f"$\\text{{Gower}}\downarrow$", "NCE": f"$N_\\text{{CE}}\\uparrow$"}, errors="raise")
    print(table)
    

    # Print data frame as latex table that can be copied into latex file. 
    #s = table.style.highlight_max(props = "cellcolor: [HTML]{F2F2F2}")
    #table.style.format(escape = "latex")
    print(table.to_latex(escape = False, caption = "Average counterfactual performance metrics for 100 test observations from "+f"\\textbf{{{data_code}}}"+". Performance is indicated with TabDDPM, TVAE and MCCE as generative models, while the postprocessing steps are equal in all three cases. "+f"$\\boldsymbol{{K = 10000}}$"+", meaning that we generate this many possible counterfactuals per test observation (factual). $L_0$ represents the sparsity metric, Gower represents Gower's distance and $N_\\text{{CE}}$ represents  the number of test observations that are given a counterfactual. Downward arrows symbolize that lower is better, while upward arrow symbolize that higher is better."))

def make_individual_counterfactual_comparisons(data_code):
    df_mcce = pd.read_csv("counterfactuals/"+data_code+"_MCCE_final_K10000.csv", index_col = 0)
    df_tvae = pd.read_csv("counterfactuals/"+data_code+"_TVAE_final_K10000.csv", index_col = 0)
    df_tabddpm = pd.read_csv("counterfactuals/"+data_code+"_TabDDPM_final_K10000.csv", index_col = 0)
    real_factuals = pd.read_csv("factuals/factuals_"+data_code+"_catboost1234.csv", index_col = 0)

    columns = real_factuals.columns.tolist()[:-2] # Get the correct column names (remove "y_true" and "y_pred").

    decimal_rounding = 2

    # Choose a random index among those that exist in all three dataframes.
    df_mcce_nona = df_mcce.dropna()
    df_tvae_nona = df_tvae.dropna()
    df_tabddpm_nona = df_tabddpm.dropna()

    common_indices = np.array(list(set(df_mcce_nona.index).intersection(set(df_tvae_nona.index), set(df_tabddpm_nona.index))))

    # DURING TESTING THIS FUNCTION WE USE THIS ONE INSTEAD! REMOVE LATER!
    common_indices = np.array(list(set(df_mcce.index).intersection(set(df_tvae.index), set(df_tabddpm.index))))

    random_index = np.random.choice(common_indices)

    # We use this random index to reselect the element from each of the dataframes. 
    el_mcce = df_mcce[df_mcce.index == random_index]
    el_tvae = df_tvae[df_tvae.index == random_index]
    el_tabddpm = df_tabddpm[df_tabddpm.index == random_index]
    el_real = real_factuals[real_factuals.index == random_index]

    # Then we display these observations in a table.
    table = pd.DataFrame(index = columns)
    table[f"$h$"] = el_real[columns].values.flatten()
    table["TabDDPM"] = el_tabddpm[columns].astype(el_real[columns].dtypes).values.flatten()
    table["TVAE"] = el_tvae[columns].astype(el_real[columns].dtypes).values.flatten()
    table["MCCE"] = el_mcce[columns].astype(el_real[columns].dtypes).values.flatten()

    print(table)   

    # Rename rows to fit our latex style. 
    index_dict = {}
    for feat in columns:
        feat_l = feat.split("_")
        if len(feat_l) > 1:
            feat_new = "\\_".join(feat_l)
            index_dict[feat] = f"\\ttfamily{{{feat_new}}}"
        else:
            index_dict[feat] = f"\\ttfamily{{{feat}}}"

    table = table.rename(index = index_dict)

    print(table.to_latex(escape = False, caption = "Comparison of three different counterfactuals for the same factual "+f"$h$"+", for dataset "+f"\\textbf{{{data_code}}}."))
    

if __name__ == "__main__":
    data_code = "AD"
    #make_ML_efficacy(data_code)
    #make_counterfactual_average_tables(data_code)
    #make_individual_counterfactual_comparisons(data_code)
