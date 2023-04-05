# Class for generating counterfactuals for a set of factuals. 
# Heavily based on and inspired by mccepy/mcce.MCCE,
# modified for our case with three different generative models, where only MCCE samples conditionally from the latent distribution. 
# Parts of the methods are simply copied from the file indicated above. 

import sys
import re
import numpy as np
import pandas as pd

class ModifiedMCCE():
    """Class for generating possible counterfactuals and post-processing. Specialized to handle MCCE, TVAE or TabDDPM as generative models."""
    def __init__(self, dataset, model, generative_model):
        self.dataset = dataset
        self.continuous = dataset.continuous
        self.categorical = dataset.categorical
        self.immutables = dataset.immutables

        self.model = model # Already fitted model to the data is passed. 
        self.generative_model = generative_model # Name of the generative model we are working with. 
        assert generative_model in ["MCCE", "TVAE", "TabDDPM"], \
                    f"'generative_model' {generative_model} is not valid, must be either 'MCCE', 'TVAE' or 'TabDDPM'."

        self.method = None
        self.visit_sequence = None
        self.predictor_matrix = None

        if hasattr(dataset, 'categorical_encoded'):
            self.categorical_encoded = dataset.categorical_encoded
        else:
            # Get the new categorical feature names after encoding
            self.categorical_encoded = dataset.encoder.get_feature_names(self.categorical).tolist()

        if hasattr(dataset, 'immutables_encoded'):
            self.immutables_encoded = dataset.immutables_encoded
        else:
            # Get the new immutable feature names after encoding
            immutables_encoded = []
            for immutable in self.immutables:
                if immutable in self.categorical:
                    for new_col in self.categorical_encoded:
                        match = re.search(immutable, new_col)
                        if match:
                            immutables_encoded.append(new_col)
                else:
                    immutables_encoded.append(immutable)

            self.immutables_encoded = immutables_encoded

        if not hasattr(self.model, "predict"):
            sys.exit("model does not have predict function.")
        
    def postprocess(self, cfs, test_factual, cutoff = 0.5):
        """Returns final counterfactual for each factual in 'test_factual'."""
        if self.generative_model != "MCCE":
            # Add extra preprocessing to make sure that there are K samples per index of the factual in the generated data. 
            # Then follow the same procedure as MCCE. 
            # PASS FOR NOW! TEST THAT MCCE MAKES SENSE FIRST!
            K = int(cfs.shape/test_factual.shape)
            indices = list(np.array([[i]*K  for i in test_factual.index]).flatten)
            # Not sure if this is correct, but I think so!
            pass

        cols = cfs.columns.to_list()
        self.cutoff = cutoff

        # Predict response of generated data and remove negative predictions, i.e. remove non-valid possible counterfactuals. 
        cfs_positive = cfs[self.model.predict(cfs) >= cutoff]
        
        # We have to do this whole dance to drop duplicates in the same index.
        # But not across indices -- there must be a better way!
        cfs_positive = cfs_positive.reset_index()
        cfs_positive = cfs_positive.drop_duplicates()
        cfs_positive = cfs_positive.set_index(cfs_positive['index'])
        cfs_positive = cfs_positive.drop(columns=['index'])
        self.cfs_positive = cfs_positive
        
        # Duplicate original test observations N times where N is number of positive counterfactuals.
        n_counterfactuals = cfs_positive.groupby(cfs_positive.index).size()
        n_counterfactuals = pd.DataFrame(n_counterfactuals, columns=['nb_unique_pos'])

        fact_repeated = test_factual.copy()
        
        fact_repeated = fact_repeated.join(n_counterfactuals)
        fact_repeated.dropna(inplace = True)

        fact_repeated = fact_repeated.reindex(fact_repeated.index.repeat(fact_repeated['nb_unique_pos']))
        fact_repeated.drop(['nb_unique_pos'], axis=1, inplace=True)
        
        self.fact_repeated = fact_repeated

        # Calculate L0, L1 and L2 for each of the possible counterfactuals by comparing to their corresponding factual. 
        # We make sure the indices are sorted in both dataframes for extra safekeeping.
        self.results = self.calculate_metrics(cfs=cfs_positive.sort_index(), 
                                              test_factual=self.fact_repeated.sort_index()) 
        
        # Find the best sample for each test obs. 
        # This is done by removing rows with sparsity larger than the minimal sparsity and 
        # returning the counterfactuals with smallest Gower distance among these. 
        results_sparse = pd.DataFrame(columns=self.results.columns)

        for idx in list(set(self.results.index)):
            idx_df = self.results.loc[idx]
            if(isinstance(idx_df, pd.DataFrame)): # If you have multiple rows
                sparse = min(idx_df.L0) # Find least number of features changed.
                sparse_df = idx_df[idx_df.L0 == sparse] # Remove all rows that have larger sparsity than minimum.
                closest = min(sparse_df.L1) # Find smallest Gower distance. This is only Manhattan as it is. 
                # The numerical features have not been normalized according to range for Gower!
                # Add this or ignore?
                close_df = sparse_df[sparse_df.L1 == closest].head(1) # Return the remaining row with smallest Gower distance. 

            else: # If you have only one row - return that row.
                close_df = idx_df.to_frame().T
                
            results_sparse = pd.concat([results_sparse, close_df], axis=0) # Add this new row to the dataframe.
        
        # Add the number of positive instances per test observation, i.e. the number of possible counterfactuals this one was chosen from.
        results_sparse = results_sparse.merge(n_counterfactuals, left_index=True, right_index=True)

        cols = cols + ['nb_unique_pos', 'L0', 'L1']
        return results_sparse[cols] # Return the features + number of possible counterfactuals + sparsity + Gower. 
            
    def calculate_metrics(self, cfs, test_factual):
        """Calculate the distance between the potential counterfactuals and the original factuals."""
        features = cfs.columns.to_list()
        cfs_metrics = cfs.copy()
        
        # Make sure that both dataframes contain the same features in the same order, for extra safekeeping.
        factual = test_factual[features]
        counterfactuals = cfs[features]

        print(f"Indices in both lists are equal? --> {all(factual.index == counterfactuals.index)}")
        
        # Calculate sparsity and Euclidean distances.
        distances = pd.DataFrame(self.distance(counterfactuals, factual, self.dataset), index=factual.index)
        cfs_metrics = pd.concat([cfs_metrics, distances], axis=1)

        return cfs_metrics

    def distance(self, cfs, fact, dataset):
        """Calculates three distance functions between potential counterfactuals and original factuals.
        
        Assume that the given dataframes are already descaled and decoded. 
        """
        continuous = dataset.continuous
        categorical = dataset.categorical    

        cf_label_encode = dataset.label_encode(cfs)
        fact_label_encode = dataset.label_encode(fact)

        cfs_categorical = cf_label_encode[categorical].sort_index().to_numpy()
        factual_categorical = fact_label_encode[categorical].sort_index().to_numpy()

        cfs_continuous = cfs[continuous].sort_index().to_numpy()
        factual_continuous = fact[continuous].sort_index().to_numpy()
        
        # Find difference between true feature values and synthetic feature values for each possible counterfactual vs. factual
        # for continuous and categorical features. 
        delta_cont = factual_continuous - cfs_continuous
        delta_cat = factual_categorical - cfs_categorical
        delta_cat = np.where(np.abs(delta_cat) > 0, 1, 0) # If categorical level is not equal we give weight 1. If equal, we give weight 0. 

        delta = np.concatenate((delta_cont, delta_cat), axis=1)

        L0 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05)), axis=1, dtype=np.float).tolist()
        L1 = np.sum(np.abs(delta), axis=1, dtype=np.float).tolist() # The numerical features have not been normalized according to range for Gower!
        L2 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()

        return({'L0': L0, 'L1': L1, 'L2': L2})
    
class Dataset():
    """Class to be passed to ModifiedMCCE-class containing necessary variables."""
    def __init__(self, 
                continuous,
                categorical,
                categorical_encoded, 
                immutables,
                label_encode
                ):

        self.continuous = continuous
        self.categorical = categorical
        self.categorical_encoded = categorical_encoded # Add this to work with MCCE class constructor.
        self.immutables = immutables  
        self.label_encode = label_encode # Function for label encoding a dataframe. 