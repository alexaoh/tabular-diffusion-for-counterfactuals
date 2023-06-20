# Probabilistic Tabular Diffusion for Counterfactual Explanation Synthesis

> Master’s Thesis in Applied Physics and Mathematics at NTNU 

### Author: Alexander J Ohrt. 
### Supervisor: Kjersti Aas.
### June 2023. 

## *Abstract*
Recent mainstream popularization of *artificial intelligence* (AI) has led to both positive and negative sentiments concerning the future of the technology. Several of the current most notable AI systems can be categorized as *deep generative AI*, a term that encompasses highly complex models capable of generating data from different modalities. Another subfield called *explainable AI* (XAI) aims to develop methods to increase understanding of opaque prediction models, an objective that both researchers and legislators continue to direct considerable efforts towards. An emerging, especially human-friendly technique from XAI corresponds to *counterfactual explanations*, which are valuable explanations for individual predictions. In this thesis, we combine these two seemingly contradictory subfields of AI, by applying deep generative models to synthesize counterfactual explanations. 

Our main contributions are threefold. First, we develop an accessible and self-contained exposition of *diffusion probabilistic models*, the generative models that underpin several of the most successful technologies for generating data, for example, in art. Second, we add to the literature on diffusion models applied to tabular data, by dissecting and thoroughly explaining the key components of one such model. Third, we utilize the tabular diffusion model to generate counterfactual explanations, by altering one specific model-agnostic algorithm. The generative performance of the tabular diffusion model is evaluated on three publicly available, real datasets against two previously demonstrated models — one deep *variational autoencoder* model and one shallow *decision tree* model. Moreover, counterfactual explanations are computed using the three models as foundations, in order to evaluate their usefulness for explaining an arbitrary binary classifier. 

In our experiments, we observe that all three models are able to generate tabular data and counterfactual explanations, but with differing levels of faithfulness and reliability. In fact, we do not find sufficient evidence to conclude that the considered diffusion model is superior to the baselines, neither at generating data from an approximated unknown joint distribution nor at generating counterfactual explanations for clarifying binary predictions on test observations. Due to promising results, we urge researchers to consider the out-of-the-box tree-based model as a reference during evaluation in further work on deep generative modelling for tabular data. Finally, we provide possible directions for future research on diffusion models for tabular data and counterfactual explanations. 

__NB: Link to publication on NTNU Open will be added when available.__

### What does this repository contain?

This repository contains the code we have developed while working on the Master's thesis. Below we highlight the file structure and add a description of the most important files, to facilitate easier navigation for the interested reader. 

A short description of each of the directories and files is given in the following list: 

* *TVAE*: Contains individual files for each dataset for generating data with TVAE.
* *bash_scripts*: Contains shell scripts that are used to produce all our results. The scripts are used to generate data, for calculating ML efficacy metrics and for generating counterfactuals from each of the generative models. Use these scripts as examples of how the other code can be run. 
* *factuals*: Contains .csv files with randomly sampled factuals from each test dataset in each trial.
* *generate_counterfactuals*: Contains code for generating counterfactuals for the factuals in *factuals* from each of the generative models for each dataset. 
* *loading_data*: Contains separate directories with code for each dataset that performs initial data pre-processing, as described in Appendix D. The pre-processed datasets are saved both as .csv and .pkl (binary) for use in the rest of the code.
* *mccepy*: Contains the entire open source [implementation](https://github.com/NorskRegnesentral/mccepy) of MCCE, in addition to our added files. We add the files "generate_data_??.py", one for each dataset, for generating tabular data from the first two steps of MCCE. 
* *original_data*: Contains all the original data, as it comes when downloading from the internet. 
* *plotting*: Contains Jupyter Notebooks we used to analyze our results. All the figures we added in the thesis can be reconstructed with these notebooks. 
* *predictors*: Contains the CatBoost predictors we trained in each trial on each dataset. Saved as .dump in order to avoid unnecessary retraining. 
* *pytorch_models*: Contains some of the models we trained during experimentation, saved as .pth files. All the models are not saved here, because of memory restrictions on GitHub.
* *splitted_data*: Contains code for splitting the datasets into train, validation and test, as well as saving each of the datasets as both .csv and .pkl (binary) for use in the rest of the code.
* *Data.py*: Class for pre-processing and handling data.
* *Gaussian_diffusion.py*: Main class for Gaussian diffusion process. 
* *ML_efficacy_catBoost_??.py*: Code for calculating ML efficacy metrics, based on CatBoost classifier. Each time the scripts are run they add a new row with results for the given trial to the .csv file with the same name as the code file. 
* *Modified_MCCE.py*: Class for performing the three-step process on each of the generative models. Contains methods for post-processing and for calculating metrics, heavily inspired by the MCCE-implementation of [Redelmeier et al.](https://github.com/NorskRegnesentral/mccepy). 
* *Neural_net.py*: Class for the neural network that is used to predict the reverse process distributional parameters. 
* *Sampler.py*: Classes that contain methods for sampling from the diffusion models after their parameters are estimated. 
* *Trainer.py*: Classes that contain methods for training the diffusion models. 
* *find_factuals_??.py*: Code for training CatBoost classifiers, making predictions and randomly selecting negatively predicted test observations (factuals).
* *make_tables.py*: Code for generating LaTeX tables. All the tables with results in the thesis can be reconstructed with this code (after having produced the necessary results during experimentation).
* *prediction_model_utils.py*: Utility functions for evaluating prediction models, e.g. calculating confusion matrices, $F_1$ scores, AUC, etc. 
* *train_diffusion.py*: Code for training any of our diffusion model-variants for each dataset. It is also used to sample from the trained models. 
* *utils.py*: Utility functions for implementing the diffusion models in PyTorch. 


# Disclaimer
We have tried to document the code as clearly as possible, but this has not been a priority during research and development. Thus, errors or inaccuracies in documentation of classes, methods, etc, are to be expected. We hope the code is not overly messy. 
