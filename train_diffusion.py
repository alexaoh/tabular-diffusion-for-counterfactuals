# Train Gaussian_multinomial_diffusion, sample from it and save the synthetic samples to disk.
import argparse
import pandas as pd
import numpy as np
import torch
from torchinfo import summary

from Data import Data, CustomDataset, ToTensor
from Trainer import Gaussian_trainer, Multinomial_trainer, Gaussian_multinomial_trainer
from Sampler import Gaussian_sampler, Multinomial_sampler, Gaussian_multinomial_sampler

from Gaussian_diffusion import Gaussian_diffusion
from Multinomial_diffusion import Multinomial_diffusion
from Neural_net import Neural_net

def take_args():
    """Take args from command line."""
    parser = argparse.ArgumentParser(prog = "train_diffusion.py", 
                                     description = "Train a diffusion model and generate synthetic data from it.")
    parser.add_argument("-s", "--seed", help="Seed for random number generators. Default is 1234.", 
                        type=int, default = 1234, required = False)
    parser.add_argument("-d", "--data-code", 
                        help = "Give the desired data code ('AD', 'CH' or 'DI'). 'AD' is default.", 
                        default = "AD", required = False)
    parser.add_argument("-t", "--train", 
                        help = "Train the diffusion model or use pre-trained model. Default is 'False' (bool).", 
                        default = False, type = bool, required = False)
    parser.add_argument("-g", "--generate", 
                        help = "Generate synthetic data from the diffusion model. Default is 'True' (bool).",
                        default = True, type = bool, required = False)
    parser.add_argument("-p", "--plot-losses", 
                        help = "Plot losses after training. Default is 'False' (bool).", 
                        default = False, type = bool, required = False)
    parser.add_argument("--savename", 
                        help = "Name for saving synthetic samples. Default depends on 'data-code'.",
                        required = False)
    parser.add_argument("--num-samples", 
                        help = "Number of samples to generate. Default is the number of observations in the real dataset.",
                        required = False)
    
    # Hyperparameters.
    hyperparams = parser.add_argument_group("Hyperparameters")
    hyperparams.add_argument("-T", 
                             help = "Number of diffusion steps. Default is 1000.",
                             default = 1000, type = int, required = False)
    hyperparams.add_argument("-b", "--batch-size", 
                             help = "Batch size. Default is 128.",
                             type = int, default = 128, required = False)
    hyperparams.add_argument("-e", "--epochs", 
                             help = "Number of epochs. Default is 200.",
                             type = int, default = 200, required = False)
    hyperparams.add_argument("--mlp-blocks", 
                             help = "MLPBlocks (hidden layers). Default is [512, 512, 512, 512].",
                             nargs = "*", type = int, default = [512, 512, 512, 512])
    hyperparams.add_argument("--dropout-ps", 
                             help = "Dropout probabilities for hidden layers. Default is [0, 0, 0, 0].",
                             nargs = "*", type = int, default = [0, 0, 0, 0])
    hyperparams.add_argument("--schedule", 
                             help = "Variance schedule ('linear' or 'cosine'). Default is 'linear'.",
                             default = "linear", required = False)
    hyperparams.add_argument("--early-stop-tolerance", 
                             help = "Early stop tolerance. Default is 10. Setting to 'None' turns off early stopping.",
                             default = 10, required = False)
    hyperparams.add_argument("--learning-rate", 
                             help = "Learning rate for Adam optimizer. Default is 0.0001.",
                             type = float, default = 0.0001, required = False)
    
    args = parser.parse_args()
    return args

def main(args):
    # Load data etc here. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' device.")

    # Set seeds for reproducibility. 
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set main parameters for program. 
    data_code = args.data_code
    if data_code in ["AD", "CH"]:
        diffusion_code = "Gaussian_multinomial" # Gaussian, Multinomial or Gaussian_multinomial.
    elif data_code == "DI":
        diffusion_code = "Gaussian"
    else:
        raise ValueError(f"'data_code' {data_code} is not valid. Needs to be 'AD', 'CH' or 'DI' (for now).")
    scale_version = "quantile"
    train = args.train # If the model should be trained or it already has been trained. 
    sample = args.generate # If you want to sample from the trained model or not. 

    data_paths = {
        "AD": "splitted_data/AD/AD_",
        "CH": "splitted_data/CH/CH_",
        "DI": "splitted_data/DI/DI_",
    }

    # Load the data as csv. Could have loaded as pickle as well. 
    training = pd.read_csv(data_paths[data_code]+"train.csv", index_col = 0)
    test = pd.read_csv(data_paths[data_code]+"test.csv", index_col = 0)
    valid = pd.read_csv(data_paths[data_code]+"valid.csv", index_col = 0)
    data = {"Train":training, "Test":test, "Valid":valid}

    if data_code == "AD":
        categorical_features = ["workclass","marital_status","occupation","relationship", \
                            "race","sex","native_country"]
        numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    elif data_code == "CH":
        categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
        numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    elif data_code == "DI":
        categorical_features = []
        numerical_features = ["num_pregnant", "plasma", "dbp", "skin", "insulin", "bmi", "pedi", "age"]

    if diffusion_code == "Gaussian":
        Data_object = Data(data, cat_features = [], num_features = numerical_features, 
                           seed = args.seed, already_splitted_data=True, scale_version=scale_version, valid = True)
    elif diffusion_code == "Multinomial":
        Data_object = Data(data, cat_features = categorical_features, num_features = [], 
                           seed = args.seed, already_splitted_data=True, scale_version=scale_version, valid = True)
    elif diffusion_code == "Gaussian_multinomial":
        Data_object = Data(data, categorical_features, numerical_features, 
                           seed = args.seed, already_splitted_data=True, 
                           scale_version=scale_version, valid = True)
    else:
        raise ValueError("'diffusion_code' has to be either 'Gaussian', 'Multinomial' or 'Gaussian_Multinomial'.")

    
    X_train, y_train = Data_object.get_training_data_preprocessed()
    X_test, y_test = Data_object.get_test_data_preprocessed()
    X_valid, y_valid = Data_object.get_validation_data_preprocessed()
    print(f"X_train shape:{X_train.shape}")
    print(f"X_test shape:{X_test.shape}")
    print(f"X_valid shape:{X_valid.shape}")

    # Use validation and testing data as validation while training, since we do not need to leave out any testing data for after training. 
    X_valid = pd.concat((X_test, X_valid))
    y_valid = pd.concat((y_test, y_valid))
    print(f"X_valid shape after concat of valid and test:{X_valid.shape}")

    lens_categorical_features = Data_object.lens_categorical_features
    print(f"Levels of categorical features: {lens_categorical_features}")

    # Set hyperparameters.
    T = args.T
    batch_size = args.batch_size
    num_epochs = args.epochs

    mlp_blocks = args.mlp_blocks
    dropout_ps = args.dropout_ps
    schedule = args.schedule # Tror det er noe feil med "cosine"!! Er helt klart noe feil med denne. Hvis ikke er den ræva!
    learning_rate = args.learning_rate
    early_stop_tolerance = None if args.early_stop_tolerance == 'None' else args.early_stop_tolerance
    model_is_class_cond = True
    num_output_classes = 2

    # Define neural network.
    model = Neural_net(X_train.shape[1], mlp_blocks, dropout_ps, num_output_classes, model_is_class_cond, seed).to(device)
    summary(model) # Plot the summary from torchinfo.

    # Define Trainer object.
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid, 
        "y_valid": y_valid
    }
    if diffusion_code == "Gaussian":
        gauss_diffusion = Gaussian_diffusion(numerical_features, T, schedule, device) 
        trainer = Gaussian_trainer(data, model, gauss_diffusion, num_epochs, batch_size, learning_rate, early_stop_tolerance, data_code)
    elif diffusion_code == "Multinomial":
        mult_diffusion = Multinomial_diffusion(categorical_features, lens_categorical_features, T, schedule, device) 
        trainer = Multinomial_trainer(data, model, mult_diffusion, num_epochs, batch_size, learning_rate, early_stop_tolerance, data_code)
    elif diffusion_code == "Gaussian_multinomial":
        gauss_diffusion = Gaussian_diffusion(numerical_features, T, schedule, device) 
        mult_diffusion = Multinomial_diffusion(categorical_features, lens_categorical_features, T, schedule, device) 
        trainer = Gaussian_multinomial_trainer(data, model, mult_diffusion, gauss_diffusion, num_epochs, batch_size, 
                                               learning_rate, early_stop_tolerance, data_code)
    else:
        raise ValueError("'diffusion_code' has to be either 'Gaussian', 'Multinomial' or 'Gaussian_Multinomial'.")

    # Train the model. 
    if train:
        trainer.train()
        if args.plot_losses:
            trainer.plot_losses()

    # Load the models instead of training them again.   
    save_names = {
        "Gaussian": ["pytorch_models/"+data_code+"_Gaussian_diffusion_Neural_net"+str(seed)+".pth", 
                     "pytorch_models/"+data_code+"_Gaussian_diffusion"+str(seed)+".pth"],
        "Multinomial": ["pytorch_models/"+data_code+"_Multinomial_diffusion_Neural_net"+str(seed)+".pth", 
                        "pytorch_models/"+data_code+"_Multinomial_diffusion"+str(seed)+".pth"],
        "Gaussian_multinomial": ["pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Neural_net"+str(seed)+".pth", 
                                 "pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Multinomial_part"+str(seed)+".pth",
                                 "pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Gaussian_part"+str(seed)+".pth"]
    }  

    model.load_state_dict(torch.load(save_names[diffusion_code][0]))
    if diffusion_code == "Gaussian":
        gauss_diffusion.load_state_dict(torch.load(save_names[diffusion_code][1]))
    elif diffusion_code == "Multinomial":
        mult_diffusion.load_state_dict(torch.load(save_names[diffusion_code][1]))
    elif diffusion_code == "Gaussian_multinomial":
        mult_diffusion.load_state_dict(torch.load(save_names[diffusion_code][1]))
        gauss_diffusion.load_state_dict(torch.load(save_names[diffusion_code][2]))
    else: 
        raise ValueError("'diffusion_code' has to be either 'Gaussian', 'Multinomial' or 'Gaussian_Multinomial'.")
    
    # Sample from the model.
    if diffusion_code == "Gaussian":
        sampler = Gaussian_sampler(model, Data_object, gauss_diffusion, data_code)
    elif diffusion_code == "Multinomial":
        sampler = Multinomial_sampler(model, Data_object, mult_diffusion, data_code)
    elif diffusion_code == "Gaussian_multinomial":
        sampler = Gaussian_multinomial_sampler(model, Data_object, mult_diffusion, gauss_diffusion, data_code)
    else: 
        raise ValueError("'diffusion_code' has to be either 'Gaussian', 'Multinomial' or 'Gaussian_Multinomial'.")
    
    if sample:
        if args.num_samples is None:
            sampler.sample(n = Data_object.get_original_data().shape[0]) 
        else: 
            sampler.sample(n = int(args.num_samples))

        # Save the synthetic data to the harddrive. 
        if args.savename is None:
            sampler.save_synthetics()
        else: 
            sampler.save_synthetics(savename = args.savename)

if __name__ == "__main__":
    args = take_args()
    main(args = args)
