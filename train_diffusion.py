# Train Gaussian_multinomial_diffusion, sample from it and save the synthetic samples to disk.
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

def main():
    # Load data etc here. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' device.")

    # Set seeds for reproducibility. 
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set main parameters for program. 
    diffusion_code = "Gaussian_multinomial" # Gaussian, Multinomial or Gaussian_multinomial.
    data_code = "AD"
    scale_version = "quantile"
    train = True # If the model should be trained or it already has been trained. 
    sample = True # If you want to sample from the trained model or not. 

    data_paths = {
        "AD": "splitted_data/AD/AD_",
        "CH": "splitted_data/CH/CH_",
        "DI": "splitted_data/DI/DI_",
    }

    training = pd.read_csv(data_paths[data_code]+"train.csv", index_col = 0)
    test = pd.read_csv(data_paths[data_code]+"test.csv", index_col = 0)
    valid = pd.read_csv(data_paths[data_code]+"valid.csv", index_col = 0)
    data = {"Train":training, "Test":test, "Valid":valid}

    if data_code == "AD":
        categorical_features = ["workclass","marital_status","occupation","relationship", \
                            "race","sex","native_country"]
        numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    elif data_code == "CH":
        categorical_features = ["Surname", "Geography", "Gender", "HasCrCard", "IsActiveMember"]
        numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    elif data_code == "DI":
        categorical_features = []
        numerical_features = ["num_pregnant", "plasma", "dbp", "skin", "insulin", "bmi", "pedi", "age"]

    if diffusion_code == "Gaussian":
        Data_object = Data(data, cat_features = [], num_features = numerical_features, 
                           already_splitted_data=True, scale_version=scale_version, valid = True)
    elif diffusion_code == "Multinomial":
        Data_object = Data(data, cat_features = categorical_features, num_features = []
                           , already_splitted_data=True, scale_version=scale_version, valid = True)
    elif diffusion_code == "Gaussian_multinomial":
        Data_object = Data(data, categorical_features, numerical_features, already_splitted_data=True, 
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
    T = 1000
    batch_size = 128
    num_epochs = 200
    num_mlp_blocks = 6
    mlp_block_width = 512
    dropout_p = 0.0
    schedule = "linear" # Tror det er noe feil med "cosine"!! Er helt klart noe feil med denne. Hvis ikke er den ræva!
    learning_rate = 0.0001
    early_stop_tolerance = 10
    model_is_class_cond = True
    num_output_classes = 2

    # Define neural network.
    model = Neural_net(X_train.shape[1], num_mlp_blocks, mlp_block_width, dropout_p, num_output_classes, model_is_class_cond).to(device)

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
        #trainer.plot_losses()

    # Load the models instead of training them again.   
    save_names = {
        "Gaussian": ["pytorch_models/"+data_code+"_Gaussian_diffusion_Neural_net.pth", "pytorch_models/"+data_code+"_Gaussian_diffusion.pth"],
        "Multinomial": ["pytorch_models/"+data_code+"_Multinomial_diffusion_Neural_net.pth", "pytorch_models/"+data_code+"_Multinomial_diffusion.pth"],
        "Gaussian_multinomial": ["pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Neural_net.pth", 
                                 "pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Multinomial_part.pth",
                                 "pytorch_models/"+data_code+"_Gaussian_multinomial_diffusion_Gaussian_part.pth"]
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
        sampler = Gaussian_sampler(model, Data_object, gauss_diffusion)
    elif diffusion_code == "Multinomial":
        sampler = Multinomial_sampler(model, Data_object, mult_diffusion)
    elif diffusion_code == "Gaussian_multinomial":
        sampler = Gaussian_multinomial_sampler(model, Data_object, mult_diffusion, gauss_diffusion)
    else: 
        raise ValueError("'diffusion_code' has to be either 'Gaussian', 'Multinomial' or 'Gaussian_Multinomial'.")
    
    if sample:
        sampler.sample(n = Data_object.get_original_data().shape[0]) 

    # Save the synthetic data to the harddrive. 
    sampler.save_synthetics()

if __name__ == "__main__":
    main()
