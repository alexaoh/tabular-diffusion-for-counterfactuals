# Train Gaussian_multinomial_diffusion, sample from it and save the synthetic samples to disk.
import pandas as pd
import numpy as np
import torch
from torchinfo import summary

from Data import Data, CustomDataset, ToTensor
from Trainer import Trainer, Gaussian_trainer, Multinomial_trainer, Gaussian_multinomial_trainer

from Gaussian_diffusion import Gaussian_diffusion
from Multinomial_diffusion import Multinomial_diffusion
from Gaussian_multinomial_diffusion import Gaussian_multinomial_diffusion # We want to feed the two others to this class eventually!
                                                                        # Perhaps we don't need this one after all!
                                                                        # Make a class for sampling instead!
from Neural_net import Neural_net


def main(dataset = "AD"):
    # Load data etc here. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' device.")

    # Set seeds for reproducibility. 
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    data_paths = {
        "AD": "splitted_data/AD/AD_"
    }

    training = pd.read_csv(data_paths["AD"]+"train.csv", index_col = 0)
    test = pd.read_csv(data_paths["AD"]+"test.csv", index_col = 0)
    valid = pd.read_csv(data_paths["AD"]+"valid.csv", index_col = 0)
    data = {"Train":training, "Test":test, "Valid":valid}

    categorical_features = ["workclass","marital_status","occupation","relationship", \
                            "race","sex","native_country"]
    numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

    Data_object = Data(data, categorical_features, numerical_features, already_splitted_data=True, scale_version="quantile", valid = True)
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
    num_epochs = 150
    num_mlp_blocks = 6
    mlp_block_width = 526
    dropout_p = 0.0
    schedule = "linear"
    learning_rate = 0.0001

    # Define neural network and diffusion objects.
    gauss_diffusion = Gaussian_diffusion(numerical_features, T, schedule, device) 
    mult_diffusion = Multinomial_diffusion(categorical_features, lens_categorical_features, T, schedule, device) 
    model = Neural_net(X_train.shape[1], num_mlp_blocks, mlp_block_width, dropout_p, num_output_classes = 2, is_class_cond = False).to(device)

    summary(model) # Plot the summary from torchinfo.

    # Define Trainer object.
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid, 
        "y_valid": y_valid
    }
    trainer = Gaussian_multinomial_trainer(data, model, mult_diffusion, gauss_diffusion, num_epochs, batch_size, learning_rate, early_stop_tolerance = 10)

    # Train the model. 
    trainer.train()
    trainer.plot_losses()

    # Sample from the model.

    # Perhaps make a class for sampling as well, just like the trainer! 
    # I think this might be the smoothest way of doing it!

    # Save the synthetic data to the harddrive. 

if __name__ == "__main__":
    main(dataset = "AD")
