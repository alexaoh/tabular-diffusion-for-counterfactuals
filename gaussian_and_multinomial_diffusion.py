# We use both Gaussian and Multinomial diffusion at once,
# in order to model a heterogeneous tabular data set at once. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from GaussianDiffusion import GaussianDiffusion, NeuralNetModel, extract
from MultinomialDiffusion import MultinomialDiffusion, index_to_log_onehot, log_sub_exp, sliced_logsumexp
from Data import Data, CustomDataset, ToTensor

def train(X_train, y_train, X_valid, y_valid, numerical_features, categorical_feature_names, categorical_levels,
            device, T = 1000, schedule = "linear", batch_size = 4096, 
            num_epochs = 100, num_mlp_blocks = 4, mlp_block_width = 256, dropout_p = 0.4):
    """Function for the main training loop of the gaussian_and_multinomial diffusion model."""
    input_size = X_train.shape[1] # Columns in the training data is the input size of the neural network model. 
    num_numerical_features = len(numerical_features)
    num_categorical_features = len(categorical_feature_names)

    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor())         
    valid_data = CustomDataset(X_valid, y_valid, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_data, batch_size = X_valid.shape[0], num_workers = 2) # We want to validate on the entire validation set in each epoch

    # Define diffusion objects.
    gauss_diffusion = GaussianDiffusion(numerical_features, T, schedule, device) 
    mult_diffusion = MultinomialDiffusion(categorical_feature_names, categorical_levels, T, schedule, device) 

    # Define model for predicting noise (Gaussian diffusion) and probability parameter (Multinomial diffusion)
    model = NeuralNetModel(input_size, num_mlp_blocks, mlp_block_width, dropout_p).to(device)
    summary(model) # Plot the summary from torchinfo.

    # Define the optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Main training loop.
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    gaussian_losses = np.zeros(num_epochs)
    multinomial_losses = np.zeros(num_epochs)

    min_valid_loss = np.inf # For early stopping and saving model. 
    count_without_improving = 0 # For early stopping.

    for epoch in range(num_epochs):

        # Set PyTorch objects to training mode. Not necessary for all configurations, but good practice. 
        model.train()
        gauss_diffusion.train() 
        mult_diffusion.train()

        train_loss = 0.0
        gaussian_loss = 0.0
        multinomial_loss = 0.0
        for i, (inputs,_) in enumerate(train_loader):
            # Load data onto device.
            inputs = inputs.to(device)

            # Split the input data into numerical and categorical covariates. 
            gauss_inputs = inputs[:,:num_numerical_features]
            mult_inputs = inputs[:, num_numerical_features:]
            log_mult_inputs = torch.log(mult_inputs.float().clamp(min=1e-40)) # We effectively change all zeros to a very small number using "clamp", to avoid log(0).

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t, pt = mult_diffusion.sample_timesteps(inputs.shape[0])
            t = t.to(device)
            pt = pt.to(device)
            # TRY WITHOUT THE PRIOR AND THE pt IN THE LOSS FOR mult_diffusion LATER!
            
            # Noise the numerical inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
            x_t_gauss, noise_gauss = gauss_diffusion.noise_data_point(gauss_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Noise the categorical inputs.
            log_x_t_mult = mult_diffusion.forward_sample(log_mult_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Concatenate the inputs for the neural network. 
            x_t = torch.cat((x_t_gauss, log_x_t_mult), dim = 1)

            # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
            predictions = model(x_t, t)

            predicted_noise = predictions[:, :num_numerical_features]
            predicted_log_probs = predictions[:, num_numerical_features:]

            # Gaussian diffusion uses MSE loss. 
            gauss_loss = gauss_diffusion.loss(noise_gauss, predicted_noise)

            # Multinomial diffusion uses KL for discrete quantities. 
            mult_loss = mult_diffusion.loss(log_mult_inputs, log_x_t_mult, predicted_log_probs, t, pt) / num_categorical_features 

            # Calculate total loss. Downweigh the multinomial diffusion loss by the number of categorical features. 
            loss = gauss_loss + mult_loss 
            # Not sure why the Gaussian diffusion part is not learning almost anything?! (multinomial works fine it seems like).
            # Try to find the error tomorrow!

            optimizer.zero_grad()
            loss.backward() # Calculate gradients. 
            optimizer.step() # Update parameters. 
            train_loss += loss.item() # Calculate total training loss over the entire epoch.
            gaussian_loss += gauss_loss.item()
            multinomial_loss += mult_loss.item()

        train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.  

        gaussian_loss = gaussian_loss / (i+1)
        multinomial_loss = multinomial_loss / (i+1)
        
        ######################### Validation.
        # Set PyTorch objects to evaluation mode. Not necessary for all configurations, but good practice.
        model.eval()
        gauss_diffusion.eval() 
        mult_diffusion.eval()

        valid_loss = 0.0
        for i, (inputs,_) in enumerate(valid_loader):
            # Load data onto device.
            inputs = inputs.to(device)

            # Split the input data into numerical and categorical covariates. 
            gauss_inputs = inputs[:,:num_numerical_features]
            mult_inputs = inputs[:, num_numerical_features:]
            log_mult_inputs = torch.log(mult_inputs.float().clamp(min=1e-40)) # We effectively change all zeros to a very small number using "clamp", to avoid log(0).

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t, pt = mult_diffusion.sample_timesteps(inputs.shape[0])
            t = t.to(device)
            pt = pt.to(device)
            # TRY WITHOUT THE PRIOR AND THE pt IN THE LOSS FOR mult_diffusion LATER!
            
            # Noise the numerical inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
            x_t_gauss, noise_gauss = gauss_diffusion.noise_data_point(gauss_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Noise the categorical inputs.
            log_x_t_mult = mult_diffusion.forward_sample(log_mult_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Concatenate the inputs for the neural network. 
            x_t = torch.cat((x_t_gauss, log_x_t_mult), dim = 1)

            # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
            predictions = model(x_t, t)

            predicted_noise = predictions[:, :num_numerical_features]
            predicted_log_probs = predictions[:, num_numerical_features:]

            # Gaussian diffusion uses MSE loss. 
            gauss_loss = gauss_diffusion.loss(noise_gauss, predicted_noise)

            # Multinomial diffusion uses KL for discrete quantities. 
            mult_loss = mult_diffusion.loss(log_mult_inputs, log_x_t_mult, predicted_log_probs, t, pt) / num_categorical_features 

            # Calculate total loss. Downweigh the multinomial diffusion loss by the number of categorical features. 
            loss = gauss_loss + mult_loss 
            valid_loss += loss # Calculate the sum of validation loss over the entire epoch.
        #########################         

        training_losses[epoch] = train_loss
        validation_losses[epoch] = valid_loss
        gaussian_losses[epoch] = gaussian_loss
        multinomial_losses[epoch] = multinomial_loss
        # We do not divide the validation loss by the number of validation batches, since we validate on the entire validation set at once. 
        
        print(f"Epoch {epoch+1}: GLoss: {gaussian_loss:.4f}, MLoss: {multinomial_loss:.4f}, Total: {train_loss:.4f}. VLoss: {valid_loss:.4f}.")
        
        # Saving models each time the validation loss reaches a new minimum.
        if min_valid_loss > valid_loss:
            print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
            
            min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

            # Saving the new "best" models.             
            torch.save(gauss_diffusion.state_dict(), "./GaussianDiffusionBoth.pth")
            torch.save(mult_diffusion.state_dict(), "./MultinomialDiffusionBoth.pth")
            torch.save(model.state_dict(), "./NeuralNetBoth.pth")
            count_without_improving = 0
        else:
            count_without_improving += 1

        # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
        if count_without_improving >= 10:
            return training_losses, validation_losses, gaussian_losses, multinomial_losses
        
    return training_losses, validation_losses, gaussian_losses, multinomial_losses

def sample(n, model, gaussian_diffusion, multinomial_diffusion):
    """Common function for sampling from Gaussian and Multinomial diffusion using the neural net prediction model."""
    print("Entered function for sampling from Gaussian and Multinomial diffusion.")
    model.eval()
    gaussian_diffusion.eval()
    multinomial_diffusion.eval()
    x_list = {}

    num_numerical_features = len(gaussian_diffusion.numerical_features)
    num_categorical_features = multinomial_diffusion.num_categorical_variables
    num_categorical_onehot_encoded_columns = len(multinomial_diffusion.num_classes_extended)

    device = gaussian_diffusion.device

    with torch.no_grad():
        x_gauss = torch.randn((n,num_numerical_features)).to(device) # Sample from standard Gaussian (sample from x_T). 
        uniform_sample = torch.zeros((n, num_categorical_onehot_encoded_columns), device=device) # I think this could be whatever number, as long as all of them are equal!  
        log_x_mult = multinomial_diffusion.log_sample_categorical(uniform_sample).to(device) # The sample at T is uniform (sample from x_T).
        for i in reversed(range(gaussian_diffusion.T)): # I start it at 0.
            if i % 25 == 0:
                print(f"Sampling step {i}.")
            x = torch.cat((x_gauss, log_x_mult), dim = 1)
            #x_list[i] = x_gauss # Don't really need these after development is over. 
            t = (torch.ones(n) * i).to(torch.int64).to(device)
            predictions = model(x,t)

            # Gaussian part. 
            predicted_noise_gauss = predictions[:,:num_numerical_features]
            sh = predicted_noise_gauss.shape

            betas = extract(gaussian_diffusion.betas, t, sh) 
            sqrt_recip_alpha = extract(gaussian_diffusion.sqrt_recip_alpha, t, sh)
            sqrt_recip_one_minus_alpha_bar = extract(gaussian_diffusion.sqrt_recip_one_minus_alpha_bar, t, sh) 
            
            # Version #2 of sigma.
            sigma = extract(torch.sqrt(gaussian_diffusion.beta_tilde), t, sh)

            if i > 0:
                noise = torch.randn_like(x_gauss)
            else: # We don't want to add noise at t = 0, because it would make our outcome worse (this comes from the fact that we have another term in Loss for x_0|x_1, I believe).
                noise = torch.zeros_like(x_gauss)
            x_gauss = sqrt_recip_alpha * (x_gauss - (betas * sqrt_recip_one_minus_alpha_bar)*predicted_noise_gauss) + sigma * noise # Use formula in line 4 in Algorithm 2.

            # Multinomial part. 
            predicted_log_x_mult = predictions[:,num_numerical_features:]
            log_tilde_theta_hat = multinomial_diffusion.reverse_pred(predicted_log_x_mult, log_x_mult, t) # Get reverse process probability parameter. 
            log_x_mult = multinomial_diffusion.log_sample_categorical(log_tilde_theta_hat) # Sample from a categorical distribution based on theta_post. 

        # Get final x (x_0, i.e. the sample we generated).
        x_mult = torch.exp(log_x_mult)
        x = torch.cat((x_gauss, x_mult), dim = 1)

    model.train() # Indicate to Pytorch that we are back to doing training. 
    gaussian_diffusion.train()
    multinomial_diffusion.train()
    return x, x_list # We return a list of the x'es to see how they develop from t = 99 to t = 0 (just while developing the models).
    
def find_levels(df, categorical_features):
        """Returns a list of levels of features of each of the categorical features."""
        lens_categorical_features = []
        for feat in categorical_features:
            unq = len(df[feat].value_counts().keys().unique())
            print(f"Feature '{feat}'' has {unq} unique levels")
            lens_categorical_features.append(unq)
        print(f"The sum of all levels is {sum(lens_categorical_features)}. This will be the number of cat-columns after one-hot encoding (non-full rank)")
        return(lens_categorical_features)

def plot_losses(training_losses, validation_losses, label1 = "Training", label2 = "Validation"):
        print(f"Length of non-zero training losses: {len(training_losses[training_losses != 0])}")
        print(f"Length of non-zero validation losses: {len(validation_losses[validation_losses != 0])}")
        plt.plot(training_losses[training_losses != 0], label = label1)
        plt.plot(validation_losses[validation_losses != 0], label = label2)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.legend()
        #plt.show()

def count_parameters(model):
    """Function for counting how many parameters require optimization."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(n, input_size, numerical_features, categorical_feature_names, categorical_levels, device, 
             T = 1000, schedule = "linear", num_mlp_blocks = 4, mlp_block_width = 256, dropout_p = 0.0, generate = True):
    """Evalute the model (gaussiand_and_multinomial_diffusion)."""
    # Load the previously saved models.
    model = NeuralNetModel(input_size, num_mlp_blocks, mlp_block_width, dropout_p).to(device)
    gauss_diffusion = GaussianDiffusion(numerical_features, T, schedule, device)
    mult_diffusion = MultinomialDiffusion(categorical_feature_names, categorical_levels, T, schedule, device)
    
    model.load_state_dict(torch.load("./NeuralNetBoth.pth"))
    gauss_diffusion.load_state_dict(torch.load("./GaussianDiffusionBoth.pth")) 
    mult_diffusion.load_state_dict(torch.load("./MultinomialDiffusionBoth.pth")) 
    # Don't think it is necessary to save and load the diffusion model!
    # We still do it to be safe. 

    with torch.no_grad():
        model.eval()
        gauss_diffusion.eval()
        mult_diffusion.eval()    

        # Run the noise backwards through the backward process in order to generate new data. 
        if generate:
            #samples_gauss, reverse_points_list_gauss = gauss_diffusion.sample(model, n) # This works now!
            #samples_mult, reverse_points_list_mult = mult_diffusion.sample(model, n) # This does not quite work, since the dimensions of model and data do not match. 
            # We make a combined sampling loop for both the diffusion models instead. 
            synthetic_samples, reverse_points_list = sample(n, model, gauss_diffusion, mult_diffusion)
            synthetic_samples = synthetic_samples.cpu()
            synthetic_samples = pd.DataFrame(synthetic_samples, columns = X_train.columns.tolist())
            synthetic_samples.to_csv("synthetic_sample_both.csv")
        else:
            # Load the synthetic sample we already created. 
            synthetic_samples = pd.read_csv("synthetic_sample_both.csv", index_col = 0)
            reverse_points_list = {}

    return synthetic_samples, reverse_points_list

def plot_correlation():
    """Plot correlation between numerical data."""
    pass

def plot_numerical_features(synthetic, real, numerical_features):
    """Plot the numerical features are histograms."""
    synthetic_data = synthetic.loc[:,numerical_features]
    real_data = real.loc[:,numerical_features]
    fig, axs = plt.subplots(3,2)
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.hist(synthetic_data.iloc[:,idx], density = True, color = "b", label = "Synth.", bins = 50)
        ax.hist(real_data.iloc[:,idx], color = "orange", alpha = 0.6, density = True, label = "OG.", bins = 50)
        ax.legend()
        ax.title.set_text(real_data.columns.tolist()[idx])
    plt.tight_layout()

def plot_categorical_features(synthetic_data, real_data, categorical_features):
    """Plot the categorical features are barplots."""
    fig, axs = plt.subplots(2,2)
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        (synthetic_data[categorical_features[idx]].value_counts()/synthetic_data.shape[0]*100).plot(kind='bar', ax = ax, label = "Synth.")
        (real_data[categorical_features[idx]].value_counts()/real_data.shape[0]*100).plot(kind='bar', ax = ax, color = "orange", alpha = 0.6, label = "OG.")
        ax.xaxis.set_ticklabels([])
        ax.legend()
        ax.title.set_text(f"% {categorical_features[idx]}")
        
    plt.tight_layout()

    # Make two grids since 7 is not an even number of categorical features. 
    fig, axs2 = plt.subplots(2,2)
    axs2 = axs2.ravel()
    for idx, ax in enumerate(axs2, start = 4):
        if idx > len(categorical_features)-1:
            break
        (synthetic_data[categorical_features[idx]].value_counts()/synthetic_data.shape[0]*100).plot(kind='bar', ax = ax, label = "Synth.")
        (real_data[categorical_features[idx]].value_counts()/real_data.shape[0]*100).plot(kind='bar', ax = ax, color = "orange", alpha = 0.6, label = "OG.")
        ax.xaxis.set_ticklabels([])
        ax.legend()
        ax.title.set_text(f"% {categorical_features[idx]}")
    plt.tight_layout()

if __name__ == "__main__":
    # Run functions I only want to run when running precisely this file (and not importing into another file) here.

    # Load data etc here. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' device.")

    # Set seeds for reproducibility. 
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    adult_data = pd.read_csv("adult_data_no_NA.csv", index_col = 0)
    print(adult_data.shape)
    categorical_features = ["workclass","marital_status","occupation","relationship", \
                            "race","sex","native_country"]
    numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

    Adult = Data(adult_data, categorical_features, numerical_features, scale_version = "quantile", splits = [0.85,0.15])
    X_train, y_train = Adult.get_training_data_preprocessed()
    X_test, y_test = Adult.get_test_data_preprocessed()
    print(X_train.shape)

    lens_categorical_features = Adult.lens_categorical_features
    print(lens_categorical_features)

    # X_train = X_train.iloc[[0]] # Follow one sample through the process to see how it works and try to understand why it does not work well. 
    # X_test = X_test.iloc[[0]]

    # Hyperparameters.
    T = 1000
    batch_size = 128
    num_epochs = 150
    num_mlp_blocks = 6
    mlp_block_width = 526
    dropout_p = 0.0
    schedule = "linear"

    # training_losses, validation_losses, gaussian_loss, multinomial_loss = train(X_train, y_train, X_test, y_test, 
    #                         numerical_features, categorical_features, 
    #                         lens_categorical_features, device, T = T, 
    #                         schedule = schedule, batch_size = batch_size, num_epochs = num_epochs, 
    #                         num_mlp_blocks = num_mlp_blocks, mlp_block_width = mlp_block_width, dropout_p = dropout_p)

    # plot_losses(training_losses, validation_losses)
    # plot_losses(gaussian_loss, multinomial_loss, label1="Gaussian Loss", label2="Multinomial Loss")
    # print(f"Gaussian training losses: {gaussian_loss}")
    # print(f"Multinomial training losses: {multinomial_loss}")
    # plt.show()

    synthetic_samples, reverse_points_list = evaluate(n = X_train.shape[0], input_size = X_train.shape[1], numerical_features=numerical_features, 
                                categorical_feature_names=categorical_features, categorical_levels=lens_categorical_features,
                                device=device, T=T, schedule=schedule, dropout_p=dropout_p,
                                num_mlp_blocks=num_mlp_blocks, mlp_block_width=mlp_block_width, generate=False)
    
    print(synthetic_samples.head())

    # Descale and decode the synthetic data.
    #synthetic_samples = Adult.descale(synthetic_samples)
    #synthetic_samples = Adult.decode(synthetic_samples)

    # Change datatypes to match the X_train datatypes.
    synthetic_samples[numerical_features] = synthetic_samples[numerical_features].astype("int64")

    # Save the data to disk after decoding and descaling. 
    #synthetic_samples.to_csv("synthetic_sample_both.csv")

    # Get original training data we want to compare to.
    X_train = Adult.get_training_data()[0]

    print(synthetic_samples.head())

    # Something must have gone wrong in the Gaussian diffusion part during training or sampling?!
    plot_numerical_features(synthetic_samples, X_train, numerical_features)
    plt.show()

    plot_categorical_features(synthetic_samples, X_train, categorical_features)
    plt.show()

    def look_at_reverse_process_steps(x_list, T):
        """We plot some of the steps in the backward process to visualize how it changes the data."""
        times = [0, int(T/5), int(2*T/5), int(3*T/5), int(4*T/5), T-1][::-1] # Reversed list of the same times as visualizing forward process. 
        for i, feat in enumerate(numerical_features):
            fig, axs = plt.subplots(2,3)
            axs = axs.ravel()
            for idx, ax in enumerate(axs):
                ax.hist(x_list[times[idx]][:,i], density = True, color = "b", bins = 100) 
                ax.set_xlabel(f"Time {times[idx]}")
                #ax.set_xlim(np.quantile(x_list[times[idx]][:,i], 0.05), np.quantile(x_list[times[idx]][:,i], 0.95))
            fig.suptitle(f"Feature '{feat}'")
            plt.tight_layout()
        plt.show()
            
    #look_at_reverse_process_steps(reverse_points_list, T)
