# This is my first iteration of tabDDPM
# DRAFT, very rough.

import torch
import torch.nn as nn
import numpy as np
from functools import partial
import math
from torchinfo import summary

def extract(a, t, x_shape):
    """Changes the dimensions of the input a depending on t and x_t.

    Makes them compatible such that pointwise multiplication of tensors can be done.
    """
    t = t.to(a.device) # Change the device to make sure they are compatible. 
    out = a[t] # Get the correct time steps from a. 
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape) # Expand such that all elements are correctly multiplied by the noise. 

class GaussianDiffusion(nn.Module):
    """Class for Gaussian diffusion, for all the numerical features in the data set."""

    def __init__(self, numerical_features, T, schedule_type, device):
        super(GaussianDiffusion, self).__init__()
        self.numerical_features = numerical_features
        self.T = T
        self.schedule_type = schedule_type
        self.device = device
        
        # We hardcode the first and last beta in the linear schedule for now (for testing).
        self.beta_start = 1e-4
        self.beta_end = 0.02

        # Prepare the noise schedule (either linear or cosine).
        betas = self.prepare_noise_schedule()

        # Calculate necessary quantities for formulas in papers.
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas, axis = 0)
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        one_minus_alpha_bar = 1 - alpha_bar
        sqrt_one_minus_alpha_bar = np.sqrt(1-alpha_bar)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
        alpha_bar_next = np.append(alpha_bar[1:], 0.0)

        beta_tilde = betas * (1.0 - alpha_bar_prev)/(1.0 - alpha_bar) # Equation 7 in DDPM.
        mu_tilde_coef1 = np.sqrt(alpha_bar_prev)*betas / (1.0 - alpha_bar) # Equation 7 in DDPM. 
        mu_tilde_coef2 = np.sqrt(alphas)*(1.0 - alpha_bar_prev) / (1.0 - alpha_bar) # Equation 7 in DDPM. 
        
        # Make partial function to make Pytorch tensors with dtype float32 when registering the buffers of each variable.
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Parameters for "many operations".
        self.register_buffer("betas", to_torch(betas).to(self.device))
        self.register_buffer("alphas", to_torch(alphas).to(self.device))
        self.register_buffer("alpha_bar", to_torch(alpha_bar).to(self.device))
        self.register_buffer("sqrt_alpha_bar", to_torch(sqrt_alpha_bar).to(self.device))
        self.register_buffer("one_minus_alpha_bar", to_torch(one_minus_alpha_bar).to(self.device))
        self.register_buffer("sqrt_one_minus_alpha_bar", to_torch(sqrt_one_minus_alpha_bar).to(self.device))
        self.register_buffer("alpha_bar_prev", to_torch(alpha_bar_prev).to(self.device))
        self.register_buffer("alpha_bar_next", to_torch(alpha_bar_next).to(self.device))

        # Parameters for forward posterior. 
        self.register_buffer("beta_tilde", to_torch(beta_tilde).to(self.device))
        self.register_buffer("mu_tilde_coef1", to_torch(mu_tilde_coef1).to(self.device))
        self.register_buffer("mu_tilde_coef2", to_torch(mu_tilde_coef2).to(self.device))
        
    def noise_data_point(self, x_0, t):
        """Get x_t (noised input x_0 at times t), following the closed form Equation 4 in DDPM by Ho et. al."""
        noise = torch.randn_like(x_0)
        assert x_0.shape == noise.shape
        return extract(self.sqrt_alpha_bar, t, x_0.shape)*x_0 \
                + extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)*noise, noise
        # This is precisely why the use the "extract" function in tabDDPM I think!
        # UNDERSTAND HOW THIS WORKS. 
        # Shapes are wrong when dealing with tensors! 
        # sqrt_alpha_bar has shape (batch_size x 1) (vector)
        # x_0 and noise have shape (batch_size x input_size) (tensor)
        # We want to return tensor of shape (batch_size x input_size). 

    def sample(self, model, n):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper.
        'model' is the neural network that is used to predict the noise in each time step. 
        """
        print("Entered function for sampling.")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,model.input_size)).to(self.device) # Sample from standard Gaussian (sample from x_T). We are not doing it for images, but for data points. This is only 1D for now. 
            # For tabular data I probably need (n,2) I think! Test later. 
            for i in reversed(range(self.T)): # I start it at 0
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                predicted_noise = model(x,t)
                alphas = extract(self.alphas, t, predicted_noise.shape) # Perhaps need to use extract(self.alphas, t, predicted_noise.shape) here?
                alpha_bar = extract(self.alpha_bar, t, predicted_noise.shape) #[:, None, None, None] # I think this is for the images perhaps. 
                betas = extract(self.betas, t, predicted_noise.shape) #[:, None, None, None] # I think this is for the images perhaps. 
                if i > 0:
                    noise = torch.randn_like(x)
                else: # We don't want to add noise at t = 0, because it would make our outcome worse (this comes from the fact that we have another term in Loss for x_0|x_1, I believe).
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1.0-alpha_bar)))*predicted_noise) + torch.sqrt(betas) * noise # Use formula in line 4 in Algorithm 2.
                                                                                            # Here we have defined sigma^2 = beta, which was one of the options the authors tested. 
                # Kan nok sikkert bruke flere av de variablene jeg definerte som buffere her i stedet for å beregne reciprocals og sqrt, etc på nytt her. 
        model.train() # Indicate to Pytorch that we are back to doing training. 
        return x

    def prepare_noise_schedule(self):
        """Prepare the betas in the variance schedule."""
        if self.schedule_type == "linear":
            # Linear schedule from Ho et. al, extended to work for any number of diffusion steps. 
            scale = 1000/self.T
            beta_start = scale * self.beta_start
            beta_end = scale * self.beta_end
            return np.linspace(beta_start, beta_end, self.T)
        elif self.schedule_type == "cosine":
            return self.betas_for_alpha_bar(
            self.T,
            lambda t: (math.cos((t + 0.008) / 1.008 * math.pi / 2)/math.cos(0.008/1.008 * math.pi/2))** 2, 
            # I divided everything by f(0) as the formula in the paper states. 
        )
        else:
            raise NotImplementedError(f"The schedule type {self.schedule_type} has not been implemented.")

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def sample_timesteps(self, n):
        """Sample timesteps (uniformly) for use when training the model."""
        return torch.randint(low=1, high=self.T, size = (n,))

    def loss(self, real_noise, model_pred):
        """Function to return Gaussian loss, i.e. MSE."""
        return torch.mean((real_noise - model_pred)**2) # We take the mean over all dimensions. Not sure if this is correct. 

class MultinomialDiffusion():
    """Class for Multinomial diffusion. One for each categorical feature in the data set."""    

    def __init__(self, categorical_feature_name, levels):
        self.categorical_feature_name = categorical_feature_name
        self.levels = levels

class GaussianMultinomialDiffusion():
    """Class for concatenation of Gaussian and Multinomial diffusion."""

    def __init__(self, num_levels: np.array, num_numerical_features: int, T = 100, 
            schedule_name = "cosine", device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.num_levels = num_levels
        self.num_numerical_features = num_numerical_features
        self.T = T
        self.schedule_name = schedule_name
        self.device = device
        print(f"The GaussianMultinomialDiffusion object is on device {device}.")

class NeuralNetModel(nn.Module):
    """Main model for predicting Gaussian noise.
    
    We make a simple model to begin with, just to see if we are able to train something. 
    """

    def __init__(self, input_size, num_mlp_blocks, dropout_p):
        super(NeuralNetModel, self).__init__()
        self.input_size = input_size
        self.num_mlp_blocks = num_mlp_blocks
        assert self.num_mlp_blocks >= 1, ValueError("The number of MLPBlocks needs to be at least 1.")
        self.dropout_p = dropout_p
        assert dropout_p >= 0 and dropout_p <= 1, ValueError("The dropout probability must be a real number between 0 and 1.")

        # Layers.
        self.l1 = nn.Linear(128, 256) # For first MLPBlock. 
        self.linear_layers = nn.ModuleList() # MLPBlocks inbetween the first MLPBlock and the linear output layer. 
        for _ in range(self.num_mlp_blocks-1):
            self.linear_layers.append(nn.Linear(256, 256))
        self.outlayer = nn.Linear(256, input_size)
        
        # Activation functions. 
        self.relu = nn.ReLU()

        # Dropout.
        self.dropout = nn.Dropout(p = self.dropout_p) # Set some random dropout probability during training. 

        # Neural network for time embedding according to tabDDPM (Equation 5).
        self.time_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        # Neural network for embedding the feature vector to the same space as the time embedding.
        self.proj = nn.Linear(input_size, 128)
    
    def timestep_embedding(self, timesteps, dim = 128, max_period = 10000):
        """Function for constructing Sinusoidal time embeddings (positional embeddings) as in 'Attention is all you need'.
        
        THIS HAS BEEN COPIED FROM CODE IN TABDDPM FOR NOW. 
        
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        # I stole the function above directly from https://github.com/rotot0/tab-ddpm/blob/main/tab_ddpm/modules.py
        # Have a look at it later in order to understand how these sinusoidal time embeddings work. 

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, t):
        """Forward steps for Pytorch."""
        # First we make the embeddings. 
        t_emb = self.time_embed(self.timestep_embedding(t)) # Not sure if this should take one t or several at once! 
                                                            # Here I assume that it takes several (according to output from sample_timesteps).
        x_emb = self.proj(x)
        x = x_emb + t_emb # Final embedding vector (consisting of features and time).

        # Feed the embeddings to our "regular" (simple) MLP. 
        x = self.dropout(self.relu(self.l1(x))) # First MLPBlock
        for ll in self.linear_layers:
            x = self.dropout(self.relu(ll(x))) # MLPBlocks in between. 
        x = self.outlayer(x) # Linear out-layer.
        return x

def validate_one_epoch(model, diffusion, valid_loader):
    """Function to call in the training loop if we want to calculate validation loss."""
    model.eval()
    diffusion.eval()
    valid_loss = 0.0
    for i, (inputs,_) in enumerate(valid_loader):
        # Load data onto device.
        inputs = inputs.to(device)

        # We sample new times. 
        t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

        # We noise the validation inputs at the new sampled times. 
        x_t, noise = diffusion.noise_data_point(inputs, t) 

        predicted_noise = model(x_t, t) # Predict the noise of the validation data. 

        # Gaussian diffusion uses MSE loss. 
        loss = diffusion.loss(noise, predicted_noise)
        valid_loss += loss # Calculate the sum of validation loss over the entire epoch.
    return valid_loss


def train(X_train, y_train, X_valid, y_valid, numerical_features, device, T = 1000, schedule = "linear", batch_size = 4096, 
            num_epochs = 100, num_mlp_blocks = 4, dropout_p = 0.4, validate = True):
    """Function for the main training loop of the Gaussian diffusion model."""
    input_size = X_train.shape[1] # Columns in the training data is the input size of the neural network model. 

    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor())         

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    # Define Gaussian Diffusion object.
    diffusion = GaussianDiffusion(numerical_features, T, schedule, device) # Numerical_features is not used for anything now. 

    # Define model for predicting noise in each step. 
    model = NeuralNetModel(input_size, num_mlp_blocks, dropout_p).to(device)
    summary(model) # Plot the summary from torchinfo.

    # Define the optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Main training loop.
    training_losses = np.zeros(num_epochs)
    if validate: 
        valid_data = CustomDataset(X_valid, y_valid, transform = ToTensor()) 
        valid_loader = DataLoader(valid_data, batch_size = X_valid.shape[0], num_workers = 2) # We want to validate on the entire validation set in each epoch
        validation_losses = np.zeros(num_epochs)
        min_valid_loss = np.inf # For early stopping and saving model. 
        count_without_improving = 0 # For early stopping.

    for epoch in range(num_epochs):
        model.train()
        diffusion.train() # I do not think this is strictly necessary for the diffusion model. 
        train_loss = 0.0
        for i, (inputs,_) in enumerate(train_loader):
            # Load data onto device.
            inputs = inputs.to(device)

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

            # Noise the inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
            x_t, noise = diffusion.noise_data_point(inputs, t) # x_t is the noisy version of the input x, at time t.

            # Feed the noised data and the time step to the model, which then predicts the noise at that time. 
            predicted_noise = model(x_t, t)

            # Gaussian diffusion uses MSE loss. 
            loss = diffusion.loss(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward() # Calculate gradients. 
            optimizer.step() # Update parameters. 
            train_loss += loss.item() # Calculate total training loss over the entire epoch.

            if validate:
                #valid_loss = validate_one_epoch(model, diffusion, valid_loader)
                # Function call is very slow! I think it is slow now because of the if clauses!
                #######################################################################
                # Code from function "validate_one_epoch"
                model.eval()
                diffusion.eval()
                valid_loss = 0.0
                for i, (inputs,_) in enumerate(valid_loader):
                    # Load data onto device.
                    inputs = inputs.to(device)

                    # We sample new times. 
                    t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

                    # We noise the validation inputs at the new sampled times. 
                    x_t, noise = diffusion.noise_data_point(inputs, t) 

                    predicted_noise = model(x_t, t) # Predict the noise of the validation data. 

                    # Gaussian diffusion uses MSE loss. 
                    loss = diffusion.loss(noise, predicted_noise)
                    valid_loss += loss # Calculate the sum of validation loss over the entire epoch.
                #######################################################################
                # We do not divide the validation loss by the number of validation batches, since we validate on the entire validation set at once. 
                validation_losses[epoch] = valid_loss

        train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.             

        training_losses[epoch] = train_loss
        
        if validate:
            print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}. Validation loss after epoch {epoch+1} is {valid_loss:.4f}.")
            
            # Saving models each time the validation loss reaches a new minimum.
            if min_valid_loss > valid_loss:
                print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
                
                min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

                # Saving the new "best" models.             
                torch.save(diffusion.state_dict(), "./firstGaussianDiffusion.pth")
                torch.save(model.state_dict(), "./firstGaussianNeuralNet.pth")
                count_without_improving = 0
            else:
                count_without_improving += 1

            # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
            if count_without_improving >= 6:
                return training_losses, validation_losses
        else:
            print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}.")

    if validate:
        return training_losses, validation_losses
    return training_losses

# We import the Data-class (++) which we made for the Adult data. 
from Data import Data, CustomDataset, ToTensor
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

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

Adult = Data(adult_data, categorical_features, numerical_features, splits = [0.85,0.15])
X_train, y_train = Adult.get_training_data_preprocessed()
X_test, y_test = Adult.get_test_data_preprocessed()

# We are only interested in the numerical features when working with Gaussian diffusion. 
X_train = X_train[numerical_features]
X_test = X_test[numerical_features]

def plot_losses(training_losses, validation_losses):
    print(len(training_losses[training_losses != 0]))
    print(len(validation_losses[validation_losses != 0]))
    plt.plot(training_losses[training_losses != 0], color = "b", label = "Training")
    plt.plot(validation_losses[validation_losses != 0], color = "orange", label = "Validation")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def count_parameters(model):
    """Function for counting how many parameters require optimization."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

training_losses, validation_losses = train(X_train, y_train, X_test, y_test, numerical_features, device, T = 100, 
                        schedule = "linear", batch_size = 4096, num_epochs = 100, 
                        num_mlp_blocks = 6, dropout_p = 0.0, validate = True)

plot_losses(training_losses, validation_losses)

# Try to evaluate the model.
def evaluate(n, generate = True, plot_correlation = True): 
    """Try to see if we can sample synthetic data from the Gaussian Diffusion model."""

    # Load the previously saved models.
    model = NeuralNetModel(X_train.shape[1], 6, 0.0).to(device)
    diffusion = GaussianDiffusion(numerical_features, 100, "linear", device)
    model.load_state_dict(torch.load("./firstGaussianNeuralNet.pth"))
    diffusion.load_state_dict(torch.load("./firstGaussianDiffusion.pth")) 
    # Don't think it is necessary to save and load the diffusion model!

    with torch.no_grad():
        model.eval()
        diffusion.eval()    

        # Run the noise backwards through the backward process in order to generate new data. 
        if generate:
            synthetic_samples = diffusion.sample(model, n)
            synthetic_samples = pd.DataFrame(synthetic_samples, columns = X_train.columns.tolist())
            synthetic_samples.to_csv("first_synthetic_sample.csv")
        else:
            # Load the synthetic sample we already created. 
            synthetic_samples = pd.read_csv("first_synthetic_sample.csv", index_col = 0)

        print(synthetic_samples.shape)
        print(f"Synthetic samples: {synthetic_samples}")
        print(synthetic_samples.head())
        print(f"Synthetic: {synthetic_samples.describe()}")
        print(f"Train: {X_train.describe()}")

        def visualize_synthetic_data(synthetic_data, real_data):
            """Plot histograms over synthetic data against real training data."""
            fig, axs = plt.subplots(3,2)
            axs = axs.ravel()
            for idx, ax in enumerate(axs):
                ax.hist(synthetic_data.iloc[:,idx], density = True, color = "b", label = "Synth.", bins = 100)
                ax.hist(real_data.iloc[:,idx], color = "orange", alpha = 0.7, density = True, label = "OG.", bins = 100)
                ax.set_xlim(np.quantile(synthetic_data.iloc[:,idx], 0.01), np.quantile(synthetic_data.iloc[:,idx], 0.99))
                ax.legend()
                ax.title.set_text(real_data.columns.tolist()[idx])
            plt.tight_layout()
            #plt.show()

        visualize_synthetic_data(synthetic_samples, X_train)
        plt.show()

        def plot_correlation():
            synthetic_corr = synthetic_samples.corr()
            true_corr = X_train.corr()
            _, ax = plt.subplots()
            sns.heatmap(synthetic_corr, annot = True, fmt = ".2f", ax = ax)
            ax.set_title("Synthetic Correlation")
            plt.tight_layout()
            _, ax2 = plt.subplots()
            ax2 = sns.heatmap(true_corr, annot = True, fmt = ".2f", ax = ax2)
            ax2.set_title("True Correlation")
            plt.tight_layout()
            plt.show()

        if plot_correlation:
            plot_correlation()

        # Visualize again after descaling.
        visualize_synthetic_data(Adult.descale(synthetic_samples), Adult.get_training_data()[0][numerical_features])
        visualize_synthetic_data(Adult.descale(synthetic_samples), Adult.descale(X_train)) # Just making sure both these lines are the same!
        plt.show()
        print(f"Descaled synthetic: {Adult.descale(synthetic_samples).describe()}")
        print(f"Original training: {Adult.get_training_data()[0][numerical_features].describe()}")
        # print(f"Descaled: {Adult.descale(X_train).describe()}")
        # print(Adult.get_training_data()[0][numerical_features].equals(Adult.descale(X_train).astype("int64")))
        # print(Adult.get_training_data()[0][numerical_features].describe() == Adult.descale(X_train).astype("int64").describe())
        # print(Adult.get_training_data()[0][numerical_features].info())
        # print(Adult.get_training_data()[0][numerical_features].head())
        # print(Adult.descale(X_train).astype("int64").info())
        # print(Adult.descale(X_train).astype("int64").head())
        # print(np.allclose(Adult.get_training_data()[0][numerical_features].to_numpy(), 
        #                    Adult.descale(X_train).to_numpy(),rtol = 1e-40)) # They are equal, even though the tests above don't say so. 
        

#evaluate(X_train.shape[0], generate = True)

def check_forward_process(X_train, y_train, numerical_features, T, schedule, device, batch_size = 1, mult_steps = False):
    """Check if the forward diffusion process in Gaussian diffusion works as intended."""
    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    diffusion = GaussianDiffusion(numerical_features, T, schedule, device) # Numerical_features is not used for anything now. 

    inputs, _ = next(iter(train_loader)) # Check for first batch. 

    inputs = inputs.to(device)

    if mult_steps:
        # If we want to visualize the data in several steps along the way. 
        times = [0, int(T/5), int(2*T/5), int(3*T/5), int(4*T/5), T-1] # The six times we want to visualize.
        x_T_dict = {}
        for i, time in enumerate(times):
            x_T, noise = diffusion.noise_data_point(inputs, torch.tensor([times[i]])) 
            x_T = x_T.cpu().numpy()
            x_T_dict[i] = x_T
    else:
        # If we only want to visualize the data in the last latent variable. 
        x_T, noise = diffusion.noise_data_point(inputs, torch.tensor([T-1])) # We have to give it a tensor with the index value of the last time step. 
        x_T = x_T.cpu().numpy()
        noise = noise.cpu().numpy()
    
    inputs = inputs.cpu().numpy()
    
    # Plot the numerical features after forward diffusion together with normally sampled noise and the original data. 
    if mult_steps: 
        for i, feat in enumerate(numerical_features):
            fig, axs = plt.subplots(2,3)
            axs = axs.ravel()
            for idx, ax in enumerate(axs):
                ax.hist(x_T_dict[idx][:,i], density = True, color = "b", bins = 100) 
                ax.set_xlabel(f"Time {times[idx]}")
            fig.suptitle(f"Feature '{feat}'")
            plt.tight_layout()
        plt.show()
    else: 
        fig, axs = plt.subplots(3,2)
        axs = axs.ravel()
        for idx, ax in enumerate(axs):
            ax.hist(x_T[:,idx], density = True, color = "b", label = "Synth.", bins = 100)
            ax.hist(inputs[:,idx], color = "orange", alpha = 0.7, density = True, label = "OG.", bins = 100)
            #ax.hist(noise[:,idx], color = "purple", alpha = 0.5, density = True, label = "Noise Added", bins = 100)
            ax.legend()
            ax.title.set_text(numerical_features[idx])
        plt.tight_layout()
        plt.show()

    print(f"x_T = {pd.DataFrame(x_T, columns = X_train.columns.to_list()).describe()}")
    print()
    print(f"inputs = {pd.DataFrame(inputs, columns = X_train.columns.to_list()).describe()}")
    print()
    print(f"noise = {pd.DataFrame(noise, columns = X_train.columns.to_list()).describe()}")
    print()

    # Randomly sample from standard normal.
    # r_nsample = np.random.standard_normal(size = (batch_size, len(numerical_features)))
    # print(f"random = {pd.DataFrame(r_nsample).describe()}")
    # fig, axs = plt.subplots(3,2)
    # axs = axs.ravel()
    # for idx, ax in enumerate(axs):
    #     ax.hist(r_nsample[:,idx], density = True, color = "b", label = "Std. Normal")
    #     ax.hist(noise[:,idx], color = "orange", alpha = 0.5, density = True, label = "Noise Added")
    #     ax.legend()
    #     ax.title.set_text(numerical_features[idx])
    # plt.tight_layout()
    # plt.show() 

# The forward process seems to work fine for both schedules!
#check_forward_process(X_train, y_train, numerical_features, T = 1000, schedule = "linear", device = device, batch_size = X_train.shape[0], mult_steps=True)
#check_forward_process(X_train, y_train, numerical_features, T = 1000, schedule = "linear", device = device, batch_size = X_train.shape[0])

def plot_schedules(numerical_features, T, device):
    """Check if the schedules make sense (compare to plot in Improved DDPMs by Nichol and Dhariwal)."""

    diffusion_linear = GaussianDiffusion(numerical_features, T, "linear", device)
    diffusion_cosine = GaussianDiffusion(numerical_features, T, "cosine", device)
    
    # Get alpha_bar from the two schedules. 
    alpha_bar_linear = diffusion_linear.state_dict()["alpha_bar"].cpu().numpy()
    alpha_bar_cosine = diffusion_cosine.state_dict()["alpha_bar"].cpu().numpy()

    t = np.linspace(0,1,T)
    plt.plot(t, alpha_bar_linear, color = "blue", label = "linear")
    plt.plot(t, alpha_bar_cosine, color = "orange", label = "cosine")
    plt.title("Variance Schedules")
    plt.xlabel("diffusion step (t/T)")
    plt.ylabel("alpha_bar")
    plt.legend()
    plt.show() # Looks good!

# Schedules look qualitatively correct (similar to Figure 5 in Improved DDPMs).
#plot_schedules(numerical_features, T = 1000, device = device)
