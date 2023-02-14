# Multinomial diffusion for categorical features in tabular data, first implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import math
from torchinfo import summary

def extract(a, t, x_shape):
    """Changes the dimensions of the input a depending on t and x_t.

    Makes them compatible such that pointwise multiplication of tensors can be done.
    Each column in the same row gets the same value after this "extract" function, 
    such that each data point column can be multiplied by the same number when noising and denoising. 
    """
    t = t.to(a.device) # Change the device to make sure they are compatible. 
    out = a[t] # Get the correct time steps from a. 
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape) # Expand such that all elements are correctly multiplied by the noise. 

def index_to_log_onehot(x, num_classes):
    """Convert a vector with an index to a one-hot-encoded vector.
    
    This has been copied directly from https://github.com/ehoogeboom/multinomial_diffusion/blob/main/diffusion_utils/diffusion_multinomial.py. 
    """
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}' # Code fails on this one!
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x

class MultinomialDiffusion(nn.Module):
    """Class for Multinomial diffusion, for all the categorical features in the data set."""

    # Denne fungerer vel kanskje bare for én kategorisk feature om gangen?

    def __init__(self, categorical_feature_names, categorical_levels, T, schedule_type, model, device):
        super(MultinomialDiffusion, self).__init__()
        self.categorical_feature_names = categorical_feature_names
        self.categorical_levels = categorical_levels
        self.num_classes = len(categorical_levels)
        assert len(categorical_levels) == len(categorical_feature_names), \
                            f"Categorical levels {categorical_levels} and features names {categorical_feature_names} must be two lists of same length."
        self.T = T
        self.schedule_type = schedule_type
        self.model = model
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
        sqrt_recip_alpha = np.sqrt(1.0 / alphas)
        sqrt_recip_one_minus_alpha_bar = np.sqrt(1.0 / one_minus_alpha_bar)
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
        self.register_buffer("sqrt_recip_alpha", to_torch(sqrt_recip_alpha).to(self.device))
        self.register_buffer("sqrt_recip_one_minus_alpha_bar", to_torch(sqrt_recip_one_minus_alpha_bar).to(self.device))
        self.register_buffer("alpha_bar_prev", to_torch(alpha_bar_prev).to(self.device))
        self.register_buffer("alpha_bar_next", to_torch(alpha_bar_next).to(self.device))

        # Parameters for forward posterior. 
        self.register_buffer("beta_tilde", to_torch(beta_tilde).to(self.device))
        self.register_buffer("mu_tilde_coef1", to_torch(mu_tilde_coef1).to(self.device))
        self.register_buffer("mu_tilde_coef2", to_torch(mu_tilde_coef2).to(self.device))
        
    def noise_one_step(self, log_x_t_1, t):
        """Noise x_{t-1} to x_t, following Equation 11 Hoogeboom et. al.

        q(x_t|x_{t-1}).
        
        Returns the log of the new probability that is used in the categorical distribution to sample x_t.
        """
        log_alpha_t = torch.log(extract(self.alphas, t, log_x_t_1.shape))
        log_beta_t = torch.log(extract(self.betas, t, log_x_t_1.shape))
        return torch.logaddexp(log_alpha_t + log_x_t_1, log_beta_t - np.log(self.num_classes))

    def noise_data_point(self, log_x_0, t):
        """Get x_t (noised input x_0 at times t), following the closed form Equation 12 in Hoogeboom et. al.
        
        q(x_t|x_0).

        Returns the log of the new probability that is used in the categorical distribution to sample x_t.
        """
        log_alpha_bar_t = torch.log(extract(self.alpha_bar, t, log_x_0.shape))
        log_one_minus_alpha_bar_t = torch.log(extract(self.one_minus_alpha_bar, t, log_x_0.shape))
        return torch.logaddexp(log_alpha_bar_t + log_x_0, log_one_minus_alpha_bar_t - np.log(self.num_classes))

    def theta_post(self, log_x_t, log_x_0, t):
        """This is the probability parameter in the posterior categorical distribution, called theta_post by Hoogeboom et. al."""
        
        log_probs_x_t_1 = self.noise_data_point(log_x_0, t-1) # This is [\bar \alpha_{t-1} x_0 + (1-\bar \alpha_{t-1})/K].

        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        
        # If t == 0 log_probs_x_t_1 should be set to log_x_0, else to self.noise_data_point(log_x_0, t-1).
        log_probs_x_t_1 = torch.where(t_broadcast == 0, log_x_0, log_probs_x_t_1)

        log_tilde_theta = log_probs_x_t_1 + self.noise_one_step(log_x_t, t) # This is the entire \tilde{\theta} probability vector.

        normalized_log_tilde_theta = log_tilde_theta - torch.logsumexp(log_tilde_theta, dim = 1, keepdim=True) # This is log_theta_post. 
        return normalized_log_tilde_theta

    def reverse_pred(self, log_x_t, t):
        """Returns the probability parameter of the categorical distribution of the backward process, based on the predicted x_0 from the neural net."""
        hat_x_0 = self.model(log_x_t,t) # Predict x_0 from the model. We keep the one-hot-encoding of log_x_t and feed it like that into the model. 
        log_hat_x_0 = F.log_softmax(hat_x_0, dim = 1) # log_softmax: find the logarithm of softmax of hat_x_0.
        log_tilde_theta_hat = self.theta_post(log_x_t, log_hat_x_0, t) # Find the probability parameter of the reverse process, based on the prediction from the neural net. 
        return log_tilde_theta_hat

    def sample(self, n):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper, modified for multinomial diffusion.
        'model' is the neural network that is used to predict the x_0 from a noised x_t each time step t. 
        """
        print("Entered function for sampling.")
        self.model.eval()
        x_list = {}
        with torch.no_grad():
            x = torch.randn((n,self.model.input_size)).to(self.device) # Sample from standard Gaussian (sample from x_T). 
            # For multinomial diffusion we should probably sample from a uniform to begin with?
            for i in reversed(range(self.T)): # I start it at 0
                if i % 25 == 0:
                    print(f"Sampling step {i}.")
                x_list[i] = x
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                
                # Get reverse process probability parameter. 
                log_tilde_theta_hat = self.reverse_pred(x, t)

                # Sample from a categorical distribution based on theta_post. 
                # For this we use the gumbel-softmax trick. We put this in another function.
                x = self.sample_categorical(log_tilde_theta_hat)

        self.model.train() # Indicate to Pytorch that we are back to doing training. 
        return x, x_list # We return a list of the x'es to see how they develop from t = 99 to t = 0.

    def sample_categorical(self, log_probs):
        """Sample from a categorical distribution using the gumbel-softmax trick."""
        uniform = torch.rand_like(log_probs) # Why logits (log(p/(1-p))) and not log_prob?
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + log_probs).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes) # Want to check how the index_to_log_onehot function actually works!
        return log_sample

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
        return torch.randint(low=0, high=self.T, size = (n,)) 

    def categorical_kl(self, log_prob_a, log_prob_b):
        """Calculate the KL divergence between log_prob_a and log_prob_b, following the definition of KL divergence for discrete quantities.
        
        The definition states
        D(p_a || p_b) = \sum p_a * log(p_a/p_b) = \sum p_a * (log_prob_a - log_prob_b).
        """
        return (log_prob_a.exp() * (log_prob_a - log_prob_b)).sum(dim = 1)

    def loss(self, log_x_0, log_x_t, t):
        """Function to return the loss. This loss represents each term L_{t-1} in the ELBO of diffusion models.
        
        KL( q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t) ) = KL( Cat(\pi(x_t,x_0)) || Cat(\pi(x_t, \hatx_0)) ).

        We also need to compute the term log p(x_0|x_1) if t = 1.
        """

        # Find the true theta post, i.e. based on the true log_x_0.
        log_true_theta = self.theta_post(log_x_t, log_x_0, t)

        # Find the predicted theta post, i.e. based on the predicted log_x_0 based on the neural network. 
        log_predicted_theta = self.reverse_pred(log_x_t, t)

        # Calculate the KL divergence between the categorical distributions with probability parameters true theta post and predicted theta post. 
        lt = self.categorical_kl(log_true_theta, log_predicted_theta)

        # Make mask for t == 0, where we need a different calculation for log(p(x_0|x_1)). We call this different calculation the decoder_loss.
        mask = (t == torch.zeros_like(t)).float() # If t == 0, we calculate sum x_0*log \hatx_0 over all classes (columns). Else we return L_{t-1}.

        decoder_loss = -(log_x_0.exp() * log_predicted_theta).sum(dim=1) # Dette kan vel umulig stemme? Det burde vel være log(\hatx_0) i andre ledd? (og ikke theta_post(x_t,\hatx_0)).

        loss = mask * decoder_loss + (1. - mask) * lt
        return  torch.nanmean(loss)# We take the mean such that we return a scalar, which we can backprop through.
                        # Use nanmean() since we have nans in the loss! How can we deal with this?

class NeuralNetModel(nn.Module):
    """Main model for predicting multinomial initial point.
    
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
        # I stole this function directly from https://github.com/rotot0/tab-ddpm/blob/main/tab_ddpm/modules.py
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

        # Feed the embeddings to our "regular" MLP. 
        x = self.dropout(self.relu(self.l1(x))) # First MLPBlock
        for ll in self.linear_layers:
            x = self.dropout(self.relu(ll(x))) # MLPBlocks in between. 
        x = self.outlayer(x) # Linear out-layer.
        return x

def train(X_train, y_train, X_valid, y_valid, categorical_feature_names, categorical_levels, 
            device, T = 1000, schedule = "linear", batch_size = 4096, 
            num_epochs = 100, num_mlp_blocks = 4, dropout_p = 0.4):
    """Function for the main training loop of the Gaussian diffusion model."""
    input_size = 1#X_train.shape[1] # Columns in the training data is the input size of the neural network model. 

    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor())         
    valid_data = CustomDataset(X_valid, y_valid, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_data, batch_size = X_valid.shape[0], num_workers = 2) # We want to validate on the entire validation set in each epoch

    # Define model for predicting noise in each step. 
    model = NeuralNetModel(input_size, num_mlp_blocks, dropout_p).to(device)
    summary(model) # Plot the summary from torchinfo.

    # Define Multinomial Diffusion object.
    diffusion = MultinomialDiffusion(categorical_feature_names, categorical_levels, T, schedule, model, device)

    # Define the optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Main training loop.
    training_losses = np.zeros(num_epochs)
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
            
            # Assuming the data is already one-hot-encoded, we do a log-transformation of the data. 
            log_inputs = torch.log(inputs)

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

            # Noise the inputs and return the noised input in log_space.
            log_x_t = diffusion.noise_data_point(log_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Calculate the loss between the noised input and the true input. 
            # Predictions via the model is done directly inside the loss function below. 
            loss = diffusion.loss(log_inputs, log_x_t, t) # This needs to return a scalar, which it is not at the moment!

            optimizer.zero_grad()
            loss.backward() # Calculate gradients. 
            optimizer.step() # Update parameters. 
            train_loss += loss.item() # Calculate total training loss over the entire epoch.

        train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.  
        
        ######################### Validation.
        model.eval()
        diffusion.eval()
        valid_loss = 0.0
        for i, (inputs,_) in enumerate(valid_loader):
            # Load data onto device.
            inputs = inputs.to(device)
            
            # Assuming the data is already one-hot-encoded, we do a log-transformation of the data. 
            log_inputs = torch.log(inputs)

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

            # Noise the inputs and return the noised input in log_space.
            log_x_t = diffusion.noise_data_point(log_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Calculate the loss between the noised input and the true input. 
            # Predictions via the model is done directly inside the loss function below. 
            loss = diffusion.loss(log_inputs, log_x_t, t)
            
            valid_loss += loss # Calculate total validation loss over the entire epoch.
        #########################         

        training_losses[epoch] = train_loss
        validation_losses[epoch] = valid_loss
        # We do not divide the validation loss by the number of validation batches, since we validate on the entire validation set at once. 
        
        print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}. Validation loss after epoch {epoch+1} is {valid_loss:.4f}.")
        
        # Saving models each time the validation loss reaches a new minimum.
        if min_valid_loss > valid_loss:
            print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
            
            min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

            # Saving the new "best" models.             
            torch.save(diffusion.state_dict(), "./MultDiffusion.pth")
            torch.save(model.state_dict(), "./MultNeuralNet.pth")
            count_without_improving = 0
        else:
            count_without_improving += 1

        # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
        if count_without_improving >= 8:
            return training_losses, validation_losses
        
    return training_losses, validation_losses

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

Adult = Data(adult_data, categorical_features, numerical_features, scale_version = "quantile", splits = [0.85,0.15])
X_train, y_train = Adult.get_training_data_preprocessed()
X_test, y_test = Adult.get_test_data_preprocessed()
print(X_train.shape)

def find_levels(df, categorical_features):
    """Returns a list of levels of features of each of the categorical features."""
    lens_categorical_features = []
    for feat in categorical_features:
        unq = len(df[feat].value_counts().keys().unique())
        print(f"Feature '{feat}'' has {unq} unique levels")
        lens_categorical_features.append(unq)
    print(f"The sum of all levels is {sum(lens_categorical_features)}. This will be the number of cat-columns after one-hot encoding (non-full rank)")
    return(lens_categorical_features)

lens_categorical_features = find_levels(adult_data.loc[:,adult_data.columns != "y"], categorical_features)
print(lens_categorical_features)

# We are only interested in the categorical features when working with Multinomial diffusion. 
X_train = X_train.drop(numerical_features, axis = 1)
X_test = X_test.drop(numerical_features, axis = 1)

# I think the Multinomial Diffusion only works for one variable at a time at this point!
# Check if this is true!
X_train = X_train.iloc[:,0]
X_test = X_test.iloc[:,0]
print(X_train.shape)

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

training_losses, validation_losses = train(X_train, y_train, X_test, y_test, categorical_features, 
                        lens_categorical_features, device, T = 100, 
                        schedule = "linear", batch_size = 4096, num_epochs = 100, 
                        num_mlp_blocks = 4, dropout_p = 0.0)

plot_losses(training_losses, validation_losses)

# Try to evaluate the model.
def evaluate(n, generate = True, save_figs = False): 
    """Try to see if we can sample synthetic data from the Gaussian Diffusion model."""

    # Load the previously saved models.
    model = NeuralNetModel(X_train.shape[1], 4, 0.0).to(device)
    model.load_state_dict(torch.load("./MultNeuralNet.pth"))
    diffusion = MultinomialDiffusion(categorical_features, lens_categorical_features, 100, "linear", model, device)
    diffusion.load_state_dict(torch.load("./MultDiffusion.pth")) 
    # Don't think it is necessary to save and load the diffusion model!
    # We still do it to be safe. 

    with torch.no_grad():
        model.eval()
        diffusion.eval()    

        # Run the noise backwards through the backward process in order to generate new data. 
        if generate:
            synthetic_samples, reverse_points_list = diffusion.sample(n)
            synthetic_samples = Adult.decode(synthetic_samples)
            synthetic_samples = pd.DataFrame(synthetic_samples, columns = X_train.columns.tolist())
            synthetic_samples.to_csv("synthetic_sample_mult_diff.csv")
        else:
            # Load the synthetic sample we already created. 
            synthetic_samples = pd.read_csv("first_synthetic_sample_mult_diff.csv", index_col = 0)

        print(synthetic_samples.shape)
        print(synthetic_samples.head())

        def visualize_categorical_data(synthetic_data, real_data):
            """Plot barplots and mosaic plots of the synthetic data against the real training data for categorical features."""
            fig, axs = plt.subplots(2,2)
            axs = axs.ravel()
            for idx, ax in enumerate(axs):
                synthetic_data[categorical_features[idx]].value_counts().plot(kind='bar', ax = ax)
                real_data[categorical_features[idx]].value_counts().plot(kind='bar', ax = ax, color = "orange", alpha = 0.6)
                ax.title.set_text(categorical_features[idx])
            plt.tight_layout()

            # Make two grids since 7 is not an even number of categorical features. 
            fig, axs2 = plt.subplots(2,2)
            axs2 = axs2.ravel()
            for idx, ax in enumerate(axs2, start = 4):
                if idx > len(categorical_features)-1:
                    break
                synthetic_data[categorical_features[idx]].value_counts().plot(kind='bar', ax = ax)
                
                real_data[categorical_features[idx]].value_counts().plot(kind='bar', ax = ax, color = "orange", alpha = 0.6)
                ax.title.set_text(categorical_features[idx])
            plt.tight_layout()


            # Make mosaic plots later if I feel like it!
            # E.g. https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib

        # Visualize again after descaling.
        visualize_categorical_data(synthetic_samples, Adult.get_training_data()[0][categorical_features])
        if save_figs:
            plt.savefig("synthetic_mult_diff.pdf")

        # The function below needs to be changed to fit the categorical data!
        def look_at_reverse_process_steps(x_list, T):
            """We plot some of the steps in the backward process to visualize how it changes the data."""
            times = [0, int(T/5), int(2*T/5), int(3*T/5), int(4*T/5), T-1][::-1] # Reversed list of the same times as visualizing forward process. 
            for i, feat in enumerate(categorical_features):
                fig, axs = plt.subplots(2,3)
                axs = axs.ravel()
                for idx, ax in enumerate(axs):
                    ax.hist(x_list[times[idx]][:,i], density = True, color = "b", bins = 100) 
                    ax.set_xlabel(f"Time {times[idx]}")
                    #ax.set_xlim(np.quantile(x_list[times[idx]][:,i], 0.05), np.quantile(x_list[times[idx]][:,i], 0.95))
                fig.suptitle(f"Feature '{feat}'")
                plt.tight_layout()
            plt.show()
        
        #look_at_reverse_process_steps(reverse_points_list, diffusion.T)
        #print(reverse_points_list)

evaluate(X_train.shape[0], generate=True, save_figs=False)

# The function below needs to be changed to fit for the categorical data!
def check_forward_process(X_train, y_train, numerical_features, T, schedule, device, batch_size = 1, mult_steps = False):
    """Check if the forward diffusion process in Gaussian diffusion works as intended."""
    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    diffusion = MultinomialDiffusion(numerical_features, T, schedule, device) # Numerical_features is not used for anything now. 

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

    diffusion_linear = MultinomialDiffusion(numerical_features, T, "linear", device)
    diffusion_cosine = MultinomialDiffusion(numerical_features, T, "cosine", device)
    
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
