# This is my first iteration of tabDDPM
# DRAFT, very rough.

import torch
import torch.nn as nn
import numpy as np
from functools import partial
import math

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
        """
        pass

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

class NoisePredictor(nn.Module):
    """Class for neural network for predicting noise for numerical features."""

    def __init__(self, input_size):
        super(NoisePredictor, self).__init__()
        self.input_size = input_size

        # Layers. Can tune the number of layers by extending this code later. 
        self.l = nn.Linear(128, 128)
        self.final_l = nn.Linear(128, input_size)

        # Activation functions.
        self.relu = nn.ReLU()

        # Other.
        self.dropout = nn.Dropout(p = 0.0) # Set to 0.0 in TabDDPM.

    def forward(self, x):
        """Forward function for Pytorch."""
        MLPBlock = self.dropout(self.relu(self.l(x)))
        MLPBlock = self.dropout(self.relu(self.l(MLPBlock)))
        # We try to layers for now, but this can be tuned later. 
        out = self.final_l(MLPBlock)
        return out

class OneHotPredictor(nn.Module):
    """Class for neural network for predicting one hot encoded input for categorical features."""

    def __init__(self):
        super(OneHotPredictor, self).__init__()
        
    def forward(self):
        """Forward function for Pytorch."""

class timeEmbedding(nn.Module):
    """Class for time embeddings (sinusoidal), following Equation (5) in TabDDPM by Kotelnikov et. al."""

    def __init__(self):
        super(timeEmbedding, self).__init__()

    @staticmethod
    def sinusoidal_embedding(t):
        """Sinusoidal time embedding. Returns the embedding."""
        pass

    @staticmethod # or @classmethod https://www.programiz.com/python-programming/methods/built-in/classmethod
    def time_embedding(t, input_size):
        """The rest of the time embedding, returns t_{emb}."""
        #return Linear(SiLU(Linear(sinusoidal_embedding(t)))
        dim_t = 128
        time_embed_nn = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        #time_emb = time_embed_nn(sinusoidal_embedding())
        

class NeuralNetModel(nn.Module):
    """Main model for predicting Gaussian noise.
    
    We make a simple model to begin with, just to see if we are able to train something. 
    """

    def __init__(self, input_size):
        super(NeuralNetModel, self).__init__()
        self.input_size = input_size

        # Layers.
        self.l1 = nn.Linear(128, 256)
        self.outlayer = nn.Linear(256, input_size)
        
        # Activation functions. 
        self.relu = nn.ReLU()

        # Dropout.
        self.dropout = nn.Dropout(p = 0.1) # Set some random dropout probability during training. 

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
        out = self.l1(x) 
        out = self.relu(out)
        out = self.dropout(out)
        out = self.outlayer(out)
        return out

    def sample(self, model, n):
        """Sample 'n' new data points using the diffusion model.
        
        'model' is the neural network that is used to predict the noise in each time step. 
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,)).to(self.device) # Sample from standard Gaussian (sample from x_T). We are not doing it for images, but for data points. This is only 1D for now. 
            # For tabular data I probably need (n,2) I think! Test later. 
            for i in reversed(range(1, self.noise_steps)): # Could add progress bar using tqdm here, as in the video.
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alphas[t] #[:, None, None, None] # I think this is for the images perhaps. 
                alpha_bar = self.alpha_bar[t] #[:, None, None, None] # I think this is for the images perhaps. 
                beta = self.betas[t] #[:, None, None, None] # I think this is for the images perhaps. 
                if i > 1:
                    noise = torch.rand_like(x)
                else: # We don't want to add noise at t = 1, because it would make our outcome worse (this comes from the fact that we have another term in Loss for x_0|x_1, I believe).
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1.0-alpha_bar)))*predicted_noise) + torch.sqrt(beta) * noise # Use formula in line 4 in Algorithm 2.
                                                                                            # Here we have defined sigma^2 = beta, which was one of the options the authors tested. 

        model.train() # Indicate to Pytorch that we are back to doing training. 

def train(X_train, y_train, numerical_features, T, schedule, device, batch_size = 1, num_epochs = 100):
    """Function for the main training loop of the Gaussian diffusion model."""
    input_size = X_train.shape[1] # Columns in the training data is the input size of the neural network model. 

    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    # Define Gaussian Diffusion object.
    # Fungerer ikke med mindre enn 1000 diffusion steps!
    diffusion = GaussianDiffusion(numerical_features, T, schedule, device) # Numerical_features is not used for anything now. 
    diffusion.train()

    # Define model for predicting noise in each step. 
    model = NeuralNetModel(input_size).to(device)
    model.train() # Set model to training mode (for correct dropout calculations).

    # Define the optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # Main training loop.
    training_losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
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
        training_losses[epoch] = loss
        print(f"The loss after epoch {epoch} is {loss}.")
    return training_losses, model, diffusion

# We import the Data-class (++) which we made for the Adult data. 
from Data import Data, CustomDataset, ToTensor
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

Adult = Data(adult_data, categorical_features, numerical_features, valid = True)
X_train, y_train = Adult.get_training_data_preprocessed()
X_test, y_test = Adult.get_test_data_preprocessed()
X_valid, y_valid = Adult.get_validation_data_preprocessed()

# We are only interested in the numerical features when working with Gaussian diffusion. 
X_train  = X_train[numerical_features]
X_test  = X_test[numerical_features]
X_valid  = X_valid[numerical_features]

batch_size = int(X_train.shape[0]/2)
num_epochs = 50

#training_losses, model, diffusion = train(X_train, y_train, numerical_features, 100, "linear", device, batch_size, num_epochs)
#print(len(training_losses))
#plt.plot(training_losses)
#plt.show()

# Save the model.
#torch.save(diffusion.state_dict(), "./firstGaussianDiffusion.pth")
#torch.save(model.state_dict(), "./firstGaussianNeuralNet.pth")

# Load the previously saved models.
#model = NeuralNetModel(X_train.shape[1]).to(device)
#diffusion = GaussianDiffusion(numerical_features, 100, "linear", device)
#model.load_state_dict(torch.load("./firstGaussianDiffusion.pth"))
#diffusion.load_state_dict(torch.load("./firstGaussianNeuralNet.pth"))

# Try to evaluate the model.
def evaluate(model, diffusion, X): # Could try feeding it both X_train and X_test.
    """Try to see if we can sample synthetic data from the Gaussian Diffusion model."""
    with torch.no_grad():
        model.eval()
        diffusion.eval()    
        # Not quite sure if I need both models for now. I think I need both: one for predicting noise in previous step and another for ...


        # Sample from a standard normal. 
        noise = torch.randn_like(X)

        # Run the noise backwards through the backward process in order to generate new data. 

        # Må gjøre noe slikt som nedenfor (lignende). 
        # Dette er direkte kopiert fra den andre kodebasen, og det er en del ting jeg ikke forstår her!
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn( # Dette er output fra modellen i hvert steg. 
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)


def check_forward_process(X_train, y_train, numerical_features, T, schedule, device, batch_size = 1):
    """Check if the forward diffusion process in Gaussian diffusion works as intended."""
    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    # Define Gaussian Diffusion object.
    # Fungerer ikke med mindre enn 1000 diffusion steps!
    diffusion = GaussianDiffusion(numerical_features, T, schedule, device) # Numerical_features is not used for anything now. 

    inputs, labels = next(iter(train_loader)) # Check for first batch. 

    inputs = inputs.to(device)

    x_T, noise = diffusion.noise_data_point(inputs, torch.tensor([T-1])) # We have to give it a tensor with the index value of the last time step. 

    x_T = x_T.cpu().numpy()
    noise = noise.cpu().numpy()
    inputs = inputs.cpu().numpy()
    
    # Plot the numerical features after forward diffusion together with normally sampled noise and the original data. 
    fig, axs = plt.subplots(3,2)
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.hist(x_T[:,idx], density = True, color = "b", label = "Synth.", bins = 100)
        ax.hist(inputs[:,idx], color = "orange", alpha = 0.7, density = True, label = "OG.", bins = 100)
        ax.hist(noise[:,idx], color = "purple", alpha = 0.5, density = True, label = "Noise Added", bins = 100)
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
    r_nsample = np.random.standard_normal(size = (batch_size, len(numerical_features)))
    print(f"random = {pd.DataFrame(r_nsample).describe()}")
    fig, axs = plt.subplots(3,2)
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.hist(r_nsample[:,idx], density = True, color = "b", label = "Std. Normal")
        ax.hist(noise[:,idx], color = "orange", alpha = 0.5, density = True, label = "Noise Added")
        ax.legend()
        ax.title.set_text(numerical_features[idx])
    plt.tight_layout()
    plt.show() 

#check_forward_process(X_train, y_train, numerical_features, T = 1000, schedule = "cosine", device = device, batch_size = 1500)

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

#plot_schedules(numerical_features, T = 1000, device = device)
