# This is my first iteration of tabDDPM
# DRAFT, very rough.

import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion():
    """Class for Gaussian diffusion, for all the numerical features in the data set."""

    def __init__(self, numerical_features, T, schedule_type):
        self.numerical_features = numerical_features
        self.T = T
        self.schedule_type = schedule_type
        
        # Prepare the noise schedule (either linear or cosine).
        self.betas = self.prepare_noise_schedule().to(self.device)

        # Calculate necessary quantities for formulas in papers.
        # Add to buffer? (Pytorch)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim = 0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)
        self.alpha_bar_prev = torch.cat(torch.tensor([1.0]), self.alpha_bar[:,-1])
        self.alpha_bar_next = torch.cat(self.alpha_bar[1:], torch.tensor([0.0]))
        # Sjekk at alle disse størrelsene er korrekte!
        # Hvis ikke; kan også gjøre disse beregningene i numpy og deretter heller gjøre dem om til tensorer via pytorch. 

        self.beta_tilde = self.betas * (1.0 - self.alpha_bar_prev)/(1.0 - self.alpha_bar) # Equation 7 in DDPM.
        self.mu_tilde_coef1 = torch.sqrt(self.alpha_bar_prev)*self.betas / (1.0 - self.alpha_bar) # Equation 7 in DDPM. 
        self.mu_tilde_coef2 = torch.sqrt(self.alphas)*(1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar) # Equation 7 in DDPM. 
        
        pass

    def forward_diffusion(self):
        """Forward diffuse all the way, i.e. return x_T."""
        # Use 'noise_data_point' to noise data point x. 
        # This function is perhaps not needed then, since we can do it in one line with 'noise_data_point'.
        return self.noise_data_point(x_0, self.T)
        
    def noise_data_point(self, x_0, t):
        """Get x_t (noised input x_0 at time t), following the closed form Equation 4 in DDPM by Ho et. al."""
        noise = torch.rand_like(x_0)
        assert x_0.shape == noise.shape
        return self.sqrt_alpha_bar*x_0 + self.sqrt_one_minus_alpha_bar*noise
        # Perhaps the dimensions etc are wrong, but the idea holds.
        # Need to test the functions later. 

    def sample(self, model, n):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper.
        """
        pass

    def prepare_noise_schedule(self):
        """Prepare the betas in the variance schedule."""
        if self.schedule_type == "linear":
            pass # Return the linear schedule.
        elif self.schedule_type == "cosine":
            pass # Return the cosine schedule. 
        else:
            raise NotImplementedError(f"The schedule type {self.schedule_type} has not been implemented.")

    def sample_timesteps(self, n):
        """Sample timesteps (uniformly) for use when training the model."""
        return torch.randint(low=1, high=self.noise_steps, size = (n,))

    def loss(self, real_noise, model_pred):
        """Function to return Gaussian loss, i.e. MSE."""
        return (real_noise - model_pred)**2

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
        
        

# Where to implement the time embeddings?
# Are both the classes above an MLP like in Equation 4 in TabDDPM?


# I stole the function above directly from https://github.com/rotot0/tab-ddpm/blob/main/tab_ddpm/modules.py
# Have a look at it later in order to understand how these sinusoidal time embeddings work. 
import math
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
