# This is my first iteration of tabDDPM
# DRAFT, very rough.

import torch
import torch.nn as nn

class GaussianDiffusion():
    """Class for Gaussian diffusion, for all the numerical features in the data set."""

    def __init__(self, numerical_features, T, schedule_type):
        self.numerical_features = numerical_features
        self.T = T
        self.schedule_type = schedule_type
        
        # Prepare the noise schedule (either linear or cosine).
        self.betas = self.prepare_noise_schedule()

        # Calculate necessary quantities for formulas in papers.
        # Add to buffer? (Pytorch)
        self.alphas = 1 - self.betas 
        pass

    def forward_diffusion(self, x):
        """Forward diffuse a data point x, return x_T."""
        
        # Use 'noise_data_point' to noise data point x. 
        # This function is perhaps not needed then, since we can do it in one line with 'noise_data_point'.
        pass
        
    def noise_data_point(self, x, t):
        """Add noise to the data point x at time t, following the closed form Equation 4 in DDPM by Ho et. al."""
        pass

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


class MultinomialDiffusion():
    """Class for Multinomial diffusion. One for each categorical feature in the data set."""    

    def __init__(self, categorical_feature_name, levels):
        self.categorical_feature_name = categorical_feature_name
        self.levels = levels


class GaussianMultinomialDiffusion():
    """Class for concatenation of Gaussian and Multinomial diffusion."""

    def __init__(self):
        pass


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

    @staticmethod
    def time_embedding(t, input_size):
        """The rest of the time embedding, returns t_{emb}."""
        #return Linear(SiLU(Linear(sinusoidal_embedding(t)))
        pass
        

# Where to implement the time embeddings?
# Are both the classes above an MLP like in Equation 4 in TabDDPM?