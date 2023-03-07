# Class for Gaussian diffusion.

import numpy as np
from functools import partial
import torch
import torch.nn as nn
import math

from utils import extract

class Gaussian_diffusion(nn.Module):
    """Class for Gaussian diffusion. Handles only numerical features. 
    
    Parameters
    ----------
    numerical_features : list
        List of names of the numerical features in the data set.
    T : int
        Number of diffusion steps. 
    schedule_type : string
        The variance schedule to use. Only "linear" and "cosine" are implemented. 
    device : torch.device
        Device to train the PyTorch models on (cpu or gpu).

    Methods 
    -------
    noise_data_point : 

    sample :

    prepare_noise_schedule :

    
    loss :
    
    """
    def __init__(self, numerical_features, T, schedule_type, device):
        super(Gaussian_diffusion, self).__init__()
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

        # Parameters for forward posterior. These are not used in my sampling right now, but they could come in handy later, so I will leave them here for now.
        self.register_buffer("beta_tilde", to_torch(beta_tilde).to(self.device))
        self.register_buffer("mu_tilde_coef1", to_torch(mu_tilde_coef1).to(self.device))
        self.register_buffer("mu_tilde_coef2", to_torch(mu_tilde_coef2).to(self.device))
        
    def noise_data_point(self, x_0, t):
        """Get x_t (noised input x_0 at times t), following the closed form Equation 4 in DDPM by Ho et. al."""
        noise = torch.randn_like(x_0)
        assert x_0.shape == noise.shape
        return extract(self.sqrt_alpha_bar, t, x_0.shape)*x_0 \
                + extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)*noise, noise

    def sample(self, model, n, y_dist=None):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper.
        'model' is the neural network that is used to predict the noise in each time step. 
        """
        print("Entered function for sampling in Gaussian_diffusion.")

        if model.is_class_cond:
            if y_dist is None:
                raise Exception("You need to supply the distribution of labels (vector) when the model is class-conditional.")
                # For example for "Adult": supply y_dist = torch.tensor([0.75, 0.25]), since about 25% of the data set are reported with positive outcome. 
        
        y = None
        if model.is_class_cond:
            y = torch.multinomial( # This makes sure we sample the classes according to their proportions in the real data set, at each step in the generative process. 
                y_dist,
                num_samples=n,
                replacement=True
            ).to(self.device)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n,len(self.numerical_features))).to(self.device) # Sample from standard Gaussian (sample from x_T). 
            for i in reversed(range(self.T)): # I start it at 0.
                if i % 25 == 0:
                    print(f"Sampling step {i}.")
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                predicted_noise = model(x,t,y)
                sh = predicted_noise.shape

                betas = extract(self.betas, t, sh) 
                sqrt_recip_alpha = extract(self.sqrt_recip_alpha, t, sh)
                sqrt_recip_one_minus_alpha_bar = extract(self.sqrt_recip_one_minus_alpha_bar, t, sh)
                
                # Version #1 of sigma. 
                #sigma = extract(torch.sqrt(self.betas), t, sh) # Ho et al. used this as sigma.
                
                # Version #2 of sigma. This is better in line with the theory discussed in the thesis. 
                sigma = extract(torch.sqrt(self.beta_tilde), t, sh)
            
                if i > 0:
                    noise = torch.randn_like(x)
                else: # We don't want to add noise at t = 0, because it would make our outcome worse (this comes from the fact that we have another term in Loss for x_0|x_1, I believe).
                    noise = torch.zeros_like(x)
                x = sqrt_recip_alpha * (x - (betas * sqrt_recip_one_minus_alpha_bar)*predicted_noise) + sigma * noise # Use formula in line 4 in Algorithm 2.

        model.train() # Indicate to Pytorch that we are back to doing training. 
        if model.is_class_cond:
            y = y.reshape(-1,1)
        return x, y

    def prepare_noise_schedule(self):
        """Prepare the betas in the variance schedule."""
        if self.schedule_type == "linear":
            # Linear schedule from Ho et. al, extended to work for any number of diffusion steps. 
            # ERROR: Fails for small number of steps, e.g. 10. Use cosine if using small number of steps!
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
        return torch.randint(low=0, high=self.T, size = (n,)) # Tror denne må starte på 0!

    def loss(self, real_noise, model_pred):
        """Function to return Gaussian loss, i.e. MSE."""
        return torch.mean((real_noise - model_pred)**2) # We take the mean over all dimensions. Seems correct. 
        # Hva med når t = 0, dvs p(x_0|x_1). Må jeg ha med en slags korreksjon for dette?
        # Kanskje en slik korreksjon ville gjort at jeg kunne brukt Gaussian Diffusion til både numeriske og kategoriske variabler?
        # Dvs hatt en siste "decoder" i lossen som gir softmax for kategorisk of vanlig output for numeriske. 
        # Men hva slags label kan jeg da sammenligne en slik softmax med? Det er dette jeg sliter med!
