# Multinomial diffusion for categorical features in tabular data, first implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import math
from torchinfo import summary
from statsmodels.graphics.mosaicplot import mosaic # For mosaic plots. 
from torchmetrics.functional.nominal import theils_u_matrix # For Theil's U statistic between categorical variables. 

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

def index_to_log_onehot_OLD(x, num_classes):
    """Convert a vector with an index to a one-hot-encoded vector.
    
    This has been copied directly from https://github.com/ehoogeboom/multinomial_diffusion/blob/main/diffusion_utils/diffusion_multinomial.py. 
    """
    # Denne støtter kun én kategorisk feature på én gang. 
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}' # Code fails on this one!
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order) # Why do they bother doing this permutation?

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x

def index_to_log_onehot(x, categorical_levels):
    """Convert a vector with an index to a one-hot-encoded vector.
    
    This has been heavily inspired by implementation in TabDDPM, which is a modified version of the original function from Hoogeboom et al.,
    such that it works for several categorical features at once. 
    """
    onehot = [] # Make one common list of one-hot-vectors. 
    for i in range(len(categorical_levels)):
        onehot.append(F.one_hot(x[:,i], categorical_levels[i]))  # One-hot-encode each of the categorical features separately. 
    
    x = torch.cat(onehot, dim = 1) # Concatenate the one-hot-vectors columnwise. 

    log_x = torch.log(x.float().clamp(min=1e-40)) # Take logarithm of the concatenated one-hot-vectors. 

    return log_x

def log_sub_exp(a, b):
    """Subtraction in log-space does not exist in Pytorch by default. This is the same as logaddexp(), but with subtraction instead."""
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

def sliced_logsumexp(x, slices):
    """Function copied from TabDDPM implementation. Do not understand what it does yet! This is used in the final step in theta_post()."""
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')), # add -inf as a first column to x. 
        dim=-1) # Then take the logarithm of the cumulative summation of the exponent of the elements in x along the columns. 

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    #slice_lse = torch.logaddexp(lse[:, slice_ends], lse[:, slice_starts]) # Add the offset values of "one difference in index" together in log-space.
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts]) # Subtract the offset values of "one difference in index" in log-space. 
                                                                      # This is essentially doing a torch.logsumexp() of each feature (individually) at once, 
                                                                      # like they do for one feature in Hoogeboom et al. implementation.
                                                                      # This works because of the cumulative sums 
                                                                      # E.g. for feature 3 we take cumsum of all columns up to last level in feature 3
                                                                      # and subtract the sumsum of all columns up to first level in feature 3
                                                                      # ==> this is the logsumexp() of all columns in feature 3.
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        torch.from_numpy(slice_ends - slice_starts), 
        dim=-1
    ) # This function call copies the values from slice_lse columnwise a number of times corresponding to the number of levels in each categorical variable. 
    return slice_lse_repeated

class MultinomialDiffusion(nn.Module):
    """Class for Multinomial diffusion, for all the categorical features in the data set."""

    # Denne fungerer vel kanskje bare for én kategorisk feature om gangen?

    def __init__(self, categorical_feature_names, categorical_levels, T, schedule_type, device):
        super(MultinomialDiffusion, self).__init__()
        self.categorical_feature_names = categorical_feature_names
        self.categorical_levels = categorical_levels
        self.num_categorical_variables = len(categorical_levels) 
        assert len(categorical_levels) == len(categorical_feature_names), \
                            f"Categorical levels {categorical_levels} and features names {categorical_feature_names} must be two lists of same length."
        
        self.num_classes_extended = torch.from_numpy(
            np.concatenate([np.repeat(self.categorical_levels[i], self.categorical_levels[i]) for i in range(len(self.categorical_levels))])
        ).to(device)

        self.slices_for_classes = [[] for i in range(self.num_categorical_variables)]
        self.slices_for_classes[0] = np.arange(self.categorical_levels[0])
        self.offsets = np.cumsum(self.categorical_levels)
        for i in range(1,self.num_categorical_variables):
            self.slices_for_classes[i] = np.arange(self.offsets[i-1], self.offsets[i])
        self.offsets = np.append([0], self.offsets) # Add a zero to the beginning of offsets. This is such that sliced_logsumexp will work correctly. 

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

        # Logarithmic versions for Multinomial diffusion in log space.
        log_alphas = np.log(alphas)
        log_one_minus_alphas = np.log(1 - np.exp(log_alphas) + 1e-40) # Add a small offset for numerical stability.
        log_alpha_bar = np.log(alpha_bar)
        log_one_minus_alpha_bar = np.log(1 - np.exp(log_alpha_bar) + 1e-40) # Add small offset.

        beta_tilde = betas * (1.0 - alpha_bar_prev)/(1.0 - alpha_bar) # Equation 7 in DDPM.
        mu_tilde_coef1 = np.sqrt(alpha_bar_prev)*betas / (1.0 - alpha_bar) # Equation 7 in DDPM. 
        mu_tilde_coef2 = np.sqrt(alphas)*(1.0 - alpha_bar_prev) / (1.0 - alpha_bar) # Equation 7 in DDPM. 
        
        # Make partial function to make Pytorch tensors with dtype float32 when registering the buffers of each variable.
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Parameters for Gaussian diffusion++.
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

        # Parameters for Multinomial diffusion.
        self.register_buffer("log_alphas", to_torch(log_alphas).to(self.device))
        self.register_buffer("log_one_minus_alphas", to_torch(log_one_minus_alphas).to(self.device))
        self.register_buffer("log_alpha_bar", to_torch(log_alpha_bar).to(self.device))
        self.register_buffer("log_one_minus_alpha_bar", to_torch(log_one_minus_alpha_bar).to(self.device))

        # Parameters for forward posterior. 
        self.register_buffer("beta_tilde", to_torch(beta_tilde).to(self.device))
        self.register_buffer("mu_tilde_coef1", to_torch(mu_tilde_coef1).to(self.device))
        self.register_buffer("mu_tilde_coef2", to_torch(mu_tilde_coef2).to(self.device))
        
    def noise_one_step(self, log_x_t_1, t):
        """Noise x_{t-1} to x_t, following Equation 11 Hoogeboom et. al.

        q(x_t|x_{t-1}).
        
        Returns the log of the new probability that is used in the categorical distribution to sample x_t.
        """
        log_alpha_t = extract(self.log_alphas, t, log_x_t_1.shape)
        log_one_minus_alpha_t = extract(self.log_one_minus_alphas, t, log_x_t_1.shape)

        log_prob = torch.logaddexp(log_alpha_t + log_x_t_1, log_one_minus_alpha_t - torch.log(self.num_classes_extended))
        return log_prob

    def noise_data_point(self, log_x_0, t):
        """Returns the log of the new probability that is used in the categorical distribution to sample x_t, 

        in q(x_t|x_0) from Equation 12 in Hoogeboom et al. 
        """
        log_alpha_bar_t = extract(self.log_alpha_bar, t, log_x_0.shape)
        log_one_minus_alpha_bar_t = extract(self.log_one_minus_alpha_bar, t, log_x_0.shape)

        log_prob = torch.logaddexp(log_alpha_bar_t + log_x_0, log_one_minus_alpha_bar_t - torch.log(self.num_classes_extended))
        return log_prob

    def theta_post(self, log_x_t, log_x_0, t):
        """This is the probability parameter in the posterior categorical distribution, called theta_post by Hoogeboom et. al."""

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        #t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        # TabDDPM implementation: Here they removed all negative zeros from t-1, i.e. set them equal to zero. 
        # Added the code above, could try adding this as well, to see if it means anything for the performance of the model. 
        log_probs_x_t_1 = self.noise_data_point(log_x_0, t_minus_1) # This is [\bar \alpha_{t-1} x_0 + (1-\bar \alpha_{t-1})/K].

        # Variables used to distinguish between t = 0 and t != 0 (in a vectorized way).
        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        
        # If t == 0 log_probs_x_t_1 should be set to log_x_0, else to self.noise_data_point(log_x_0, t-1).
        log_probs_x_t_1 = torch.where(t_broadcast == 0, log_x_0, log_probs_x_t_1.to(torch.float32)) # We also make sure the datatype is compatible. 

        log_tilde_theta = log_probs_x_t_1 + self.noise_one_step(log_x_t, t) # This is the entire \tilde{\theta} probability vector.

        normalizing_constant = sliced_logsumexp(log_tilde_theta, self.offsets) 

        normalized_log_tilde_theta = log_tilde_theta - \
                normalizing_constant # This is log_theta_post. 
        return normalized_log_tilde_theta

    def reverse_pred(self, model_pred, log_x_t, t):
        """Returns the probability parameter of the categorical distribution of the backward process, based on the predicted x_0 from the neural net."""
        #hat_x_0 = self.model(log_x_t,t) # Predict x_0 from the model. We keep the one-hot-encoding of log_x_t and feed it like that into the model. 

        assert model_pred.size(0) == log_x_t.size(0) # Make sure the number of observations are conserved. 
        assert model_pred.size(1) == sum(self.categorical_levels), f'{model_pred.size()}' # Make sure the number of feature columns (one-hot) are conserved.

        # Need to do softmax over only the levels corresponding to each categorical variable at once. 
        # Therefore we need to loop over the collection of indices of each feature column after one-hot-encoding, 
        # and perform log-softmax only over each set of indices at the same time. 
        log_hat_x_0 = torch.empty_like(model_pred)
        for ix in self.slices_for_classes:
           log_hat_x_0[:, ix] = F.log_softmax(model_pred[:, ix], dim=1) # log_softmax: find the logarithm of softmax of hat_x_0 (the prediction from the model).

        # This is what I had before (left just for testing now).
        #log_hat_x_0 = F.log_softmax(model_pred, dim = 1)

        # All the above is contained in "predict_start" in TabDDPM implementation. 
    
        log_tilde_theta_hat = self.theta_post(log_x_t, log_hat_x_0, t) # Find the probability parameter of the reverse process, based on the prediction from the neural net. 
        return log_tilde_theta_hat

    def forward_sample(self, log_x_0, t):
        """Sample a new data point from q(x_t|x_0). Returns log x_t (noised input x_0 at times t, in log space), 
        
        following the closed form Equation 12 in Hoogeboom et al. q(x_t|x_0).
        """
        # First we get the probability parameter. 
        log_prob = self.noise_data_point(log_x_0, t)
        # Then we sample from the categorical. 
        log_x_t = self.log_sample_categorical(log_prob)
        return log_x_t

    def sample(self, model, n):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper, modified for multinomial diffusion.
        """
        print("Entered function for sampling.")
        model.eval()
        x_list = {}
        with torch.no_grad():
            uniform_sample = torch.zeros((n, len(self.num_classes_extended)), device=device) # I think this could be whatever number, as long as all of them are equal!         
                        # Sjekk om dette stemmer og sjekk hva denne faktisk gjør!
            log_x = self.log_sample_categorical(uniform_sample).to(device) # The sample at T is uniform (sample from x_T).
            
            for i in reversed(range(self.T)): # I start it at 0
                if i % 25 == 0:
                    print(f"Sampling step {i}.")
                #x_list[i] = torch.exp(log_x)
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                
                # Predict x_0 using the neural network model.
                log_x_hat = model(log_x,t) # Does this return as logarithm or not? This is an important detail!
                # And should it take x in log space as input or not? Need to do this in the same way as during training!

                # Get reverse process probability parameter. 
                log_tilde_theta_hat = self.reverse_pred(log_x_hat, log_x, t)

                # Sample from a categorical distribution based on theta_post. 
                # For this we use the gumbel-softmax trick. We put this in another function.
                log_x = self.log_sample_categorical(log_tilde_theta_hat)

        x = torch.exp(log_x)
        model.train() # Indicate to Pytorch that we are back to doing training. 
        return x, x_list # We return a list of the x'es to see how they develop from t = 99 to t = 0.

    def sample_categorical(self, log_probs):
        """Sample from a categorical distribution using the gumbel-softmax trick."""
        # This needs to be changed such that it works for all levels of all categorical variables. 
        uniform = torch.rand_like(log_probs) # Why logits (log(p/(1-p))) and not log_prob?
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + log_probs).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes) # Want to check how the index_to_log_onehot function actually works!
        return log_sample

    def log_sample_categorical(self, log_probs):
        """Heavily inspired by the version of this function in the implementation of TabDDPM."""
        # We need to treat each categorical variable as if it follows its own categorical distribution, with probabilities summing to one.
        # Because of this, we need to sample iteratively from each part of the input log_probs, corresponding to the one-hot-encoded columns pertaining to each categorical variable.  
        full_sample = []
        for i in range(self.num_categorical_variables):
            log_probs_one_cat_var = log_probs[:, self.slices_for_classes[i]] # Select only the one-hot-encoded columns pertaining to categorical variable i. 
            uniform = torch.rand_like(log_probs_one_cat_var) 
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-40) + 1e-40)
            sample = (gumbel_noise + log_probs_one_cat_var).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1)) # The unsqueeze inserts another column. This is needed to correctly concatenate after the loop. 
        full_sample = torch.cat(full_sample, dim=1) # Add all the samples to the same tensor by concatenating column-wise. 
        log_sample = index_to_log_onehot(full_sample, self.categorical_levels) # Transform back to log of one-hot-encoded data. 
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
        t = torch.randint(low=0, high=self.T, size = (n,)).to(self.device)
        pt = torch.ones_like(t).float() / self.T
        return t, pt

    def categorical_kl(self, log_prob_a, log_prob_b):
        """Calculate the KL divergence between log_prob_a and log_prob_b, following the definition of KL divergence for discrete quantities.
        
        The definition states
        D(p_a || p_b) = \sum p_a * log(p_a/p_b) = \sum p_a * (log_prob_a - log_prob_b).
        """
        return (log_prob_a.exp() * (log_prob_a - log_prob_b)).sum(dim = 1)

    def kl_prior(self, log_x_start):
        """Some prior that is added to the loss, while the other loss is upweighted by 1/T. 
        
        This is copied directly from TabDDPM, in order to see if it greatly affects the performance. 
        """
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.noise_data_point(log_x_start, t=(self.T - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_extended * torch.ones_like(log_qxT_prob))

        kl_prior = self.categorical_kl(log_qxT_prob, log_half_prob)
        return kl_prior

    def loss(self, log_x_0, log_x_t, log_hat_x_0, t, pt):
        """Function to return the loss. This loss represents each term L_{t-1} in the ELBO of diffusion models.
        
        KL( q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t) ) = KL( Cat(\pi(x_t,x_0)) || Cat(\pi(x_t, \hatx_0)) ).

        We also need to compute the term log p(x_0|x_1) if t = 1. This is done via a mask below.   

        Parameters: 
            log_x_0 is the logarithm of (one-hot-encoded) true starting data. 
            log_x_t is the logarithm of (one-hot-encoded) true noisy x_0 at time t.
            log_hat_x_0 is the logarithm of (one-hot-encoded) predicted starting data, using the model. 
            t is the vector of time steps     
        """
        # In the loss in Hoogeboom et al. and TabDDPM they use a kl_prior as well. 
        # I do not understand quite what this is, so we skip this for now. 
        # THE MODEL IS NOT PERFORMING VERY WELL! TRY ADDING THE PRIOR AND SEE IF IT DOES ANYTHING!?
        # The prior does not increase the performance (seems like it is pretty much the same with and without)
        
        kl_prior = self.kl_prior(log_x_0) # I really do not understand what this calculates.
        # Does not seem like it changes the performance (looks the same as when it is not being used I think. )

        # Find the true theta post, i.e. based on the true log_x_0.
        log_true_theta = self.theta_post(log_x_t, log_x_0, t)

        # Find the predicted theta post, i.e. based on the predicted log_x_0 based on the neural network. 
        log_predicted_theta = self.reverse_pred(log_hat_x_0, log_x_t, t)

        # Calculate the KL divergence between the categorical distributions with probability parameters true theta post and predicted theta post. 
        lt = self.categorical_kl(log_true_theta, log_predicted_theta)

        # Make mask for t == 0, where we need a different calculation for log(p(x_0|x_1)). We call this different calculation the decoder_loss.
        mask = (t == torch.zeros_like(t)).float() # If t == 0, we calculate sum x_0*log \hatx_0 over all classes (columns). Else we return L_{t-1}.

        decoder_loss = -(log_x_0.exp() * log_predicted_theta).sum(dim=1) # Dette kan vel umulig stemme? Det burde vel være log(\hatx_0) i andre ledd? (og ikke theta_post(x_t,\hatx_0)).

        loss = mask * decoder_loss + (1. - mask) * lt
        loss = loss / pt + kl_prior # Upweigh the "first loss" in the same way as in TabDDPM. 
        return  torch.mean(loss)# We take the mean such that we return a scalar, which we can backprop through.
                        # Use nanmean() since we have nans in the loss! How can we deal with this?
                #torch.nanmean(loss)

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
        args = timesteps[:, None].float() * freqs[None] # freqs[None] adds a row dimension to the vector freqs. 
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, t):
        """Forward steps for Pytorch."""
        # First we make the embeddings. 
        t_emb = self.time_embed(self.timestep_embedding(t)) # Embeds the time into 128-dimensions using sinusoidal positional embeddings. 
                                                            
        x = x.to(torch.float32) # Change the data type here for now, quick fix since the weights of proj are float32 by default.                                                     
        x_emb = self.proj(x) # Embeds x into 128-dimensions using a linear layer.
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
    input_size = X_train.shape[1] # Columns in the training data is the input size of the neural network model. 

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
    diffusion = MultinomialDiffusion(categorical_feature_names, categorical_levels, T, schedule, device)

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
            log_inputs = torch.log(inputs.float().clamp(min=1e-40)) # We effectively change all zeros to a very small number using "clamp", to avoid log(0).

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t, pt = diffusion.sample_timesteps(inputs.shape[0])

            # Noise the inputs and return the noised input in log_space.
            log_x_t = diffusion.forward_sample(log_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Make prediction with the model. In the mixed model, both the numerical and categorical features should be concatenated as input to the model.
            log_hat_x_0 = model(log_x_t, t)

            # Calculate the loss between the noised input and the true input. 
            # In order to do this we need to feed the loss function with true inputs, true noised inputs and predicted inputs from noise. 
            loss = diffusion.loss(log_inputs, log_x_t, log_hat_x_0, t, pt) 

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
            log_inputs = torch.log(inputs.float().clamp(min=1e-40))

            # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
            t, pt = diffusion.sample_timesteps(inputs.shape[0])

            # Noise the inputs and return the noised input in log_space.
            log_x_t = diffusion.forward_sample(log_inputs, t) # x_t is the noisy version of the input x, at time t.

            # Make prediction with the model. In the mixed model, both the numerical and categorical features should be concatenated as input to the model.
            log_hat_x_0 = model(log_x_t, t)

            # Calculate the loss between the noised input and the true input. 
            # In order to do this we need to feed the loss function with true inputs, true noised inputs and predicted inputs from noise. 
            loss = diffusion.loss(log_inputs, log_x_t, log_hat_x_0, t, pt) 
            
            valid_loss += loss.item() # Calculate total validation loss over the entire epoch.
        #########################         

        training_losses[epoch] = train_loss
        validation_losses[epoch] = valid_loss
        # We do not divide the validation loss by the number of validation batches, since we validate on the entire validation set at once. 
        
        print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}. Validation loss after epoch {epoch+1} is {valid_loss:.4f}.")
        
        # Saving models each time the validation loss reaches a new minimum.
        if min_valid_loss > valid_loss:
            print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
            
            min_valid_loss = valid_loss # Set new minimum validation loss. 

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
numerical_features = [] # Give the Data-class an empty list of numerical features to indicate that we only work with the categorical features.
                        # Should check if this works as I expect!!

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

# X_train = X_train.iloc[[0]] # Follow one sample through the process to see how it works and try to understand why it does not work well. 
# X_test = X_test.iloc[[0]]

# training_losses, validation_losses = train(X_train, y_train, X_test, y_test, categorical_features, 
#                         lens_categorical_features, device, T = 100, 
#                         schedule = "linear", batch_size = 4096, num_epochs = 100, 
#                         num_mlp_blocks = 4, dropout_p = 0.0)

# plot_losses(training_losses, validation_losses)

# Try to evaluate the model.
def evaluate(n, generate = True, save_figs = False, make_mosaic = False): 
    """Try to see if we can sample synthetic data from the Gaussian Diffusion model."""

    # Load the previously saved models.
    model = NeuralNetModel(X_train.shape[1], 4, 0.0).to(device)
    model.load_state_dict(torch.load("./MultNeuralNet.pth"))
    diffusion = MultinomialDiffusion(categorical_features, lens_categorical_features, 1000, "linear", device)
    diffusion.load_state_dict(torch.load("./MultDiffusion.pth")) 
    # Don't think it is necessary to save and load the diffusion model!
    # We still do it to be safe. 

    with torch.no_grad():
        model.eval()
        diffusion.eval()    

        # Run the noise backwards through the backward process in order to generate new data. 
        if generate:
            synthetic_samples, reverse_points_list = diffusion.sample(model, n)
            synthetic_samples = synthetic_samples.cpu().numpy()
            synthetic_samples = pd.DataFrame(synthetic_samples, columns = X_train.columns.tolist())
            synthetic_samples = Adult.decode(synthetic_samples)
            synthetic_samples.to_csv("synthetic_sample_mult_diff.csv")
        else:
            # Load the synthetic sample we already created. 
            synthetic_samples = pd.read_csv("synthetic_sample_mult_diff.csv", index_col = 0)

        print(synthetic_samples.shape)
        print(synthetic_samples.head())

        def visualize_categorical_data(synthetic_data, real_data):
            """Plot barplots and mosaic plots of the synthetic data against the real training data for categorical features."""
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

            plt.show()

            # Make mosaic plots later if I feel like it!
            # E.g. https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib

        def mosaic_plot(df, features, title):
            """Make some mosaic plots to look at correlations between the categorical variables."""
            assert len(features) == 2, "Make sure you give the function two features."
            labels = lambda k: ""
            mosaic(df, features, title = title, labelizer = labels, label_rotation = [45,0])

        def calculate_theils_U(df):
            """Calculate Theil's U Statistic between the categorical features.
            
            https://en.wikipedia.org/wiki/Uncertainty_coefficient

            I do not know how they do it in TabDDPM, but I will do it my own way here. 
            """
            return theils_u_matrix(df)

        # Visualize again after descaling.
        #visualize_categorical_data(synthetic_samples, Adult.get_training_data()[0][categorical_features])
        if save_figs:
            plt.savefig("synthetic_mult_diff.pdf")

        if make_mosaic: 
            features = ["race", "relationship"]
            mosaic_plot(synthetic_samples.sort_values(features), features, title = "Synth.")
            mosaic_plot(Adult.get_training_data()[0][categorical_features].sort_values(features),features, title = "OG.")
            plt.show()

            features = ["sex", "race"]
            mosaic_plot(synthetic_samples.sort_values(features), features, title = "Synth.")
            mosaic_plot(Adult.get_training_data()[0][categorical_features].sort_values(features),features, title = "OG.")
            plt.show()

            features = ["sex", "relationship"] # This one did not get sorted properly it seems like. Not very useful then.
            mosaic_plot(synthetic_samples.sort_values(features), features, title = "Synth.")
            mosaic_plot(Adult.get_training_data()[0][categorical_features].sort_values(features),features, title = "OG.")
            plt.show()
    
        synth2 = synthetic_samples.copy()
        synth2[categorical_features] = synth2[categorical_features].apply(lambda col:pd.Categorical(col).codes)   
        matrix = torch.tensor(synth2.values) 
        synth_corr = calculate_theils_U(matrix)

        # The function below needs to be changed to fit the categorical data!
        def look_at_reverse_process_steps(x_list, T):
            """We plot some of the steps in the backward process to visualize how it changes the data."""
            times = [0, int(T/5), int(2*T/5), int(3*T/5), int(4*T/5), T-1][::-1] # Reversed list of the same times as visualizing forward process. 

            for i, feat in enumerate(categorical_features):
                fig, axs = plt.subplots(2,3)
                axs = axs.ravel()
                for idx, ax in enumerate(axs):
                    # Do necessary transforms before plotting. 
                    vals = x_list[times[idx]].cpu().numpy()
                    vals = pd.DataFrame(vals, columns = X_train.columns.tolist())
                    vals = Adult.decode(vals)

                    # Plot. 
                    (vals.iloc[:,i].value_counts()/vals.shape[0]).plot(kind='bar', ax = ax)
                    ax.set_xlabel(f"Time {times[idx]}")
                    ax.xaxis.set_ticklabels([])
                fig.suptitle(f"Feature '{feat}'")
                plt.tight_layout()
            plt.show()
        
        #look_at_reverse_process_steps(reverse_points_list, diffusion.T)
        #print(reverse_points_list)

evaluate(X_train.shape[0], generate=False, save_figs=False, make_mosaic = False)

# The function below needs to be changed to fit for the categorical data!
def check_forward_process(X_train, y_train, T, schedule, device, batch_size = 1, mult_steps = False):
    """Check if the forward diffusion process in Gaussian diffusion works as intended."""
    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

    diffusion = MultinomialDiffusion(categorical_features, lens_categorical_features, T, schedule, device)

    inputs, _ = next(iter(train_loader)) # Check for first batch. 

    inputs = inputs.to(device)

    # Assuming the data is already one-hot-encoded, we do a log-transformation of the data. 
    log_inputs = torch.log(inputs.float().clamp(min=1e-30))

    if mult_steps:
        # If we want to visualize the data in several steps along the way. 
        times = [0, int(T/5), int(2*T/5), int(3*T/5), int(4*T/5), T-1] # The six times we want to visualize.
        x_T_dict = {}
        for i, time in enumerate(times):
            log_x_T = diffusion.forward_sample(log_inputs, torch.tensor([times[i]]))
            x_T = log_x_T.exp()
            #x_T = diffusion.noise_data_point(inputs, torch.tensor([times[i]])) # This is one-hot-encoded.
            x_T = x_T.cpu().numpy()
            x_T = pd.DataFrame(x_T, columns = X_train.columns.tolist()) # Perhaps I need to make dataframe first!? Not sure. 
            x_T = Adult.decode(x_T)
            x_T_dict[i] = x_T
    else:
        # If we only want to visualize the data in the last latent variable. 
        log_x_T = diffusion.forward_sample(log_inputs, torch.tensor([T-1])) # We have to give it a tensor with the index value of the last time step. 
        x_T = log_x_T.exp() 
        x_T = x_T.cpu().numpy()
        x_T = pd.DataFrame(x_T, columns = X_train.columns.tolist()) # Perhaps I need to make dataframe first!? Not sure. 
        x_T = Adult.decode(x_T)
    
    inputs = inputs.cpu().numpy()
    inputs = pd.DataFrame(inputs, columns = X_train.columns.tolist()) # Perhaps I need to make dataframe first!? Not sure. 
    inputs = Adult.decode(inputs) # Reverse one-hot-encode the inputs. 
    
    # Plot the categorical features after forward diffusion together with the original data. 
    if mult_steps: 
        for i, feat in enumerate(categorical_features):
            fig, axs = plt.subplots(2,3)
            axs = axs.ravel()
            for idx, ax in enumerate(axs):
                x_T_dict[idx].iloc[:,i].value_counts().plot(kind='bar', ax = ax)
                ax.set_xlabel(f"Time {times[idx]}")
                ax.xaxis.set_ticklabels([])
            fig.suptitle(f"Feature '{feat}'")
            plt.tight_layout()
        plt.show()
    else: 
        fig, axs = plt.subplots(2,2)
        axs = axs.ravel()
        for idx, ax in enumerate(axs):
            x_T.iloc[:,idx].value_counts().plot(kind = "bar", ax = ax, color = "blue", label = "Synth.")
            inputs.iloc[:,idx].value_counts().plot(kind = "bar", ax = ax, color = "orange", alpha = 0.7, label = "OG.")
            ax.legend()
            ax.xaxis.set_ticklabels([])
            ax.title.set_text(categorical_features[idx])
        plt.tight_layout()

        # Make two grids since 7 is not an even number of categorical features. 
        fig, axs2 = plt.subplots(2,2)
        axs2 = axs2.ravel()
        for idx, ax in enumerate(axs2, start = 4):
            if idx > len(categorical_features)-1:
                break
            x_T.iloc[:,idx].value_counts().plot(kind = "bar", ax = ax, color = "blue", label = "Synth.")
            inputs.iloc[:,idx].value_counts().plot(kind = "bar", ax = ax, color = "orange", alpha = 0.7, label = "OG.")
            ax.legend()
            ax.xaxis.set_ticklabels([])
            ax.title.set_text(categorical_features[idx])
        plt.tight_layout()
        plt.show()

# The forward process seems to work fine for both schedules! Nice!
#check_forward_process(X_train, y_train, T = 1000, schedule = "cosine", device = device, batch_size = X_train.shape[0], mult_steps=True)
#check_forward_process(X_train, y_train, T = 1000, schedule = "cosine", device = device, batch_size = X_train.shape[0])

def plot_schedules(T, device):
    """Check if the schedules make sense (compare to plot in Improved DDPMs by Nichol and Dhariwal)."""

    diffusion_linear = MultinomialDiffusion(categorical_features, lens_categorical_features, T, "linear", device)
    diffusion_cosine = MultinomialDiffusion(categorical_features, lens_categorical_features, T, "cosine", device)
    
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
#plot_schedules(T = 1000, device = device)
