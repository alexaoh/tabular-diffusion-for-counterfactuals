# Class for Multinomial diffusion.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from utils import extract, log_sub_exp, sliced_logsumexp, index_to_log_onehot

class Multinomial_diffusion(nn.Module):
    """Class for Multinomial diffusion. Handles only categorical features.
    
    Parameters
    ----------
    categorical_feature_names : list
        List of names of the categorical features in the data set.
    categorical_levels: list
        List of number of levels of each categorical feature.
        Should be in the same order as 'categorical_feature_names'.
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
    def __init__(self, categorical_feature_names, categorical_levels, T, schedule_type, device):
        super(Multinomial_diffusion, self).__init__()
        self.categorical_feature_names = categorical_feature_names
        self.categorical_levels = categorical_levels
        self.num_categorical_variables = len(categorical_levels) 
        assert len(categorical_levels) == len(categorical_feature_names), \
                            f"Categorical levels {categorical_levels} and features names {categorical_feature_names} must be two lists of same length."
        
        self.num_classes_extended = torch.from_numpy(
            np.concatenate([np.repeat(self.categorical_levels[i], self.categorical_levels[i]) for i in range(len(self.categorical_levels))])
        ).to(device)

        slices_for_classes = [[] for i in range(self.num_categorical_variables)]
        slices_for_classes[0] = np.arange(self.categorical_levels[0])
        offsets = np.cumsum(self.categorical_levels)
        for i in range(1,self.num_categorical_variables):
            slices_for_classes[i] = np.arange(offsets[i-1], offsets[i])
        self.slices_for_classes = slices_for_classes
        offsets = np.append([0], offsets) # Add a zero to the beginning of offsets. This is such that sliced_logsumexp will work correctly. 
        self.offsets = torch.from_numpy(offsets).to(device)

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

        # TabDDPM implementation: Here they removed all negative zeros from t-1, i.e. set them equal to zero. 
        log_probs_x_t_1 = self.noise_data_point(log_x_0, t-1) # This is [\bar \alpha_{t-1} x_0 + (1-\bar \alpha_{t-1})/K].

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

    def sample(self, model, n, y_dist=None):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper, modified for multinomial diffusion.
        """
        print("Entered function for sampling.")
        model.eval()

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
            )
        with torch.no_grad():
            uniform_sample = torch.zeros((n, len(self.num_classes_extended)), device=device) # I think this could be whatever number, as long as all of them are equal!         
                        # Sjekk om dette stemmer og sjekk hva denne faktisk gjør!
            log_x = self.log_sample_categorical(uniform_sample).to(device) # The sample at T is uniform (sample from x_T).
            
            for i in reversed(range(self.T)): # I start it at 0
                if i % 25 == 0:
                    print(f"Sampling step {i}.")

                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                
                # Predict x_0 using the neural network model.
                log_x_hat = model(log_x, t, y) # Does this return as logarithm or not? This is an important detail!
                # And should it take x in log space as input or not? Need to do this in the same way as during training!

                # Get reverse process probability parameter. 
                log_tilde_theta_hat = self.reverse_pred(log_x_hat, log_x, t)

                # Sample from a categorical distribution based on theta_post. 
                # For this we use the gumbel-softmax trick. We put this in another function.
                log_x = self.log_sample_categorical(log_tilde_theta_hat)

        x = torch.exp(log_x)
        model.train() # Indicate to Pytorch that we are back to doing training. 
        return x, y.reshape(-1,1) # We return x and y. 

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
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + log_probs_one_cat_var).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1)) # The unsqueeze inserts another column. Not sure why this is needed, but have a look through the debugger later. 
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