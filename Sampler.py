# General class for sampling from Gaussian_diffusion, Multinomial_diffusion and Gaussian_multinomial_diffusion models. 

import pandas as pd
import numpy as np
import torch

from utils import extract, sliced_logsumexp, index_to_log_onehot


class Sampler():
    """Sampler for diffusion models. This is a base class that the more specific samplers inherit from.
    
    Parameters
    ---------
    model : Neural network that has been trained. 
        NeuralNet Object we want to use to sample. 
    data_object : Object of type Data.


    Methods
    -------
    sample : 
        Main function to call when you want to sample from the diffusion model. 
    save_synthetics :
        Saves the synthetically sampled data to harddrive. 
    """
    
    def __init__(self, model, data_object, data_code):
        self.model = model
        self.data_object = data_object
        self.data_code = data_code # Used for saving the trained models. 

    def sample(self, n):
        """Generate synthetic data from the diffusion model with use of the Neural_net."""
        pass
    
    def save_synthetics(self):
        """Save the synthetic, newly generated data to the harddrive as csv."""
        original_data = self.data_object.get_training_data_preprocessed()[0]
        columns = original_data.columns.tolist() # Do not add the response to the columns. 
        if self.model.is_class_cond:
            columns += ["y"] # If the model is class conditional, add the column name "y".
            self.synthetic_samples[:,-1] = self.synthetic_samples[:,-1].to(torch.int) # Make sure the responses are ints.
        self.synthetic_samples = self.synthetic_samples.cpu().numpy()
        self.synthetic_samples = pd.DataFrame(self.synthetic_samples, columns = columns)
        
        # For splitting the names of the files between conditional and joint distribution modelling.
        # "extra" = "joint" is added to the name when we model the joint distribution of the features in the dataset (including response).
        self.extra = ""
        if not self.model.is_class_cond:
            self.extra = "_joint"

class Gaussian_sampler(Sampler):
    """Sampler for Gaussian_diffusion."""

    def __init__(self, model, data_object, gaussian_diffusion, data_code):
        super().__init__(model, data_object, data_code)
        self.gaussian_diffusion = gaussian_diffusion

    def sample(self, n):
        """Generate synhetic data from the Gaussian diffusion model with use of the Neural_net."""
        y_dist = torch.tensor(list(self.data_object.get_proportion_of_response()))
        self._X, self._y = self.gaussian_diffusion.sample(self.model, n, y_dist)
        if self._y is not None:
            self.synthetic_samples = torch.cat((self._X, self._y), dim = 1)
        else:
            self.synthetic_samples = self._X

    def save_synthetics(self, savename = "Gaussian_diffusion"):
        """Save the synthetic, newly generated data to the harddrive as csv."""
        super().save_synthetics()
        synthetic_samples = self.data_object.descale(self.synthetic_samples)
        # Fix the dtypes if necessary!
        filename = "synthetic_data/"+self.data_code+"_"+savename+self.extra+str(self.model.seed)+".csv"
        synthetic_samples.to_csv(filename)
        print(f"Synthetics saved to file '{filename}'")

class Multinomial_sampler(Sampler):
    """Sampler for Multinomial_diffusion."""

    def __init__(self, model, data_object, multinomial_diffusion, data_code):
        super().__init__(model, data_object, data_code)
        self.multinomial_diffusion = multinomial_diffusion

    def sample(self, n):
        """Generate synhetic data from the Multinomial diffusion model with use of the Neural_net."""
        y_dist = torch.tensor(list(self.data_object.get_proportion_of_response()))
        self._X, self._y = self.multinomial_diffusion.sample(self.model, n, y_dist)
        if self._y is not None:
            self.synthetic_samples = torch.cat((self._X, self._y), dim = 1)
        else:
            self.synthetic_samples = self._X

    def save_synthetics(self, savename = "Multinomial_diffusion"):
        """Save the synthetic, newly generated data to the harddrive as csv."""
        super().save_synthetics()
        synthetic_samples = self.data_object.decode(self.synthetic_samples)
        # Fix the dtypes if necessary!
        filename = "synthetic_data/"+self.data_code+"_"+savename+self.extra+str(self.model.seed)+".csv"
        synthetic_samples.to_csv(filename)
        print(f"Synthetics saved to file '{filename}'")
        
class Gaussian_multinomial_sampler(Sampler):
    """Sampler for Gaussian_multinomial_diffusion."""

    def __init__(self, model, data_object, multinomial_diffusion, gaussian_diffusion, data_code):
        super().__init__(model, data_object, data_code)
        self.multinomial_diffusion = multinomial_diffusion
        self.gaussian_diffusion = gaussian_diffusion

    def sample(self, n):
        """Generate synhetic data from the joint Gaussian and Multinomial diffusion model with use of the Neural_net."""
        y_dist = torch.tensor(list(self.data_object.get_proportion_of_response()))

        print("Entered function for sampling from Gaussian_multinomial_diffusion.")
        self.model.eval()
        self.gaussian_diffusion.eval()
        self.multinomial_diffusion.eval()

        num_numerical_features = len(self.gaussian_diffusion.numerical_features)
        num_categorical_onehot_encoded_columns = len(self.multinomial_diffusion.num_classes_extended)

        device = self.gaussian_diffusion.device

        if self.model.is_class_cond:
            if y_dist is None:
                raise Exception("You need to supply the distribution of labels (vector) when the model is class-conditional.")
                # For example for "Adult": supply y_dist = torch.tensor([0.75, 0.25]), since about 25% of the data set are reported with positive outcome. 
        
        y = None
        if self.model.is_class_cond:
            y = torch.multinomial( # This makes sure we sample the classes according to their proportions in the real data set, at each step in the generative process. 
                y_dist,
                num_samples=n,
                replacement=True
            ).to(torch.int).to(device)

        with torch.no_grad():
            x_gauss = torch.randn((n,num_numerical_features)).to(device) # Sample from standard Gaussian (sample from x_T). 
            uniform_sample = torch.zeros((n, num_categorical_onehot_encoded_columns), device=device) # I think this could be whatever number, as long as all of them are equal!  
            log_x_mult = self.multinomial_diffusion.log_sample_categorical(uniform_sample).to(device) # The sample at T is uniform (sample from x_T).
            for i in reversed(range(self.gaussian_diffusion.T)): # I start it at 0.
                if i % 25 == 0:
                    print(f"Sampling step {i}.")
                x = torch.cat((x_gauss, log_x_mult), dim = 1)
                t = (torch.ones(n) * i).to(torch.int64).to(device)
                predictions = self.model(x, t, y)

                # Gaussian part. 
                predicted_noise_gauss = predictions[:,:num_numerical_features]
                sh = predicted_noise_gauss.shape

                betas = extract(self.gaussian_diffusion.betas, t, sh) 
                sqrt_recip_alpha = extract(self.gaussian_diffusion.sqrt_recip_alpha, t, sh)
                sqrt_recip_one_minus_alpha_bar = extract(self.gaussian_diffusion.sqrt_recip_one_minus_alpha_bar, t, sh) 
                
                # Version #2 of sigma.
                sigma = extract(torch.sqrt(self.gaussian_diffusion.beta_tilde), t, sh)

                if i > 0:
                    noise = torch.randn_like(x_gauss)
                else: # We don't want to add noise at t = 0, because it would make our outcome worse (this comes from the fact that we have another term in Loss for x_0|x_1, I believe).
                    noise = torch.zeros_like(x_gauss)
                x_gauss = sqrt_recip_alpha * (x_gauss - (betas * sqrt_recip_one_minus_alpha_bar)*predicted_noise_gauss) + sigma * noise # Use formula in line 4 in Algorithm 2.

                # Multinomial part. 
                predicted_log_x_mult = predictions[:,num_numerical_features:]
                log_tilde_theta_hat = self.multinomial_diffusion.reverse_pred(predicted_log_x_mult, log_x_mult, t) # Get reverse process probability parameter. 
                log_x_mult = self.multinomial_diffusion.log_sample_categorical(log_tilde_theta_hat) # Sample from a categorical distribution based on theta_post. 

            # Get final x (x_0, i.e. the sample we generated).
            x_mult = torch.exp(log_x_mult)
            x = torch.cat((x_gauss, x_mult), dim = 1)

        self.model.train() # Indicate to Pytorch that we are back to doing training. 
        self.gaussian_diffusion.train()
        self.multinomial_diffusion.train()
        
        self._X = x
        self._y = y
        if self._y is not None:
            self._y = y.reshape(-1,1)
            self.synthetic_samples = torch.cat((self._X, self._y), dim = 1)
        else: 
            self.synthetic_samples = self._X
    
    def save_synthetics(self, savename = "Gaussian_multinomial_diffusion"):
        """Save the synthetic, newly generated data to the harddrive as csv."""
        super().save_synthetics()
        synthetic_samples = self.data_object.descale(self.synthetic_samples)
        synthetic_samples = self.data_object.decode(synthetic_samples)
        # Fix the dtypes if necessary!
        filename = "synthetic_data/"+self.data_code+"_"+savename+self.extra+str(self.model.seed)+".csv"
        synthetic_samples.to_csv(filename)
        print(f"Synthetics saved to file '{filename}'")
