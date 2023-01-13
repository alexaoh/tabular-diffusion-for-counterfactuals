# Simple code for DDPM, based on 
# the youtube video https://www.youtube.com/watch?v=TBCRlnwJtZU

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using '{device}' device.")

# Set seeds for reproducibility. 
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level = logging.INFO, datefmt="%I:%M:%S")

class Diffusion():
    """These are the diffusion tools (not including DNN parameterization, i.e. the encoder, decoder, etc)."""
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, device = device):
        #super(DDPM, self).__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(self.device) # Noise schedule. 
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)

    def prepare_noise_schedule(self):
        """Return linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_data(self, x, t):
        """Noise the data x at step t (without going through the entire forward process). Returns noised data point and noise."""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]) #[:, None, None, None] # I think this is for the images perhaps. 
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t]) #[:, None, None, None] # I think this is for the images perhaps. 
        epsilon = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon

    def sample_timesteps(self, n):
        """Sample timesteps for use when training the model."""
        return torch.randint(low=1, high=self.noise_steps, size = (n,))

    def sample(self, model, n):
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 2 in DDPM-paper.
        """
        logging.info(f"Sampling {n} new data points...")
        model.eval() # Indicate to Pytorch that we are doing inference, not training. De-activated dropout layers for example. 
        with torch.no_grad():
            x = torch.randn((n,)).to(self.device) # Sample from standard Gaussian (sample from x_T). We are not doing it for images, but for data points. This is only 1D for now. 
            # For tabular data I probably need (n,2) I think! Test later. 
            for i in reversed(range(1, self.noise_steps)): # Could add progress bar using tqdm here, as in the video.
                t = (torch.ones(n) * i).to(torch.int64).to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alpha[t] #[:, None, None, None] # I think this is for the images perhaps. 
                alpha_bar = self.alpha_bar[t] #[:, None, None, None] # I think this is for the images perhaps. 
                beta = self.beta[t] #[:, None, None, None] # I think this is for the images perhaps. 
                if i > 1:
                    noise = torch.rand_like(x)
                else: # We don't want to add noise at t = 1, because it would make our outcome worse. 
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1.0-alpha_bar)))*predicted_noise) + torch.sqrt(beta) * noise # Use formula in line 4 in Algorithm 2.

        model.train() # Indicate to Pytorch that we are back to doing training. 

        # The video does some data type and range changes here. Not sure if I should do any of those. 

        return x


# Dummy data in 1D. Try with tabular data later. 
dummy = np.linspace(0,10,5)
torch_dummy = torch.from_numpy(dummy.astype(np.float32)).to(device)


#### Not sure how I want to define my model yet!

class NeuralNetModel(nn.Module):
    """Main model for encoding/decoding etc."""

    # I guess I need some sort of encoder or decoder here, in addition to some utils. 

    def __init__(self, input_size):
        super(NeuralNetModel, self).__init__()
        self.input_size = input_size

        # Layers.
        self.l1 = nn.Linear(input_size, 15)
        self.l2 = nn.Linear(15, 10)
        self.l3 = nn.Linear(10, 15)
        self.l4 = nn.Linear(15, input_size)
        
        # Activation functions. 
        self.relu = nn.ReLU()


    def forward(self, x, t):
        """Forward steps for Pytorch."""
        # I simply build some sort of autoencoder/bottleneck to try things out at first. 
        # This should be modified to account for two different types of output eventually (with two different losses in the training perhaps).
        
        # I also need to embed the time step t into the forward process somehow! Not sure how to do this?!
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out


def train(training_data, num_epochs = 10):
    """Training loop. Follows Algorithm 1 from DDPM."""
    input_size = training_data.shape[1] # Columns in the training data is the input size of the neural network model. 
    num_epochs = num_epochs


    # Make trainind data set with Dataset from Pytorch.
    
    # Make train_loader dataloader from Pytorch. 
    
    diffusion = Diffusion(noise_steps = 10, beta_start = 1e-4, beta_end = 0.02, device = device)
    model = NeuralNetModel(input_size).to(device)

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}: ")
        # Could add a progress bar here as well using tqdm.
        for i, (inputs,) in enumerate(train_loader):
            
            # Load data onto device.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Sample random timesteps between 1 and noise_steps for diffusion process. 
            t = diffusion.sample_timesteps(inputs.shape[0]).to(device)

            # Noise the inputs.
            x_t, noise = diffusion.noise_data(inputs, t)

            # Feed the noised data and the time step to the model, which then predicts the noise at that time. 
            predicted_noise = model(x_t, t)

            # Calculate MSE loss between predicted noise and real noise.
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward() # Calculate gradients. 
            optimizer.step() # Update parameters. 
