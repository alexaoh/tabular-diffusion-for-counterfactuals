# Neural network used to parameterize reverse process in 
# Gaussian_diffusion, Multinomial_diffusion and Gaussian_multinomial_diffusion models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Neural_net(nn.Module):
    """Neural network for parameterizing reverse process in diffusion models.
    
    Parameters
    ----------
    input_size : int
        Number of input nodes.
    num_mlp_blocks : int
        Number of MLP Blocks to use [Kotelnikov et al. TabDDPM: MLPBlock(x) = Dropout(ReLU(Linear(x))) ]
    mlp_bloc_width : int
        Number of units in each MLP Block
    dropout_p : float in [0.0, 1.0]
        Probability of an element to be zeroed during training. 
    num_output_classes : int
        Number of levels in the dependent variable of the data we are training the model on.
    is_class_cond : boolean
        If the neural network should be class conditional (on dependent variable) or not. 

    Methods 
    -------
    timestep_embedding :
        Function to construct embedding for diffusion steps into higher order space.
    forward :
        Forward function for PyTorch nn.Module.
    """

    def __init__(self, input_size, num_mlp_blocks, mlp_block_width, dropout_p, num_output_classes, is_class_cond, seed):
        super(Neural_net, self).__init__()
        self.input_size = input_size
        self.num_mlp_blocks = num_mlp_blocks
        assert self.num_mlp_blocks >= 1, ValueError("The number of MLPBlocks needs to be at least 1.")
        self.mlp_block_width = mlp_block_width
        assert self.mlp_block_width >= 1, ValueError("The MLPBlock width needs to be positive.")
        self.dropout_p = dropout_p
        assert dropout_p >= 0 and dropout_p <= 1, ValueError("The dropout probability must be a real number between 0 and 1.")
        self.num_output_classes = num_output_classes # Number of classes that are possible for the output class to take (e.g. 2 in binary classification).
        self.is_class_cond = is_class_cond # States if the model should be trained as class-conditional.
        self.seed = seed # Random seed in main program, used to save models after training. 

        # The dimension of the embedding of the inputs (time embedding, covariate embedding and label embedding if relevant).
        self.dim_t_embedding = 128

        # Layers.
        self.l1 = nn.Linear(self.dim_t_embedding, self.mlp_block_width) # For first MLPBlock. 
        self.linear_layers = nn.ModuleList() # MLPBlocks inbetween the first MLPBlock and the linear output layer. 
        for _ in range(self.num_mlp_blocks-1):
            self.linear_layers.append(nn.Linear(self.mlp_block_width, self.mlp_block_width))
        self.outlayer = nn.Linear(self.mlp_block_width, input_size)
        
        # Activation functions. 
        self.relu = nn.ReLU()

        # Dropout.
        self.dropout = nn.Dropout(p = self.dropout_p) # Set some random dropout probability during training. 

        # Neural network for time embedding according to tabDDPM (Equation 5).
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_t_embedding, self.dim_t_embedding),
            nn.SiLU(),
            nn.Linear(self.dim_t_embedding, self.dim_t_embedding)
        )

        # Neural network for embedding the feature vector to the same space as the time embedding.
        self.proj = nn.Linear(input_size, self.dim_t_embedding)

        # Make embedding for class label, in order to train a class conditional model.
        if self.is_class_cond:
            self.label_embedding = nn.Embedding(self.num_output_classes, self.dim_t_embedding) # Linear layer for the label embedding. nn.Embedding makes it possible to skip manual one-hot encoding of label and instead give index.         

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

    def forward(self, x, t, y=None):
        """Forward steps for Pytorch."""
        # First we make the embeddings. 
        t_emb = self.time_embed(self.timestep_embedding(t)) # Not sure if this should take one t or several at once! 
                                                            # Here I assume that it takes several (according to output from sample_timesteps).
        x = x.to(torch.float32) # Change the data type here for now, quick fix since the weights of proj are float32 by default.                                                     
        x_emb = self.proj(x)

        if self.is_class_cond:
            if y is None:
                raise Exception("You need to supply the response 'y'.")
            y = y.squeeze() # Remove dimensions of size 1. This is to make sure that the label_embedding gives correct shape below. 
            # Not sure if this is needed at the moment. Check dimensions of x_emb and t_emb to see if it is necessary!
            t_emb += F.silu(self.label_embedding(y)) # Add the label embedding to the time embedding, if our model is class conditional. 

        x = x_emb + t_emb # Final embedding vector (consisting of features, time and labels if relevant).

        # Feed the embeddings to our "regular" MLP. 
        x = self.dropout(self.relu(self.l1(x))) # First MLPBlock
        for ll in self.linear_layers:
            x = self.dropout(self.relu(ll(x))) # MLPBlocks in between. 
        x = self.outlayer(x) # Linear out-layer.
        return x
    