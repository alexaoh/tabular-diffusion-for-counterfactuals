# Utility functions for diffusion models and other applications. 
import torch
import torch.nn.functional as F

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
    """Convert a vector with an index to a one-hot-encoded vector in log-space.
    
    This has been heavily inspired by implementation in TabDDPM, which is a modified version of the original function from Hoogeboom et al.,
    such that it works for several categorical features at once. 
    """
    onehot = [] # Make one common list of one-hot-vectors. 
    for i in range(len(categorical_levels)):
        onehot.append(F.one_hot(x[:,i], categorical_levels[i]))  # One-hot-encode each of the categorical features separately. 
    
    x = torch.cat(onehot, dim = 1) # Concatenate the one-hot-vectors columnwise. 

    log_x = torch.log(x.float().clamp(min=1e-30)) # Take logarithm of the concatenated one-hot-vectors. 

    return log_x

def log_sub_exp(a, b):
    """Subtraction in log-space does not exist in Pytorch by default. This is the same as logaddexp(), but with subtraction instead."""
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

def sliced_logsumexp(x, slices):
    """Function copied from TabDDPM implementation. This is used in the final step in theta_post()."""
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
        slice_ends - slice_starts, 
        dim=-1
    ) # This function call copies the values from slice_lse columnwise a number of times corresponding to the number of levels in each categorical variable. 
    return slice_lse_repeated
