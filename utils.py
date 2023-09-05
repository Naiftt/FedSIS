import torch
import random 
import numpy as np


def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def weight_vec(network):
    """
    flatten weight tensors of a PyTorch neural network.

    Args:
        network (nn.Module): The PyTorch neural network model.

    Returns:
        torch.Tensor: A concatenated tensor containing all weights.
    """
    # Initialize an empty list to store flattened weight tensors
    flattened_weights = []

    for w in network.parameters():
        # Flatten the current weight tensor and append it to the list
        flattened_weights.append(torch.flatten(w))

    # Concatenate the flattened weight tensors in the list
    # to create a single concatenated tensor containing all weights
    return torch.cat(flattened_weights)

def weight_dec_global(pyModel, weight_vec):
    """
    Reconstructs and updates the weight tensors of a PyTorch model using a flattened weight vector.

    Args:
        pyModel (nn.Module): The PyTorch model to update with the reconstructed weights.
        weight_vec (torch.Tensor): A flattened tensor containing weight values.

    Returns:
        nn.Module: The updated PyTorch model with the reconstructed weights.
    """
    c = 0
    for w in pyModel.parameters():
        m = w.numel()
        D = weight_vec[c:m+c].reshape(w.data.shape)
        c += m
        if w.data is None:
            w.data = D + 0
        else:
            with torch.no_grad():
                w.set_(D + 0)
    return pyModel

