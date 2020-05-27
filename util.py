import torch
from IPython import embed

def make_ixs(seq, to_ix, device):
    ixs = torch.tensor([to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]).to(
        device
    )
    return ixs

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def debug_mode():
    """Runs the debug mode"""
    embed(stack_depth = 1)
    exit()