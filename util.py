import torch


def make_ixs(seq, to_ix, device):
    ixs = torch.tensor([to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]).to(
        device
    )
    return ixs

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def add_element_wise(list1, list2):
    return [a + b for a, b in zip(list1, list2)]
