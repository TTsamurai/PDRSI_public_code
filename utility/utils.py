import pickle
import random

import numpy as np
import torch


def load(filename):
    with open(filename, "rb+") as file:
        return pickle.load(file)


def save(obj, filename, type="pickle"):
    if type == "pickle":
        with open(filename, "wb") as file:
            pickle.dump(obj, file, protocol=4)


def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
