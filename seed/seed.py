import torch
import torch.cuda
import random
import numpy as np


def seed(number):
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    random.seed(number)
    np.random.seed(number)