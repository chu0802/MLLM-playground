import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
