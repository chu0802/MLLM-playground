import numpy as np
import torch


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(len(dataset), max_sample_num, replace=False)
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset
