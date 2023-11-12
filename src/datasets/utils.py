import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets import get_dataset
from itertools import chain


def default_collater(batch):
    return_dict = {key: [dict[key] for dict in batch] for key in batch[0]}
    return_dict["answers"] = list(chain(*return_dict["answers"]))
    return return_dict


def build_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    **kwargs
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=default_collater,
    )


def get_dataloaders(config):
    dataloaders = {}
    for split_type, split_config_dict in config.dataset.split.items():
        dataset = sample_dataset(
            get_dataset(config, split_type),
            config.dataset.sample_num,
            config.dataset.sample_seed,
        )
        dataloaders[split_type] = build_dataloader(dataset, **split_config_dict)
    return dataloaders


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if len(dataset) > max_sample_num and max_sample_num != -1:
        np.random.seed(seed)
        random_indices = np.random.choice(len(dataset), max_sample_num, replace=False)
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    if torch.is_tensor(sample):
        return f(sample)
    elif isinstance(sample, dict):
        return {key: apply_to_sample(f, value) for key, value in sample.items()}
    elif isinstance(sample, list):
        return [apply_to_sample(f, x) for x in sample]
    else:
        return sample


def prepare_sample(samples):
    samples = apply_to_sample(lambda x: x.cuda(), samples)

    return samples


if __name__ == "__main__":
    from src.utils.config import get_config

    config = get_config()
    dataloaders = get_dataloaders(config)
    for batch in dataloaders["eval"]:
        print(batch)
        pass
