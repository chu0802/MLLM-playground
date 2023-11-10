import logging
import torch


def get_optimizer(model, config):
    optim_params = model.get_params(config.task.weight_decay)

    num_parameters = 0
    for param_group in optim_params:
        for p in param_group["params"]:
            num_parameters += p.data.nelement()
    logging.info(f"number of trainable parameters: {num_parameters}")

    return torch.optim.AdamW(
        optim_params,
        lr=float(config.task.init_lr),
        betas=(0.9, 0.999),
    )
