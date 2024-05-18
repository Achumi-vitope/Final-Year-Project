from collections import OrderedDict


from omegaconf import DictConfig

import torch
import numpy as np
from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):

    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        text = '40X_T_'
        loss, metrics = test(model, testloader[0], device, text)
        return loss, metrics

    return evaluate_fn