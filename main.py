import os
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl
from flwr.common import Parameters

from dataset import prepare_dataset
from model import Net
import torch
from torchviz import make_dot

from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

import time

from conf_mat import show_cm 


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    save_path = HydraConfig.get().runtime.output_dir

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.batch_size)

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)
    

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),

    )  # a function to run on the server side to evaluate the global model.
    
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 8.0,
            "num_gpus": 1.0,
        },  
    )


    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    results_path = Path(save_path) / "results.pkl"
    
    results = {"history": history}
    
    model = Net(cfg.num_classes)
    os.makedirs("model_", exist_ok=True)
    print("Saving model")
    torch.save(model.state_dict(), "model_/200X_model.pth")
      
    show_cm()
    
    # Save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    main()