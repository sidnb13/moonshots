"""Refactored Reptile meta-learning demo for sine wave regression tasks."""

import os
from collections.abc import Callable
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from utils import LoggerAggregator, _resolve_device, set_seed


def make_model(cfg: DictConfig) -> nn.Module:
    """Create the neural network model for sine wave regression."""
    if cfg.activation.lower() == "tanh":
        activation = nn.Tanh()
    elif cfg.activation.lower() == "relu":
        activation = nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {cfg.activation}")

    model = nn.Sequential(
        nn.Linear(1, cfg.hidden_dim),
        activation,
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        activation,
        nn.Linear(cfg.hidden_dim, 1),
    )
    return model


def gen_task(cfg: DictConfig, rng: np.random.RandomState) -> Callable:
    """Generate a random sine wave task."""
    phase = rng.uniform(low=0, high=2 * np.pi)
    ampl = rng.uniform(0.1, 5)

    def f_randomsine(x):
        return np.sin(x + phase) * ampl

    return f_randomsine


def totorch(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor with gradient tracking."""
    return torch.tensor(x, dtype=torch.float32, requires_grad=True)


def train_on_batch(
    model: nn.Module, x: np.ndarray, y: np.ndarray, innerstepsize: float
):
    """Train the model on a single batch using inner loop SGD."""
    x_tensor = totorch(x)
    y_tensor = totorch(y)

    model.zero_grad()
    ypred = model(x_tensor)
    loss = (ypred - y_tensor).pow(2).mean()
    loss.backward()

    # Manual SGD update
    for param in model.parameters():
        if param.grad is not None:
            param.data -= innerstepsize * param.grad.data


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Make predictions using the model."""
    x_tensor = totorch(x)
    with torch.no_grad():
        return model(x_tensor).detach().numpy()


def plot_results(
    model: nn.Module,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    iteration: int,
    cfg: DictConfig,
    logger: LoggerAggregator,
):
    """Plot and log the current results."""
    plt.cla()
    weights_before = deepcopy(model.state_dict())  # save snapshot before evaluation

    # Plot predictions after 0 inner steps
    plt.plot(x_all, predict(model, x_all), label="pred after 0", color=(0, 0, 1))

    # Plot predictions after various numbers of inner steps
    for inneriter in range(32):
        train_on_batch(model, xtrain_plot, f_plot(xtrain_plot), cfg.innerstepsize)
        if (inneriter + 1) % 8 == 0:
            frac = (inneriter + 1) / 32
            plt.plot(
                x_all,
                predict(model, x_all),
                label="pred after %i" % (inneriter + 1),
                color=(frac, 0, 1 - frac),
            )

    # Plot true function and training points
    plt.plot(x_all, f_plot(x_all), label="true", color=(0, 1, 0))
    plt.plot(xtrain_plot, f_plot(xtrain_plot), "x", label="train", color="k")

    # Calculate and log loss
    lossval = np.square(predict(model, x_all) - f_plot(x_all)).mean()

    plt.ylim(-4, 4)
    plt.legend(loc="lower right")
    plt.title(f"Reptile Training - Iteration {iteration + 1}")

    # Save plot if assets directory is specified
    if hasattr(cfg, "assets_dir") and cfg.assets_dir:
        os.makedirs(cfg.assets_dir, exist_ok=True)
        plot_path = os.path.join(
            cfg.assets_dir, f"reptile_iteration_{iteration + 1}.png"
        )
        plt.savefig(plot_path)

        # Log to wandb if enabled
        if cfg.wandb.log:
            logger.log_dict(
                {"plot": wandb.Image(plot_path)},
                section="visualization",
                kind="image",
                step=iteration,
            )

    plt.pause(0.01)
    model.load_state_dict(weights_before)  # restore from snapshot

    # Log metrics
    logger.log_dict(
        {
            "loss_on_plotted_curve": lossval,
        },
        section="training",
        kind="scalar",
        step=iteration,
    )

    print("-----------------------------")
    print(f"iteration               {iteration + 1}")
    print(f"loss on plotted curve   {lossval:.3f}")


def meta_train_loop(
    model: nn.Module,
    cfg: DictConfig,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    rng: np.random.RandomState,
    logger: LoggerAggregator,
):
    """Main meta-learning training loop for Reptile."""
    print(f"Starting Reptile training for {cfg.niterations} iterations...")

    for iteration in tqdm(range(cfg.niterations), desc="Meta-training"):
        # Save weights before inner loop
        weights_before = deepcopy(model.state_dict())

        # Generate a new task
        f = gen_task(cfg, rng)
        y_all = f(x_all)

        # Inner loop: do SGD on this task
        inds = rng.permutation(len(x_all))
        for _ in range(cfg.innerepochs):
            for start in range(0, len(x_all), cfg.ntrain):
                mbinds = inds[start : start + cfg.ntrain]
                train_on_batch(model, x_all[mbinds], y_all[mbinds], cfg.innerstepsize)

        # Outer loop: interpolate between current weights and trained weights
        weights_after = model.state_dict()
        outerstepsize = cfg.outerstepsize0 * (
            1 - iteration / cfg.niterations
        )  # linear schedule

        # Reptile update: move towards the trained weights
        model.load_state_dict(
            {
                name: weights_before[name]
                + (weights_after[name] - weights_before[name]) * outerstepsize
                for name in weights_before
            }
        )

        # Periodically plot and log results
        if cfg.plot and (iteration == 0 or (iteration + 1) % cfg.plot_interval == 0):
            plot_results(model, x_all, f_plot, xtrain_plot, iteration, cfg, logger)


@hydra.main(config_path="config", config_name="hyper_reptile", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for the Reptile meta-learning demo."""
    # Set up logging
    logger = LoggerAggregator(cfg)

    # Set random seeds
    set_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set up device
    device = _resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Create model
    model = make_model(cfg)
    model.to(device)

    # Define task distribution
    x_all = np.linspace(cfg.x_min, cfg.x_max, cfg.x_points)[:, None]

    # Choose a fixed task and minibatch for visualization
    f_plot = gen_task(cfg, rng)
    xtrain_plot = x_all[rng.choice(len(x_all), size=cfg.ntrain)]

    # Run meta-learning training
    meta_train_loop(model, cfg, x_all, f_plot, xtrain_plot, rng, logger)


if __name__ == "__main__":
    main()
