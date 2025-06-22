"""Refactored Reptile meta-learning demo for sine wave regression tasks."""

import os
from collections.abc import Callable
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.func import functional_call
from tqdm import tqdm

import wandb
from utils import LoggerAggregator, _resolve_device, set_seed


def plot_results(
    model: nn.Module,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    iteration: int,
    cfg: DictConfig,
    logger: LoggerAggregator,
    device: torch.device,
):
    """Plot and log the current results (always on CPU in background)."""
    plt.cla()
    weights_before = deepcopy(model.state_dict())
    plt.plot(
        x_all, predict(model, x_all, device), label="pred after 0", color=(0, 0, 1)
    )
    for inneriter in range(cfg.inner_plot_steps):
        train_on_batch(
            model, xtrain_plot, f_plot(xtrain_plot), cfg.inner_stepsize, device
        )
        if (inneriter + 1) % cfg.plot_inner_interval == 0:
            frac = (inneriter + 1) / cfg.inner_plot_steps
            plt.plot(
                x_all,
                predict(model, x_all, device),
                label="pred after %i" % (inneriter + 1),
                color=(frac, 0, 1 - frac),
            )
    plt.plot(x_all, f_plot(x_all), label="true", color=(0, 1, 0))
    plt.plot(xtrain_plot, f_plot(xtrain_plot), "x", label="train", color="k")
    lossval = np.square(predict(model, x_all, device) - f_plot(x_all)).mean()
    plt.ylim(-4, 4)
    plt.legend(loc="lower right")
    plt.title(f"Reptile Training - Iteration {iteration + 1}")
    if hasattr(cfg, "assets_dir") and cfg.assets_dir:
        os.makedirs(cfg.assets_dir, exist_ok=True)
        plot_path = os.path.join(
            cfg.assets_dir, f"reptile_iteration_{iteration + 1}.png"
        )
        plt.savefig(plot_path)
        if cfg.wandb.log:
            logger.log_dict(
                {"plot": wandb.Image(plot_path)},
                section="visualization",
                kind="image",
                step=iteration,
            )

    model.load_state_dict(weights_before)
    logger.log_dict(
        {
            "loss_on_plotted_curve": lossval,
        },
        section="training",
        kind="scalar",
        step=iteration,
    )


class Regressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        self.net = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class HyperNetwork(nn.Module):
    """Fast weight programmer."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        self.task_emb = nn.Embedding(cfg.num_tasks, cfg.hyper_input_dim)
        hyper_input_dim = cfg.hyper_input_dim
        if self.cfg.input_cond:
            self.x_emb = nn.Linear(1, cfg.hyper_input_dim)
            hyper_input_dim *= 2

        # Calculate total number of parameters in the target network
        self.w1_shape = (cfg.hidden_dim, 1)
        self.b1_shape = (cfg.hidden_dim,)
        self.w2_shape = (cfg.hidden_dim, cfg.hidden_dim)
        self.b2_shape = (cfg.hidden_dim,)
        self.w3_shape = (1, cfg.hidden_dim)
        self.b3_shape = (1,)
        self.total_params = (
            np.prod(self.w1_shape)
            + np.prod(self.b1_shape)
            + np.prod(self.w2_shape)
            + np.prod(self.b2_shape)
            + np.prod(self.w3_shape)
            + np.prod(self.b3_shape)
        )

        self.hyper_mlp = nn.Sequential(
            nn.Linear(hyper_input_dim, cfg.intermediate_dim),
            nn.GELU(),
            nn.Linear(cfg.intermediate_dim, int(self.total_params)),
        )

        self.net = Regressor(cfg)

        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, task_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hypernetwork.

        Args:
            task_id: Tensor of shape (batch_size,) containing the task IDs.
            x: Tensor of shape (batch_size, 1) containing the input features.

        Returns:
            Tensor of shape (batch_size, 1) containing the output of the target model.
        """
        z = self.task_emb(task_id)
        if self.cfg.input_cond:
            z = torch.cat([z, self.x_emb(x)], dim=-1)
        param_vec = self.hyper_mlp(z).squeeze(0)  # shape: (total_params,)

        idx = 0

        def _take(shape):
            nonlocal idx
            n = int(np.prod(shape))
            out = param_vec[..., idx : idx + n].view(*shape)
            idx += n
            return out

        params_list = [
            _take(self.w1_shape),
            _take(self.b1_shape),
            _take(self.w2_shape),
            _take(self.b2_shape),
            _take(self.w3_shape),
            _take(self.b3_shape),
        ]
        param_names = [
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
            "net.4.weight",
            "net.4.bias",
        ]
        params = dict(zip(param_names, params_list))

        return functional_call(self.net, params, (x,))


def gen_task(rng: np.random.RandomState) -> Callable:
    """Generate a random sine wave task."""
    phase = rng.uniform(low=0, high=2 * np.pi)
    ampl = rng.uniform(0.1, 5)

    def f_randomsine(x):
        return np.sin(x + phase) * ampl

    return f_randomsine


def totorch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_on_batch(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    inner_stepsize: float,
    device: torch.device,
):
    """Train the model on a single batch using inner loop SGD."""
    x_tensor = totorch(x, device)
    y_tensor = totorch(y, device)

    model.zero_grad()
    ypred = model(x_tensor)
    loss = (ypred - y_tensor).pow(2).mean()
    loss.backward()

    # Manual SGD update
    for param in model.parameters():
        if param.grad is not None:
            param.data -= inner_stepsize * param.grad.data


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Make predictions using the model."""
    x_tensor = totorch(x, device)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()


def hypernet_train_loop(
    hypernet: nn.Module,
    cfg: DictConfig,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    rng: np.random.RandomState,
    logger: LoggerAggregator,
    device: torch.device,
):
    print(f"Starting Hypernet training for {cfg.outer_steps} iterations...")
    pbar = tqdm(range(cfg.outer_steps), desc="Hypernet-training")
    opt = torch.optim.Adam(hypernet.parameters(), lr=cfg.inner_stepsize)
    for iteration in pbar:
        f = gen_task(rng)
        y_all = f(x_all)
        inds = rng.permutation(len(x_all))
        for start in range(0, len(x_all), cfg.ntrain):
            mbinds = torch.tensor(inds[start : start + cfg.ntrain], device=device)
            x_mb = totorch(x_all[mbinds], device)
            y_mb = totorch(y_all[mbinds], device)
            pred = hypernet(mbinds, x_mb)
            loss = (pred - y_mb).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


def meta_train_loop(
    model: nn.Module,
    cfg: DictConfig,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    rng: np.random.RandomState,
    logger: LoggerAggregator,
    device: torch.device,
):
    print(f"Starting Reptile training for {cfg.outer_steps} iterations...")
    pbar = tqdm(range(cfg.outer_steps), desc="Meta-training")
    for iteration in pbar:
        weights_before = deepcopy(model.state_dict())
        f = gen_task(rng)
        y_all = f(x_all)
        inds = rng.permutation(len(x_all))
        for start in range(0, len(x_all), cfg.ntrain):
            mbinds = inds[start : start + cfg.ntrain]
            train_on_batch(
                model, x_all[mbinds], y_all[mbinds], cfg.inner_stepsize, device
            )
        weights_after = model.state_dict()

        y_pred = predict(model, x_all, device)
        lossval = np.square(y_pred - f(x_all)).mean()

        outerstepsize = cfg.outerstepsize * (1 - iteration / cfg.outer_steps)
        model.load_state_dict(
            {
                name: weights_before[name]
                + (weights_after[name] - weights_before[name]) * outerstepsize
                for name in weights_before
            }
        )

        pbar.set_postfix({"loss": f"{lossval:.4f}"})

        logger.log_dict(
            {"inner_loop_loss": lossval},
            section="training",
            kind="scalar",
            step=iteration,
        )

        if cfg.plot and (iteration == 0 or (iteration + 1) % cfg.plot_interval == 0):
            plot_results(
                model,
                x_all,
                f_plot,
                xtrain_plot,
                iteration,
                cfg,
                logger,
                device,
            )


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
    model = Regressor(cfg)
    model.to(device)

    # Define task distribution
    x_all = np.linspace(cfg.x_min, cfg.x_max, cfg.x_points)[:, None]

    # Choose a fixed task and minibatch for visualization
    f_plot = gen_task(rng)
    xtrain_plot = x_all[rng.choice(len(x_all), size=cfg.ntrain)]

    # Run meta-learning training
    meta_train_loop(model, cfg, x_all, f_plot, xtrain_plot, rng, logger, device)


if __name__ == "__main__":
    main()
