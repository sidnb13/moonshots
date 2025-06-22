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
from tqdm import tqdm

import wandb
from utils import LoggerAggregator, _resolve_device, set_seed


def plot_results_hypernet(
    model: nn.Module,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    iteration: int,
    cfg: DictConfig,
    logger: LoggerAggregator,
    device: torch.device,
):
    """Plot and log the current hypernet results (simple: GT vs predicted)."""
    plt.cla()

    # Use a single task ID for the entire plot
    task_id = torch.tensor([0], device=device)  # Use task 0 for plotting
    x_tensor = totorch(x_all, device)

    # Get predictions from hypernet
    with torch.no_grad():
        y_pred = model(task_id, x_tensor).cpu().numpy()

    # Plot ground truth and prediction
    plt.plot(x_all, f_plot(x_all), label="ground truth", color=(0, 1, 0), linewidth=2)
    plt.plot(x_all, y_pred, label="predicted", color=(1, 0, 0), linewidth=2)
    plt.plot(
        xtrain_plot,
        f_plot(xtrain_plot),
        "x",
        label="train points",
        color="k",
        markersize=8,
    )

    # Calculate loss
    lossval = np.square(y_pred - f_plot(x_all)).mean()

    plt.ylim(-4, 4)
    plt.legend(loc="lower right")
    title = f"Hypernet Training - Iteration {iteration + 1}"
    plt.title(title)

    if hasattr(cfg, "assets_dir") and cfg.assets_dir:
        os.makedirs(cfg.assets_dir, exist_ok=True)
        plot_path = os.path.join(
            cfg.assets_dir, f"hypernet_iteration_{iteration + 1}.png"
        )
        plt.savefig(plot_path)
        if cfg.wandb.log:
            logger.log_dict(
                {"plot": wandb.Image(plot_path)},
                section="visualization",
                kind="image",
                step=iteration,
            )

    logger.log_dict(
        {
            "loss_on_plotted_curve": lossval,
        },
        section="training",
        kind="scalar",
        step=iteration,
    )


def plot_results(
    model: nn.Module,
    x_all: np.ndarray,
    f_plot: Callable,
    xtrain_plot: np.ndarray,
    iteration: int,
    cfg: DictConfig,
    logger: LoggerAggregator,
    device: torch.device,
    prefix: str = "reptile",
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
    title = f"{prefix.capitalize()} Training - Iteration {iteration + 1}"
    plt.title(title)
    if hasattr(cfg, "assets_dir") and cfg.assets_dir:
        os.makedirs(cfg.assets_dir, exist_ok=True)
        plot_path = os.path.join(
            cfg.assets_dir, f"{prefix}_iteration_{iteration + 1}.png"
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
        super().__init__()
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
        H = cfg.hidden_dim
        D = cfg.hyper_input_dim * (2 if cfg.input_cond else 1)

        # embeddings + optional input conditioning
        self.task_emb = nn.Embedding(cfg.num_tasks, cfg.hyper_input_dim)
        if cfg.input_cond:
            self.x_emb = nn.Linear(1, cfg.hyper_input_dim)

        # meta-parameters:
        self.lin1 = nn.Linear(D, 2 * H)  # [w1_vec (H), b1 (H)]
        self.lin2 = nn.Linear(D, H * H)  # [w2_vec (H×H)]
        self.lin3 = nn.Linear(D, H)  # [w3_vec (1×H) flattened as H]
        self.lin4 = nn.Linear(D, H + 1)  # [b2 (H), b3 (1)]

    def forward(self, task_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        H = self.lin3.out_features  # hidden dim

        # Accept task_id as scalar or [1]
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)
        assert task_id.shape[0] == 1, (
            "task_id should be a scalar or shape [1] for one-task-per-batch mode"
        )
        B = x.shape[0]

        # build conditioning vector for the task
        z = self.task_emb(task_id)  # [1, D]
        if hasattr(self, "x_emb"):
            z = torch.cat([z.expand(B, -1), self.x_emb(x)], dim=-1)  # [B, D]
        else:
            z = z.expand(B, -1)  # [B, D]

        # -- layer 1 params --
        v1 = self.lin1(z)  # [B, 2H]
        w1 = v1[:, :H].view(-1, H, 1)  # [B, H, 1]
        b1 = v1[:, H:]  # [B, H]

        # -- layer 2 params --
        w2 = self.lin2(z).view(-1, H, H)  # [B, H, H]

        # -- layer 3 weight --
        w3 = self.lin3(z).view(-1, 1, H)  # [B, 1, H]

        # -- biases for layer 2 & 3 --
        v4 = self.lin4(z)  # [B, H+1]
        b2 = v4[:, :H]  # [B, H]
        b3 = v4[:, H:]  # [B, 1]

        # -- forward pass --
        h1 = torch.tanh(torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1)
        h2 = torch.tanh(torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2)
        out = torch.bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3

        return out.squeeze(-1)


def gen_task(rng: np.random.RandomState) -> Callable:
    """Generate a random sine wave task."""
    phase = rng.uniform(low=0, high=2 * np.pi)
    ampl = rng.uniform(0.1, 5)

    def f_randomsine(x):
        return np.sin(x + phase) * ampl

    return f_randomsine


def gen_task_from_id(task_id: int) -> Callable:
    rng = np.random.RandomState(task_id)
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
    opt = torch.optim.Adam(hypernet.parameters(), lr=cfg.hypernet_lr)
    for iteration in pbar:
        task_id = rng.randint(0, cfg.num_tasks)
        f = gen_task_from_id(task_id)
        y_all = f(x_all)
        inds = rng.permutation(len(x_all))
        x_mb = totorch(x_all[inds], device)
        y_mb = totorch(y_all[inds], device)
        task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=device)
        pred = hypernet(task_id_tensor, x_mb)
        loss = (pred - y_mb).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_postfix({"loss": f"{loss:.4f}"})
        logger.log_dict(
            {"hypernet_loss": loss},
            section="training",
            kind="scalar",
            step=iteration,
        )

        # Validation
        if (iteration + 1) % cfg.val_interval == 0:
            val_task_id = rng.randint(0, cfg.num_tasks)
            f_val = gen_task_from_id(val_task_id)
            y_val = f_val(x_all)
            inds_val = rng.permutation(len(x_all))
            mbinds = inds_val[: cfg.ntrain]
            x_mb_val = totorch(x_all[mbinds], device)
            y_mb_val = totorch(y_val[mbinds], device)
            val_task_id_tensor = torch.tensor(
                [val_task_id], dtype=torch.long, device=device
            )
            pred_val = hypernet(val_task_id_tensor, x_mb_val)

            loss_val = (pred_val - y_mb_val).pow(2).mean()
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            logger.log_dict(
                {"hypernet_loss": loss_val},
                section="validation",
                kind="scalar",
                step=iteration,
            )

        # Plotting for hypernet
        if cfg.plot and (iteration == 0 or (iteration + 1) % cfg.plot_interval == 0):
            plot_results_hypernet(
                hypernet,
                x_all,
                f_plot,
                xtrain_plot,
                iteration,
                cfg,
                logger,
                device,
            )


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
                prefix="reptile",
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
    if cfg.mode == "hypernet":
        hypernet = HyperNetwork(cfg)
        hypernet.to(device)
        hypernet_train_loop(
            hypernet, cfg, x_all, f_plot, xtrain_plot, rng, logger, device
        )
    else:
        meta_train_loop(model, cfg, x_all, f_plot, xtrain_plot, rng, logger, device)


if __name__ == "__main__":
    main()
