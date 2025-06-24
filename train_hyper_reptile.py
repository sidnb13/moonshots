"""Refactored Reptile meta-learning demo for sine wave regression tasks."""

import os
from collections.abc import Callable
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LowRankLinear(nn.Module):
    """Use spectral initialization as recommended in: https://www.arxiv.org/pdf/2105.01029v2"""

    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim

        with torch.no_grad():
            W_target = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=False)
            nn.init.xavier_uniform_(W_target)
            U, S, V = torch.svd(W_target)
            del W_target

        self.A = nn.Parameter(U[:, :rank] * S[:rank].sqrt())  # [in_dim, rank]
        self.B = nn.Parameter(V[:, :rank] * S[:rank].sqrt())  # [rank, out_dim]
        self.bias = nn.Parameter(torch.zeros(out_dim))

    @property
    def in_features(self):
        return self.in_dim

    @property
    def out_features(self):
        return self.out_dim

    def forward(self, x):
        return F.linear(x @ self.A, self.B, self.bias)


class TinyMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            LowRankLinear(in_dim, hidden_dim, rank=8),
            nn.GELU(),
            LowRankLinear(hidden_dim, out_dim, rank=8),
        )

    def forward(self, x):
        return self.net(x)


class HyperNetwork(nn.Module):
    """Fast weight programmer with a shared generator head."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hdim = cfg.hidden_dim
        self.layer_shapes = [
            (cfg.hidden_dim, 1),  # Layer 1: (out, in)
            (cfg.hidden_dim, cfg.hidden_dim),  # Layer 2
            (1, cfg.hidden_dim),  # Layer 3
        ]
        self.bias_shapes = [
            (cfg.hidden_dim,),
            (cfg.hidden_dim,),
            (1,),
        ]
        self.n_layers = len(self.layer_shapes)
        self.task_emb = nn.Embedding(cfg.num_tasks, cfg.hyper_input_dim)
        self.layer_emb = nn.Embedding(self.n_layers, cfg.hyper_input_dim)
        self.gen_out_dim = max(
            np.prod(w) + np.prod(b) for w, b in zip(self.layer_shapes, self.bias_shapes)
        )
        self.gen = TinyMLP(
            cfg.hyper_input_dim * 2, self.gen_out_dim, cfg.intermediate_dim
        )

    def forward(self, task_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        code = self.task_emb(task_id)  # [1, D]
        out = x
        for idx, (w_shape, b_shape) in enumerate(
            zip(self.layer_shapes, self.bias_shapes)
        ):
            emb = self.layer_emb(torch.tensor(idx, device=task_id.device))  # [D]
            inp = torch.cat([code.squeeze(0), emb], dim=-1)  # [2D]
            params = self.gen(inp)  # [gen_out_dim]
            w_num = np.prod(w_shape)
            b_num = np.prod(b_shape)
            wl = params[:w_num].view(*w_shape)
            bl = params[w_num : w_num + b_num].view(*b_shape)
            out = F.linear(out, wl, bl)
            if idx < self.n_layers - 1:
                out = torch.tanh(out)
        return out


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
    pbar = tqdm(
        range(cfg.outer_steps),
        desc="Hypernet-training",
        disable=not cfg.pbar_interval,
    )
    opt = torch.optim.AdamW(
        hypernet.parameters(),
        lr=cfg.hypernet_lr,
        weight_decay=cfg.hypernet_weight_decay,
    )
    for iteration in pbar:
        # sample task and input data
        task_id = rng.randint(0, cfg.num_tasks)
        f = gen_task_from_id(task_id)
        y_all = f(x_all)
        inds = rng.permutation(len(x_all))[: cfg.ntrain]
        x_mb = totorch(x_all[inds], device)
        y_mb = totorch(y_all[inds], device)
        task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=device)
        pred = hypernet(task_id_tensor, x_mb)
        loss = (pred - y_mb).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        opt.step()

        if (iteration + 1) % cfg.pbar_interval == 0:
            pbar.set_postfix({"loss": f"{loss:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        logger.log_dict(
            {"hypernet_loss": loss, "grad_norm": grad_norm},
            section="training",
            kind="scalar",
            step=iteration,
        )

        # Validation
        with torch.no_grad():
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
            if cfg.plot and (
                iteration == 0 or (iteration + 1) % cfg.plot_interval == 0
            ):
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
    pbar = tqdm(
        range(cfg.outer_steps),
        desc="Meta-training",
        disable=not cfg.pbar_interval,
    )
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

        if (iteration + 1) % cfg.pbar_interval == 0:
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
