import itertools
import os
import random
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.optim import SGD, Adagrad, Adam, RMSprop
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import (
    LoggerAggregator,
    ProfilingContext,
    WarmupContext,
    _resolve_device,
    profile_and_log_flamegraph,
    profile_and_log_flamegraph_backward,
    set_seed,
)


def make_opt(model, cfg: DictConfig):
    lr = cfg.lr
    opt_type = cfg.opt_type
    if opt_type == "sgd":
        return SGD(model.parameters(), lr=lr)
    elif opt_type == "adam":
        return Adam(model.parameters(), lr=lr)
    elif opt_type == "adagrad":
        return Adagrad(model.parameters(), lr=lr)
    elif opt_type == "rmsprop":
        return RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


class Dictionary(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        n_features: int,
        hidden_dim: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self._l1_emb = nn.Embedding(num_tasks, hidden_dim * n_features, device=device)
        self._l1_bias = nn.Embedding(num_tasks, hidden_dim, device=device)
        self._l2_emb = nn.Embedding(num_tasks, hidden_dim, device=device)
        self._l2_bias = nn.Embedding(num_tasks, 1, device=device)
        self.act_fn = nn.GELU()

    def forward(self, x, task_ids):
        # x: [B, D], task_ids: [B]
        # Layer 1: input->hidden
        w1 = self._l1_emb.weight[task_ids].view(-1, self.n_features, self.hidden_dim)
        b1 = self._l1_bias(task_ids)  # [B, hidden_dim]
        h = (x.unsqueeze(1) @ w1).squeeze(1) + b1
        h = self.act_fn(h)
        # Layer 2: hidden->scalar head
        w2 = self._l2_emb.weight[task_ids]  # [B, hidden_dim]
        b2 = self._l2_bias(task_ids).squeeze(-1)  # [B]
        logits = (h * w2).sum(dim=1) + b2
        return logits.unsqueeze(-1)


def balance_loss(loss: torch.Tensor, task_ids: torch.Tensor, num_tasks: int):
    """
    loss:      [B] per-sample loss (reduction='none')
    task_ids:  [B] ∈ {0,…,num_tasks-1}
    num_tasks: total C
    """
    counts = torch.bincount(task_ids, minlength=num_tasks).to(loss.dtype)
    return loss / counts[task_ids]


def compute_loss(
    y_pred, y_true, task_ids: torch.LongTensor | None = None, num_tasks: int = 1
):
    """
    y_pred: [B, 1]
    y_true: [B, 1]
    task_ids: [B] ∈ {0,…,num_tasks-1}
    num_tasks: total C
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    ).squeeze(-1)
    if task_ids is not None:
        bce_loss = balance_loss(bce_loss, task_ids, num_tasks=num_tasks)
    return bce_loss.mean()


class ClassificationTaskDataset(Dataset):
    """
    For n_tasks classes and examples_per_task examples per class, this dataset now builds a balanced binary dataset by sampling equal positives and negatives per task.
    """

    def __init__(
        self,
        n_tasks: int = 10,
        total_examples_per_task: int = 300,  # Total examples that will be split
        n_features: int = 20,
        random_state: int = 0,
    ):
        self.n_tasks = n_tasks
        self.total_per_task = total_examples_per_task
        self.n_feat = n_features

        random.seed(random_state)
        np.random.seed(random_state)
        start_time = time.time()

        if n_tasks == 1:
            # Special case: single binary task
            X, y = make_classification(
                n_samples=total_examples_per_task
                * 2,  # ensure enough positives/negatives
                n_features=n_features,
                n_informative=max(2, n_features // 2),
                n_redundant=0,
                n_classes=2,
                flip_y=0.01,
                random_state=random_state,
            )
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y_global = torch.tensor(y, dtype=torch.long)
            # For the only task (tid=0), class 1 is positive, class 0 is negative
            pos_idx = (self.y_global == 1).nonzero(as_tuple=True)[0].tolist()
            neg_idx = (self.y_global == 0).nonzero(as_tuple=True)[0].tolist()
            # Sample to balance
            n_pos = min(len(pos_idx), total_examples_per_task)
            n_neg = min(len(neg_idx), total_examples_per_task)
            pos_sample = random.sample(pos_idx, k=n_pos)
            neg_sample = random.sample(neg_idx, k=n_neg)
            self.indices = []
            for idx_ in pos_sample:
                self.indices.append((idx_, 0, 1.0))
            for idx_ in neg_sample:
                self.indices.append((idx_, 0, 0.0))
        else:
            # generate a balanced multiclass dataset (larger to accommodate splits)
            X, y = make_classification(
                n_samples=total_examples_per_task * n_tasks,
                n_features=n_features,
                n_informative=max(2, n_features // 2),
                n_redundant=0,
                n_classes=n_tasks,
                flip_y=0.01,
                random_state=random_state,
            )
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y_global = torch.tensor(y, dtype=torch.long)
            self.indices = []

            y_np = self.y_global.numpy()
            all_indices = np.arange(len(y_np))

            for tid in tqdm(
                range(self.n_tasks), desc="Building ClassificationTaskDataset"
            ):
                pos_idx = all_indices[y_np == tid]
                neg_idx = all_indices[y_np != tid]
                neg_sample = np.random.choice(
                    neg_idx, size=total_examples_per_task, replace=False
                )
                # Add positive examples
                for idx_ in pos_idx:
                    self.indices.append((idx_, tid, 1.0))
                # Add negative examples
                for idx_ in neg_sample:
                    self.indices.append((idx_, tid, 0.0))
        random.shuffle(self.indices)

        elapsed = time.time() - start_time
        print(
            f"ClassificationTaskDataset generated in {elapsed:.2f}s with {len(self.indices)} examples"
        )

    def split_dataset(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """Split the dataset into train/val/test portions while maintaining task balance."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            "Ratios must sum to 1.0"
        )

        # Group indices by task to ensure balanced splits
        task_indices = {tid: [] for tid in range(self.n_tasks)}
        for i, (idx, tid, label) in enumerate(self.indices):
            task_indices[tid].append(i)

        train_indices, val_indices, test_indices = [], [], []

        for tid in range(self.n_tasks):
            task_data = task_indices[tid]
            n_total = len(task_data)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_indices.extend(task_data[:n_train])
            val_indices.extend(task_data[n_train : n_train + n_val])
            test_indices.extend(task_data[n_train + n_val :])

        return train_indices, val_indices, test_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example_idx, tid, y_val = self.indices[idx]
        x = self.X[example_idx]
        y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
        return x, y, tid


class DatasetSplit(Dataset):
    """A subset of a dataset defined by indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_fn(batch):
    x, y, task_id = zip(*batch)
    return torch.stack(x), torch.stack(y), torch.tensor(task_id)


def print_split_stats(dataset, indices, name):
    y_vals = [dataset.indices[i][2] for i in indices]
    pos = sum(1 for y in y_vals if y == 1.0)
    neg = sum(1 for y in y_vals if y == 0.0)
    print(f"{name}: Positives={pos}, Negatives={neg}, Total={len(indices)}")


def make_dataloaders(cfg: DictConfig):
    """Create train/val/test dataloaders from a single dataset split."""
    # Calculate total examples needed
    total_examples_per_task = (
        cfg.num_examples_per_task_train
        + cfg.num_examples_per_task_val
        + cfg.num_examples_per_task_test
    )

    # Create the full dataset
    full_dataset = ClassificationTaskDataset(
        n_tasks=cfg.num_tasks,
        total_examples_per_task=total_examples_per_task,
        n_features=cfg.n_features,
        random_state=cfg.seed,
    )

    # Calculate split ratios based on the desired sizes
    train_ratio = cfg.num_examples_per_task_train / total_examples_per_task
    val_ratio = cfg.num_examples_per_task_val / total_examples_per_task
    test_ratio = cfg.num_examples_per_task_test / total_examples_per_task

    # Split the dataset
    train_indices, val_indices, test_indices = full_dataset.split_dataset(
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # Print class balance stats
    print_split_stats(full_dataset, train_indices, "Train")
    print_split_stats(full_dataset, val_indices, "Val")
    print_split_stats(full_dataset, test_indices, "Test")

    # Create dataset splits
    train_dataset = DatasetSplit(full_dataset, train_indices)
    val_dataset = DatasetSplit(full_dataset, val_indices)
    test_dataset = DatasetSplit(full_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


def training_step_mixed_batch(model, batch, cfg):
    """Training step for mixed batch mode."""
    x, y, task_id = batch
    y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
    loss = compute_loss(y_pred, y.to(cfg.device), task_id.to(cfg.device), cfg.num_tasks)
    loss /= cfg.gradient_accumulation_steps
    loss.backward()
    return loss


def training_step_sequential(model, batch, cfg):
    """Training step for sequential mode."""
    x, y, task_id_batch = batch
    y_pred = model(x.to(cfg.device), task_id_batch.to(cfg.device))
    loss = compute_loss(
        y_pred, y.to(cfg.device), task_id_batch.to(cfg.device), cfg.num_tasks
    )
    loss /= cfg.gradient_accumulation_steps
    loss.backward()
    return loss


def train_one_task(
    cfg: DictConfig, model, train_loader, val_loader, task_id, logger: LoggerAggregator
) -> dict:
    opt = make_opt(model, cfg)
    step = 0
    train_iter = itertools.cycle(train_loader)
    if cfg.steps > 0:
        total_steps = cfg.steps
        print(f"[train_one_task] Using steps mode: {total_steps} steps")
    else:
        total_steps = cfg.epochs * len(train_loader)
        print(
            f"[train_one_task] Using epochs mode: {cfg.epochs} epochs, {total_steps} steps"
        )
    pbar = tqdm(range(total_steps), desc=f"Task {task_id} (steps)")
    profile_interval = getattr(cfg, "profile_interval", 1000)
    while step < total_steps:
        model.train()
        x, y, task_id_batch = next(train_iter)
        batch_size = x.shape[0]
        enable_profiler = step % profile_interval == 0 and step > 0

        # Use WarmupContext for profiling
        if enable_profiler:
            with WarmupContext(
                model, train_iter, cfg, training_step_fn=training_step_sequential
            ):
                with ProfilingContext(batch_size, step, cfg, enable_profiler=True):
                    y_pred = model(x.to(cfg.device), task_id_batch.to(cfg.device))
                    loss = compute_loss(
                        y_pred,
                        y.to(cfg.device),
                        task_id_batch.to(cfg.device),
                        cfg.num_tasks,
                    )
                    loss /= cfg.gradient_accumulation_steps
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                    else:
                        loss.backward()
        else:
            # Normal training step without profiling
            y_pred = model(x.to(cfg.device), task_id_batch.to(cfg.device))
            loss = compute_loss(
                y_pred, y.to(cfg.device), task_id_batch.to(cfg.device), cfg.num_tasks
            )
            loss /= cfg.gradient_accumulation_steps
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                loss.backward()
                opt.step()
                opt.zero_grad()
            else:
                loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )
        logger.log_dict(
            {
                "train_loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": opt.param_groups[0]["lr"],
            },
            section=f"train_task_{task_id}",
            kind="scalar",
            step=step,
        )
        if enable_profiler:
            results = getattr(ProfilingContext, "results", None)
            if results:
                logger.log_dict(
                    results,
                    section="profile",
                    kind="scalar",
                    step=step,
                )
                print(
                    f"[PROFILE] Step {step}: {results['examples_per_sec']:.2f} ex/s, {results['step_time_cuda_sync']:.4f} s/step, {results['flops_per_sec'] if results['flops_per_sec'] != -1 else 'N/A'} flops/s"
                )
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step})
        pbar.update(1)
        step += 1
        if step % cfg.val_interval == 0 or step == total_steps:
            model.eval()
            val_loss = 0
            count = 0
            all_y_true = []
            all_y_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y, task_id_batch = batch
                    y_pred = model(x.to(cfg.device), task_id_batch.to(cfg.device))
                    batch_val_loss = compute_loss(
                        y_pred,
                        y.to(cfg.device),
                        task_id_batch.to(cfg.device),
                        cfg.num_tasks,
                    ).item()
                    val_loss += batch_val_loss
                    count += 1
                    all_y_true.append(y.cpu().numpy())
                    all_y_pred.append(y_pred.cpu().numpy())
            avg_val_loss = val_loss / count
            y_true = np.concatenate(all_y_true).ravel()
            y_pred = np.concatenate(all_y_pred).ravel()
            acc = accuracy_score(y_true, (y_pred > 0.5).astype(float))
            logger.log_dict(
                {"val_loss": avg_val_loss, "val_acc": acc},
                section=f"validation_task_{task_id}",
                kind="scalar",
                step=step,
            )
    pbar.close()
    # --- Profile and log final flamegraph ---
    example_batch = next(iter(val_loader))
    profile_and_log_flamegraph(
        model, example_batch, cfg, logger, compute_loss=compute_loss
    )
    profile_and_log_flamegraph_backward(
        model, example_batch, cfg, logger, compute_loss=compute_loss
    )
    return {}


def evaluate_model(cfg: DictConfig, model, test_loader, task_id: int | None = None):
    """Evaluate model on test set with full metrics."""
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_task_ids = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, task_id_batch = batch
            y_pred = F.sigmoid(model(x.to(cfg.device), task_id_batch.to(cfg.device)))
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            all_task_ids.append(task_id_batch.cpu().numpy())

    y_true = np.concatenate(all_y_true).ravel()
    y_pred = np.concatenate(all_y_pred).ravel()
    task_ids = np.concatenate(all_task_ids).ravel()

    # Overall metrics
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(float))
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Per-task metrics
    task_metrics = {}
    for tid in np.unique(task_ids):
        mask = task_ids == tid
        y_true_tid = y_true[mask]
        y_pred_tid = y_pred[mask]
        if len(np.unique(y_true_tid)) < 2:
            continue
        try:
            task_auc = roc_auc_score(y_true_tid, y_pred_tid)
            task_acc = accuracy_score(y_true_tid, (y_pred_tid > 0.5).astype(float))
            task_metrics[tid] = {"accuracy": task_acc, "auc": task_auc}
        except Exception:
            continue

    # Generate classification report
    report = classification_report(
        y_true, (y_pred > 0.5).astype(float), output_dict=True
    )

    # --- Plotting ---
    assets_dir = cfg.assets_dir
    os.makedirs(assets_dir, exist_ok=True)
    plot_suffix = f"{'all_tasks' if task_id is None else f'task_{task_id}'}"

    # Save overall ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"Overall ROC (area = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Overall ROC Curve - {plot_suffix.replace('_', ' ').title()}")
    plt.legend(loc="lower right")
    roc_plot_path = os.path.join(assets_dir, f"roc_curve_{plot_suffix}.png")
    plt.savefig(roc_plot_path)
    plt.close()

    if cfg.wandb.log:
        wandb.log(
            {
                f"test/roc_curve_{plot_suffix}": wandb.plot.line_series(
                    xs=[fpr],
                    ys=[tpr],
                    keys=["ROC"],
                    title=f"ROC Curve - {plot_suffix.replace('_', ' ').title()}",
                    xname="FPR",
                )
            }
        )

    # Save per-task accuracy and AUC scatter plots
    if len(task_metrics) > 0:
        task_ids_list = list(task_metrics.keys())
        accuracies = [task_metrics[tid]["accuracy"] for tid in task_ids_list]
        aucs = [task_metrics[tid]["auc"] for tid in task_ids_list]

        # Accuracy Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(task_ids_list, accuracies, alpha=0.6)
        plt.xlabel("Task ID")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title(
            f"Per-Task Test Accuracy Scatter Plot - {plot_suffix.replace('_', ' ').title()}"
        )
        plt.grid(True)
        acc_scatter_path = os.path.join(
            assets_dir, f"accuracy_scatter_{plot_suffix}.png"
        )
        plt.tight_layout()
        plt.savefig(acc_scatter_path)
        plt.close()

        # AUC Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(task_ids_list, aucs, alpha=0.6)
        plt.xlabel("Task ID")
        plt.ylabel("AUC")
        plt.ylim(0, 1)
        plt.title(
            f"Per-Task Test AUC Scatter Plot - {plot_suffix.replace('_', ' ').title()}"
        )
        plt.grid(True)
        auc_scatter_path = os.path.join(assets_dir, f"auc_scatter_{plot_suffix}.png")
        plt.tight_layout()
        plt.savefig(auc_scatter_path)
        plt.close()

        if cfg.wandb.log:
            wandb.log(
                {f"test/accuracy_scatter_{plot_suffix}": wandb.Image(acc_scatter_path)}
            )
            wandb.log(
                {f"test/auc_scatter_{plot_suffix}": wandb.Image(auc_scatter_path)}
            )

    return {
        "overall": {"accuracy": acc, "auc": auc, "classification_report": report},
        "per_task": task_metrics,
    }


def train_sequential(cfg: DictConfig):
    """Train a classifier for each task sequentially."""
    # Init on device
    device = _resolve_device(cfg.device)
    model = Dictionary(cfg.num_tasks, cfg.n_features, cfg.hidden_dim, device=device)  # type: ignore
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    logger = LoggerAggregator(cfg)

    for task_id in range(cfg.num_tasks):
        print(f"Training task {task_id}")
        train_one_task(cfg, model, train_loader, val_loader, task_id, logger)

        # Evaluate on test set after training
        print(f"Evaluating task {task_id}")
        metrics = evaluate_model(cfg, model, test_loader, task_id)
        print(f"Task {task_id} Test Results:")
        print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"AUC: {metrics['overall']['auc']:.4f}")
        print("\nClassification Report:")
        print(metrics["overall"]["classification_report"])


def train_mixed_batch(cfg: DictConfig):
    """Train a library of classifers for all tasks in a single batch, step-based."""
    # Init on device
    device = _resolve_device(cfg.device)

    model = Dictionary(cfg.num_tasks, cfg.n_features, cfg.hidden_dim, device=device)  # type: ignore
    train_loader, val_loader, test_loader = make_dataloaders(cfg)
    logger = LoggerAggregator(cfg)
    opt = make_opt(model, cfg)
    step = 0
    train_iter = itertools.cycle(train_loader)
    if cfg.steps > 0:
        total_steps = cfg.steps
        print(f"[train_mixed_batch] Using steps mode: {total_steps} steps")
    else:
        total_steps = cfg.epochs * len(train_loader)
        print(
            f"[train_mixed_batch] Using epochs mode: {cfg.epochs} epochs, {total_steps} steps"
        )
    pbar = tqdm(range(total_steps), desc="Training (steps)")
    profile_interval = getattr(cfg, "profile_interval", 1000)
    while step < total_steps:
        model.train()
        x, y, task_id = next(train_iter)
        batch_size = x.shape[0]
        enable_profiler = step % profile_interval == 0 and step > 0

        # Use WarmupContext for profiling
        if enable_profiler:
            with WarmupContext(
                model, train_iter, cfg, training_step_fn=training_step_mixed_batch
            ):
                with ProfilingContext(batch_size, step, cfg, enable_profiler=True):
                    y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
                    loss = compute_loss(
                        y_pred, y.to(cfg.device), task_id.to(cfg.device)
                    )
                    loss /= cfg.gradient_accumulation_steps
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                    else:
                        loss.backward()
        else:
            # Normal training step without profiling
            y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
            loss = compute_loss(y_pred, y.to(cfg.device), task_id.to(cfg.device))
            loss /= cfg.gradient_accumulation_steps
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                loss.backward()
                opt.step()
                opt.zero_grad()
            else:
                loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )
        logger.log_dict(
            {
                "train_loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": opt.param_groups[0]["lr"],
            },
            section="train",
            kind="scalar",
            step=step,
        )
        if enable_profiler:
            results = getattr(ProfilingContext, "results", None)
            if results:
                logger.log_dict(
                    results,
                    section="profile",
                    kind="scalar",
                    step=step,
                )
                print(
                    f"[PROFILE] Step {step}: {results['examples_per_sec']:.2f} ex/s, {results['step_time_cuda_sync']:.4f} s/step, {results['flops_per_sec'] if results['flops_per_sec'] != -1 else 'N/A'} flops/s"
                )
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step})
        pbar.update(1)
        step += 1
        if step % cfg.val_interval == 0 or step == total_steps:
            model.eval()
            val_loss = 0
            count = 0
            all_y_true = []
            all_y_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y, task_id = batch
                    y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
                    batch_val_loss = compute_loss(
                        y_pred, y.to(cfg.device), task_id.to(cfg.device)
                    ).item()
                    val_loss += batch_val_loss
                    count += 1
                    all_y_true.append(y.cpu().numpy())
                    all_y_pred.append(y_pred.cpu().numpy())
            avg_val_loss = val_loss / count
            y_true = np.concatenate(all_y_true).ravel()
            y_pred = np.concatenate(all_y_pred).ravel()
            acc = accuracy_score(y_true, (y_pred > 0.5).astype(float))
            logger.log_dict(
                {"val_loss": avg_val_loss, "val_acc": acc},
                section="validation",
                kind="scalar",
                step=step,
            )
    pbar.close()
    # Evaluate on test set after training
    print("Evaluating on test set")
    metrics = evaluate_model(cfg, model, test_loader)
    print("Test Results:")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Overall AUC: {metrics['overall']['auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["overall"]["classification_report"])
    print("\nPer-task metrics:")
    # Print a summary instead of all tasks if there are many
    if len(metrics["per_task"]) > 10:
        print("(Showing first 10 tasks out of", len(metrics["per_task"]), ")")
        for i, (task_id, task_metrics) in enumerate(metrics["per_task"].items()):
            if i >= 10:
                break
            print(f"\nTask {task_id}:")
            print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"  AUC: {task_metrics['auc']:.4f}")
    else:
        for task_id, task_metrics in metrics["per_task"].items():
            print(f"\nTask {task_id}:")
            print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"  AUC: {task_metrics['auc']:.4f}")
    # --- Profile and log final flamegraph ---
    example_batch = next(iter(test_loader))
    profile_and_log_flamegraph(
        model, example_batch, cfg, logger, compute_loss=compute_loss
    )
    profile_and_log_flamegraph_backward(
        model, example_batch, cfg, logger, compute_loss=compute_loss
    )


@hydra.main(config_path="config", config_name="mixed_batching", version_base=None)
def main(cfg: DictConfig):
    # Dynamically set wandb run name BEFORE any wandb or logger usage
    n_tasks = cfg.num_tasks
    batch_size = cfg.batch_size
    lr = cfg.lr
    opt_type = cfg.opt_type
    train_mode = cfg.train_mode
    if cfg.steps > 0:
        steps_or_epochs = f"{cfg.steps}steps"
    else:
        steps_or_epochs = f"{cfg.epochs}epochs"
    run_name = f"{n_tasks}tasks_{steps_or_epochs}_bs{batch_size}_lr{lr}_{opt_type}_{train_mode}"
    cfg.wandb.run_name = run_name

    set_seed(cfg.seed)

    if cfg.train_mode == "sequential":
        train_sequential(cfg)
    elif cfg.train_mode == "mixed_batch":
        train_mixed_batch(cfg)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
