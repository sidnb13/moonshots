import itertools
import os
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

import wandb
from utils import LoggerAggregator, set_seed


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


class Dictionary(nn.Module):
    def __init__(self, num_tasks: int, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self._l1_emb = nn.Embedding(num_tasks, hidden_dim * n_features)
        # Simplified 2-layer per-task network: input->hidden, hidden->scalar head
        self._l1_bias = nn.Embedding(num_tasks, hidden_dim)
        self._l2_emb = nn.Embedding(num_tasks, hidden_dim)
        self._l2_bias = nn.Embedding(num_tasks, 1)
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


def balance_loss(loss: torch.Tensor, task_ids: torch.Tensor):
    """
    loss:      [B] per-sample loss (reduction='none')
    task_ids:  [B] ∈ {0,…,num_tasks-1}
    num_tasks: total C
    """

    counts = torch.zeros(max(task_ids) + 1, device=loss.device, dtype=loss.dtype)
    counts = counts.scatter_add_(0, task_ids, torch.ones_like(loss))
    per_sample_w = 1.0 / counts[task_ids]
    return loss * per_sample_w


def compute_loss(y_pred, y_true, task_ids: torch.LongTensor = None):
    bce_loss = F.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    ).squeeze(-1)
    bce_loss = balance_loss(bce_loss, task_ids)
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

        # build indices for balanced binary tasks
        import random
        random.seed(random_state)

        self.indices = []
        for tid in range(self.n_tasks):
            pos_idx = (self.y_global == tid).nonzero(as_tuple=True)[0].tolist()
            neg_idx = (self.y_global != tid).nonzero(as_tuple=True)[0].tolist()
            neg_sample = random.sample(neg_idx, k=total_examples_per_task)
            
            # Add positive examples
            for idx_ in pos_idx:
                self.indices.append((idx_, tid, 1.0))
            # Add negative examples
            for idx_ in neg_sample:
                self.indices.append((idx_, tid, 0.0))
        
        random.shuffle(self.indices)

    def split_dataset(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """Split the dataset into train/val/test portions while maintaining task balance."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
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
            n_test = n_total - n_train - n_val  # remainder goes to test
            
            train_indices.extend(task_data[:n_train])
            val_indices.extend(task_data[n_train:n_train + n_val])
            test_indices.extend(task_data[n_train + n_val:])
        
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


def make_dataloaders(cfg: DictConfig):
    """Create train/val/test dataloaders from a single dataset split."""
    # Calculate total examples needed
    total_examples_per_task = (
        cfg.num_examples_per_task_train + 
        cfg.num_examples_per_task_val + 
        cfg.num_examples_per_task_test
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
        train_ratio=train_ratio, 
        val_ratio=val_ratio, 
        test_ratio=test_ratio
    )
    
    # Create dataset splits
    train_dataset = DatasetSplit(full_dataset, train_indices)
    val_dataset = DatasetSplit(full_dataset, val_indices)
    test_dataset = DatasetSplit(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def train_one_task(
    cfg: DictConfig, model, train_loader, val_loader, task_id, logger: LoggerAggregator
) -> dict:
    opt = make_opt(model, cfg)
    step = 0
    train_iter = itertools.cycle(train_loader)
    total_steps = cfg.steps
    pbar = tqdm(range(total_steps), desc=f"Task {task_id} (steps)")
    while step < total_steps:
        model.train()
        x, y, task_id_batch = next(train_iter)

        y_pred = model(x.to(cfg.device), task_id_batch.to(cfg.device))

        loss = compute_loss(y_pred, y.to(cfg.device), task_id_batch.to(cfg.device))
        loss.backward()
        opt.step()
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
        opt.zero_grad()
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
                        y_pred, y.to(cfg.device), task_id_batch.to(cfg.device)
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
    acc = accuracy_score(y_true, (y_pred > 0.0).astype(float))
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Per-task metrics and ROC curves
    task_metrics = {}
    roc_curves = {}
    for tid in np.unique(task_ids):
        mask = task_ids == tid
        y_true_tid = y_true[mask]
        y_pred_tid = y_pred[mask]
        if len(np.unique(y_true_tid)) < 2:
            continue
        try:
            task_auc = roc_auc_score(y_true_tid, y_pred_tid)
            task_acc = accuracy_score(y_true_tid, (y_pred_tid > 0.5).astype(float))
            task_metrics[f"task_{tid}"] = {"accuracy": task_acc, "auc": task_auc}

            # Store ROC curve for this task
            task_fpr, task_tpr, _ = roc_curve(y_true_tid, y_pred_tid)
            roc_curves[f"task_{tid}"] = (task_fpr, task_tpr)
        except Exception as e:
            continue

    # Generate classification report
    report = classification_report(
        y_true, (y_pred > 0.5).astype(float), output_dict=True
    )

    # Save individual ROC curve
    assets_dir = cfg.assets_dir
    os.makedirs(assets_dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {'All Tasks' if task_id is None else f'Task {task_id}'}")
    plt.legend(loc="lower right")
    plot_path = os.path.join(
        assets_dir,
        f"roc_curve_{'all_tasks' if task_id is None else f'task_{task_id}'}.png",
    )
    plt.savefig(plot_path)
    plt.close()

    # Save overlay ROC curve for all tasks
    if len(roc_curves) > 0:
        plt.figure()
        for k, (task_fpr, task_tpr) in roc_curves.items():
            plt.plot(task_fpr, task_tpr, label=k)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Overlay (All Tasks)")
        plt.legend(loc="lower right")
        overlay_path = os.path.join(assets_dir, "roc_curve_overlay.png")
        plt.savefig(overlay_path)

        # Log to wandb if enabled
        if cfg.wandb.log:
            # Log individual ROC curve
            wandb.log(
                {
                    f"test/roc_curve_{'all_tasks' if task_id is None else f'task_{task_id}'}": wandb.plot.line_series(
                        xs=[fpr],
                        ys=[tpr],
                        keys=["ROC"],
                        title=f"ROC Curve - {'All Tasks' if task_id is None else f'Task {task_id}'}",
                        xname="FPR",
                    )
                }
            )

            # Log overlay ROC curve
            xs = [roc_curves[k][0] for k in roc_curves]
            ys = [roc_curves[k][1] for k in roc_curves]
            keys = list(roc_curves.keys())
            wandb.log(
                {
                    "test/roc_curve_overlay": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=keys,
                        title="ROC Curve Overlay (All Tasks)",
                        xname="FPR",
                    )
                }
            )

    return {
        "overall": {"accuracy": acc, "auc": auc, "classification_report": report},
        "per_task": task_metrics,
    }


def train_sequential(cfg: DictConfig):
    """Train a classifier for each task sequentially."""
    model = Dictionary(cfg.num_tasks, cfg.n_features, cfg.hidden_dim).to(cfg.device)
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
    model = Dictionary(cfg.num_tasks, cfg.n_features, cfg.hidden_dim).to(cfg.device)
    train_loader, val_loader, test_loader = make_dataloaders(cfg)
    logger = LoggerAggregator(cfg)
    opt = make_opt(model, cfg)
    step = 0
    train_iter = itertools.cycle(train_loader)
    total_steps = cfg.steps
    pbar = tqdm(range(total_steps), desc="Training (steps)")
    while step < total_steps:
        model.train()
        x, y, task_id = next(train_iter)
        y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
        loss = compute_loss(y_pred, y.to(cfg.device), task_id.to(cfg.device))
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
        opt.step()
        opt.zero_grad()
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
    for task_id, task_metrics in metrics["per_task"].items():
        print(f"\n{task_id}:")
        print(f"Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"AUC: {task_metrics['auc']:.4f}")


@hydra.main(config_path="config", config_name="mixed_batching", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    if cfg.train_mode == "sequential":
        train_sequential(cfg)
    elif cfg.train_mode == "mixed_batch":
        train_mixed_batch(cfg)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
