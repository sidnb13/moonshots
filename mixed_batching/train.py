import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch.optim import SGD, Adagrad, Adam, RMSprop
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb


def set_seed(seed: int):
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
        self._l2_emb = nn.Embedding(num_tasks, hidden_dim * hidden_dim)
        self._l3_emb = nn.Embedding(num_tasks, hidden_dim)
        self.act_fn = nn.GELU()

    def forward(self, x, task_ids):
        # x: [B, D]
        # task_ids: [B]

        # [B, F, D]
        l1_wei = self._l1_emb.weight[task_ids].view(
            -1, self.n_features, self.hidden_dim
        )
        h1 = self.act_fn(x.unsqueeze(1) @ l1_wei)
        # [B, F, F]
        l2_wei = self._l2_emb.weight[task_ids].view(
            -1, self.hidden_dim, self.hidden_dim
        )
        h2 = self.act_fn(h1 @ l2_wei)
        # [B, F, 1]
        l3_wei = self._l3_emb.weight[task_ids].view(-1, self.hidden_dim, 1)
        out = h2 @ l3_wei

        return F.sigmoid(out.squeeze(-1))


class LoggerAggregator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []  # Optional: keep a simple list of all logs for debugging
        if cfg.wandb.log:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                id=cfg.wandb.run_id,
                group=cfg.wandb.group,
                tags=cfg.wandb.tags,
                notes=cfg.wandb.notes,
            )
            wandb.config.update(OmegaConf.to_container(cfg))
            wandb.run.name = cfg.wandb.run_name

    def log_dict(self, log_dict, section=None, kind="scalar", step=None):
        # log_dict: {key: value, ...}
        # Optionally store in local history for debugging
        for key, value in log_dict.items():
            self.history.append(
                {
                    "section": section,
                    "key": key,
                    "value": value,
                    "kind": kind,
                    "step": step,
                }
            )
        if self.cfg.wandb.log:
            if section is not None:
                log_dict = {f"{section}/{k}": v for k, v in log_dict.items()}
            wandb.log(log_dict)


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
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="none").squeeze(-1)
    bce_loss = balance_loss(bce_loss, task_ids)
    return bce_loss.mean()


class ClassificationTaskDataset(Dataset):
    """
    For n_tasks classes and examples_per_task examples per class, this dataset flattens into (examples_per_task * n_tasks) binary samples.
    Each sample is (features, binary label, queried task id):
      - features: the input features
      - binary label: 1 if the true class matches the queried task id, else 0
      - task id: the queried class
    """

    def __init__(
        self,
        n_tasks: int = 10,
        examples_per_task: int = 100,
        n_features: int = 20,
        mode: str = "mixed",  # "mixed" or "per_task"
        task_id: int = None,  # used only if mode=="per_task"
        random_state: int = 0,
    ):
        self.n_tasks = n_tasks
        self.n_per = examples_per_task
        self.n_feat = n_features
        self.mode = mode
        self.task_id = task_id

        # generate a global multiclass dataset
        N = examples_per_task * n_tasks
        X, y = make_classification(
            n_samples=N,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=0,
            n_classes=n_tasks,
            flip_y=0.01,
            random_state=random_state,
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_global = torch.tensor(y, dtype=torch.long)

        self.total_examples = N
        self.flattened_len = N * n_tasks

    def __len__(self):
        if self.mode == "per_task":
            return self.n_per * self.n_tasks
        else:
            return self.flattened_len

    def __getitem__(self, idx):
        # Map idx to (example_idx, task_id)
        example_idx = idx // self.n_tasks
        tid = idx % self.n_tasks
        x = self.X[example_idx]
        y = (self.y_global[example_idx] == tid).float().unsqueeze(-1)
        return x, y, tid


def collate_fn(batch):
    x, y, task_id = zip(*batch)
    return torch.stack(x), torch.stack(y), torch.tensor(task_id)


def make_dataloader(
    cfg: DictConfig, generator: torch.Generator, task_id: int | None = None
):
    batch_size = cfg.batch_size
    num_examples_per_task = cfg.num_examples_per_task
    n_features = cfg.n_features
    dataset = ClassificationTaskDataset(
        n_tasks=cfg.num_tasks,
        examples_per_task=num_examples_per_task,
        n_features=n_features,
        mode=cfg.train_mode,
        task_id=task_id,
        random_state=cfg.seed,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


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
            try:
                auc = roc_auc_score(y_true, y_pred)
                acc = accuracy_score(y_true, (y_pred > 0.5).astype(float))
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                logger.log_dict(
                    {"val_loss": avg_val_loss, "val_auc": auc, "val_acc": acc},
                    section=f"validation_task_{task_id}",
                    kind="scalar",
                    step=step,
                )
                if logger.cfg.wandb.log:
                    # Save ROC curve to assets directory
                    assets_dir = cfg.assets_dir
                    os.makedirs(assets_dir, exist_ok=True)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.2f})")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve - Task {task_id} - Step {step}")
                    plt.legend(loc="lower right")
                    plot_path = os.path.join(
                        assets_dir, f"roc_curve_task_{task_id}_step_{step}.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()
                    wandb.log(
                        {
                            f"validation_task_{task_id}/roc_curve": wandb.plot.line_series(
                                xs=[fpr],
                                ys=[tpr],
                                keys=["ROC"],
                                title="ROC Curve",
                                xname="FPR",
                            )
                        },
                    )
            except Exception as e:
                logger.log_dict(
                    {"val_loss": avg_val_loss},
                    section=f"validation_task_{task_id}",
                    kind="scalar",
                    step=step,
                )
    pbar.close()


def train_sequential(cfg: DictConfig, generator: torch.Generator):
    """Train a classifier for each task sequentially."""
    model = Dictionary(cfg.num_tasks, cfg.n_features, cfg.hidden_dim).to(cfg.device)
    train_loader = make_dataloader(cfg, generator)
    val_loader = make_dataloader(cfg, generator)

    logger = LoggerAggregator(cfg)

    for task_id in range(cfg.num_tasks):
        print(f"Training task {task_id}")
        train_loader = make_dataloader(cfg, generator, task_id)
        val_loader = make_dataloader(cfg, generator, task_id)
        train_one_task(cfg, model, train_loader, val_loader, task_id, logger)


def train_mixed_batch(cfg: DictConfig, generator: torch.Generator):
    """Train a library of classifers for all tasks in a single batch, step-based."""
    model = Dictionary(cfg.num_tasks).to(cfg.device)
    train_loader = make_dataloader(cfg, generator)
    val_loader = make_dataloader(cfg, generator)
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
            all_task_ids = []
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
                    all_task_ids.append(task_id.cpu().numpy())
            avg_val_loss = val_loss / count
            y_true = np.concatenate(all_y_true).ravel()
            y_pred = np.concatenate(all_y_pred).ravel()
            task_ids = np.concatenate(all_task_ids).ravel()
            logger.log_dict(
                {"val_loss": avg_val_loss}, section="train", kind="scalar", step=step
            )
            # Per-task metrics
            roc_curves = {}
            for tid in np.unique(task_ids):
                mask = task_ids == tid
                y_true_tid = y_true[mask]
                y_pred_tid = y_pred[mask]
                if len(np.unique(y_true_tid)) < 2:
                    # ROC/AUC not defined if only one class present
                    continue
                try:
                    auc = roc_auc_score(y_true_tid, y_pred_tid)
                    acc = accuracy_score(y_true_tid, (y_pred_tid > 0.5).astype(float))
                    fpr, tpr, _ = roc_curve(y_true_tid, y_pred_tid)
                    logger.log_dict(
                        {f"val_auc_task_{tid}": auc, f"val_acc_task_{tid}": acc},
                        section=f"train_task_{tid}",
                        kind="scalar",
                        step=step,
                    )
                    roc_curves[f"task_{tid}"] = (fpr, tpr)
                except Exception as e:
                    continue
            # Overlay ROC curves for all tasks in a single plot
            if logger.cfg.wandb.log and len(roc_curves) > 0:
                xs = [roc_curves[k][0] for k in roc_curves]
                ys = [roc_curves[k][1] for k in roc_curves]
                keys = list(roc_curves.keys())
                # Save overlay ROC curve to assets directory
                assets_dir = cfg.assets_dir
                os.makedirs(assets_dir, exist_ok=True)
                plt.figure()
                for k in roc_curves:
                    fpr, tpr = roc_curves[k]
                    plt.plot(fpr, tpr, label=k)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve Overlay (All Tasks) - Step {step}")
                plt.legend(loc="lower right")
                plot_path = os.path.join(
                    assets_dir, f"roc_curve_overlay_step_{step}.png"
                )
                plt.savefig(plot_path)
                plt.close()
                wandb.log(
                    {
                        "train/roc_curve_overlay": wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title="ROC Curve Overlay (All Tasks)",
                            xname="FPR",
                        )
                    },
                )
    pbar.close()


@hydra.main(config_path="../config", config_name="mixed_batching", version_base=None)
def main(cfg: DictConfig):
    generator = set_seed(cfg.seed)

    if cfg.train_mode == "sequential":
        train_sequential(cfg, generator)
    elif cfg.train_mode == "mixed_batch":
        train_mixed_batch(cfg, generator)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
