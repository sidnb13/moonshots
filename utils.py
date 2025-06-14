import random
import numpy as np
import torch
from omegaconf import OmegaConf

import wandb


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
