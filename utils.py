import contextlib
import os
import random
import time

import numpy as np
import torch
import torch.profiler
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


def get_ordinal_device(device_str: str):
    """
    Returns the ordinal int of the cuda/accelerator device if specified (e.g., 'cuda:0' -> 0),
    or tries to infer the current device if 'cuda' (no ordinal) is given.
    Returns None for 'cpu' or unknown.
    """
    if device_str == "cpu":
        return None
    if device_str.startswith("cuda"):
        parts = device_str.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        elif device_str == "cuda":
            # Try to get the current CUDA device ordinal
            if torch.cuda.is_available():
                try:
                    return torch.cuda.current_device()
                except Exception:
                    return 0  # fallback to 0 if something goes wrong
            else:
                return None
        else:
            # malformed string
            return None
    # Could add support for other accelerators here
    return None


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
            wandb.run.name = cfg.wandb.run_name  # type: ignore

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


@contextlib.contextmanager
def ProfilingContext(
    batch_size, step, cfg, enable_profiler=False, profiler_dir=None, warmup_steps=3
):
    """
    Context manager for profiling a training step.
    Times the block, does CUDA sync, and (optionally) runs torch.profiler for a single step.
    Returns a dict with timing and throughput info.
    """
    if profiler_dir is None:
        profiler_dir = getattr(cfg, "profiler_dir", "profiler_traces")

    # Track if we've done warmup (static variable)
    if not hasattr(ProfilingContext, "_warmup_done"):
        ProfilingContext._warmup_done = False

    # Skip profiling if already done
    if enable_profiler and ProfilingContext._warmup_done:
        enable_profiler = False

    if enable_profiler:
        os.makedirs(profiler_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + (
                [torch.profiler.ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else []
            ),
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            with_stack=True,
        )
        prof.__enter__()
    else:
        prof = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        ex_per_sec = batch_size / elapsed if elapsed > 0 else float("nan")
        # Try to get flops/sec if possible
        flops_per_step = None
        flops_per_sec = None
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                flops_per_step = (
                    torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).multi_processor_count
                    * 2
                    * batch_size
                )
                flops_per_sec = (
                    flops_per_step / elapsed if elapsed > 0 else float("nan")
                )
            except Exception:
                flops_per_step = None
                flops_per_sec = None
        if prof is not None:
            prof.step()
            prof.__exit__(None, None, None)
            print(f"[PROFILER] Step {step}: trace written to {profiler_dir}")
            ProfilingContext._warmup_done = True
        # Attach results to the context manager for access
        ProfilingContext.results = {
            "step_time_cuda_sync": elapsed,
            "examples_per_sec": ex_per_sec,
            "flops_per_sec": flops_per_sec if flops_per_sec is not None else -1,
        }


@contextlib.contextmanager
def WarmupContext(model, train_iter, cfg, training_step_fn, warmup_steps=3):
    """
    Context manager for warming up a model before profiling.
    Does warmup steps to ensure model is in steady state, then yields for profiling.

    Args:
        model: The model to warm up
        train_iter: Iterator that yields batches
        cfg: Config object
        training_step_fn: Function that takes (model, batch, cfg) and returns loss
        warmup_steps: Number of warmup steps
    """
    # Track if we've done warmup (static variable)
    if not hasattr(WarmupContext, "_warmup_done"):
        WarmupContext._warmup_done = False

    if not WarmupContext._warmup_done:
        print(f"[PROFILER] Warming up for {warmup_steps} steps...")
        model.train()
        for _ in range(warmup_steps):
            batch = next(train_iter)
            training_step_fn(model, batch, cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        print("[PROFILER] Warmup complete")
        WarmupContext._warmup_done = True

    try:
        yield
    finally:
        pass  # No cleanup needed


def do_warmup(model, train_iter, cfg, training_step_fn, warmup_steps=3):
    """
    Do warmup steps before profiling to ensure model is in steady state.

    Args:
        model: The model to warm up
        train_iter: Iterator that yields batches
        cfg: Config object
        training_step_fn: Function that takes (model, batch, cfg) and returns loss
        warmup_steps: Number of warmup steps
    """
    print(f"[PROFILER] Warming up for {warmup_steps} steps...")
    model.train()
    for _ in range(warmup_steps):
        batch = next(train_iter)
        training_step_fn(model, batch, cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("[PROFILER] Warmup complete")


def profile_and_log_flamegraph(model, example_batch, cfg, logger, compute_loss=None):
    """
    Profiles a single step, saves a Chrome trace, and logs as a wandb artifact if enabled.
    example_batch: (x, y, task_id) tuple
    compute_loss: function to compute loss (if not provided, expects model to return loss)
    """
    profiler_dir = getattr(cfg, "profiler_dir", "profiler_traces")
    os.makedirs(profiler_dir, exist_ok=True)
    trace_path = os.path.join(profiler_dir, "final_flamegraph.json")
    model.eval()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        x, y, task_id = example_batch
        with torch.no_grad():
            y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
            if compute_loss is not None:
                loss = compute_loss(y_pred, y.to(cfg.device), task_id.to(cfg.device))
            else:
                loss = y_pred  # fallback
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    print(f"[PROFILER] Final flamegraph trace written to {trace_path}")
    # Log as wandb artifact if enabled
    if hasattr(cfg, "wandb") and getattr(cfg.wandb, "log", False):
        artifact = wandb.Artifact("final_flamegraph", type="profile")
        artifact.add_file(trace_path)
        wandb.log_artifact(artifact)
        print("[WANDB] Final flamegraph trace logged as artifact.")
