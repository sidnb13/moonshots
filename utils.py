import contextlib
import os
import random
import time

import numpy as np
import torch
import torch.profiler
from omegaconf import OmegaConf

import wandb


def get_total_gradient_bytes(model):
    total_bytes = 0
    for param in model.parameters():
        if param.grad is not None:
            total_bytes += param.grad.numel() * param.grad.element_size()
    return total_bytes


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


def _resolve_device(device_str: str):
    """
    Resolves device string to torch.device and sets the device if needed.
    Handles CPU, CUDA, and MPS devices.

    Args:
        device_str: Device string like 'cpu', 'cuda', 'cuda:0', 'mps'

    Returns:
        torch.device: The resolved device
    """
    if device_str == "cpu":
        return torch.device("cpu")

    elif device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

        # Handle CUDA device specification
        if device_str == "cuda":
            # Use current CUDA device or default to 0
            device_ordinal = torch.cuda.current_device()
        else:
            # Parse device ordinal from string like 'cuda:0'
            parts = device_str.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                device_ordinal = int(parts[1])
            else:
                print(f"Invalid CUDA device string: {device_str}, using device 0")
                device_ordinal = 0

        # Set the device and return
        torch.cuda.set_device(device_ordinal)
        return torch.device(f"cuda:{device_ordinal}")

    elif device_str == "mps":
        if not torch.backends.mps.is_available():
            print("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    else:
        print(f"Unknown device string: {device_str}, falling back to CPU")
        return torch.device("cpu")


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
                log_dict.update({"counters/step": step})

            wandb.log(log_dict, step=step)


class ProfilingContext:
    """
    Context manager class for profiling a training step.
    Times the block, does CUDA sync, and (optionally) runs torch.profiler for a single step.
    After exiting, results are available as ProfilingContext.results.
    """

    _warmup_done = False
    results = None

    def __init__(
        self,
        batch_size,
        step,
        cfg,
        enable_profiler=False,
        profiler_dir=None,
        warmup_steps=3,
    ):
        self.batch_size = batch_size
        self.step = step
        self.cfg = cfg
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir or getattr(
            cfg, "profiler_dir", "profiler_traces"
        )
        self.warmup_steps = warmup_steps
        self.prof = None
        self.start_time = None

        # Skip profiling if already done
        if self.enable_profiler and ProfilingContext._warmup_done:
            self.enable_profiler = False

    def __enter__(self):
        if self.enable_profiler:
            os.makedirs(self.profiler_dir, exist_ok=True)
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + (
                    [torch.profiler.ProfilerActivity.CUDA]
                    if torch.cuda.is_available()
                    else []
                ),
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.profiler_dir
                ),
                record_shapes=True,
                with_stack=True,
            )
            self.prof.__enter__()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time  # type: ignore
        ex_per_sec = self.batch_size / elapsed if elapsed > 0 else float("nan")
        flops_per_sec = None
        if self.prof is not None:
            self.prof.step()
            self.prof.__exit__(None, None, None)
            print(f"[PROFILER] Step {self.step}: trace written to {self.profiler_dir}")
            ProfilingContext._warmup_done = True
            # Get real flops from profiler output
            try:
                # Sum flops from all events if available
                flops = 0
                for evt in self.prof.key_averages():
                    if hasattr(evt, 'flops'):
                        flops += evt.flops
                    elif hasattr(evt, '__dict__') and 'flops' in evt.__dict__:
                        flops += evt.__dict__['flops']
                if flops > 0 and elapsed > 0:
                    flops_per_sec = flops / elapsed
                else:
                    flops_per_sec = -1
            except Exception:
                flops_per_sec = -1
        else:
            flops_per_sec = -1
        # Attach results to the class for access
        ProfilingContext.results = {
            "step_time_cuda_sync": elapsed,
            "examples_per_sec": ex_per_sec,
            "flops_per_sec": flops_per_sec,
        }


@contextlib.contextmanager
def warmup_context(model, train_iter, cfg, training_step_fn, warmup_steps=3):
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
    if not hasattr(warmup_context, "_warmup_done"):
        warmup_context._warmup_done = False

    if not warmup_context._warmup_done:
        print(f"[PROFILER] Warming up for {warmup_steps} steps...")
        model.train()
        for _ in range(warmup_steps):
            batch = next(train_iter)
            training_step_fn(model, batch, cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        print("[PROFILER] Warmup complete")
        warmup_context._warmup_done = True

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
        with_flops=True,
    ) as prof:
        x, y, task_id = example_batch
        with torch.no_grad():
            y_pred = model(x.to(cfg.device), task_id.to(cfg.device))
            if compute_loss is not None:
                compute_loss(y_pred, y.to(cfg.device), task_id.to(cfg.device))
            else:
                pass  # fallback
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


def profile_and_log_flamegraph_backward(
    model, example_batch, cfg, logger, compute_loss=None
):
    """
    Profiles a single backward step, saves a Chrome trace, and logs as a wandb artifact if enabled.
    example_batch: (x, y, task_id) tuple
    compute_loss: function to compute loss (if not provided, expects model to return loss)
    """
    profiler_dir = getattr(cfg, "profiler_dir", "profiler_traces")
    os.makedirs(profiler_dir, exist_ok=True)
    trace_path = os.path.join(profiler_dir, "final_flamegraph_backward.json")
    model.train()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        x, y, task_id = example_batch
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        task_id = task_id.to(cfg.device)
        y_pred = model(x, task_id)
        if compute_loss is not None:
            loss = compute_loss(y_pred, y, task_id)
        else:
            loss = y_pred  # fallback
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    print(f"[PROFILER] Final backward flamegraph trace written to {trace_path}")
    # Log as wandb artifact if enabled
    if hasattr(cfg, "wandb") and getattr(cfg.wandb, "log", False):
        artifact = wandb.Artifact("final_flamegraph_backward", type="profile")
        artifact.add_file(trace_path)
        wandb.log_artifact(artifact)
        print("[WANDB] Final backward flamegraph trace logged as artifact.")
