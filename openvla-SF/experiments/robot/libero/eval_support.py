"""Dependency-free guardrails and incremental reports for LIBERO evaluation."""

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from itertools import islice
import json
import os
from pathlib import Path
import tempfile


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_evaluation_config(cfg, *, task_suites, action_chunk_size):
    """Reject unusable configurations before model initialization."""
    checkpoint = cfg.pretrained_checkpoint
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError("pretrained_checkpoint must be a non-empty path or Hugging Face model ID")
    if cfg.model_family != "openvla":
        raise ValueError(f"Unsupported model_family: {cfg.model_family}")
    if cfg.task_suite_name not in task_suites:
        raise ValueError(f"Invalid task suite: {cfg.task_suite_name}")
    if "image_aug" in str(checkpoint) and not cfg.center_crop:
        raise ValueError("center_crop must be enabled for image_aug checkpoints")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")
    if cfg.use_l1_regression and cfg.use_diffusion:
        raise ValueError("use_l1_regression and use_diffusion cannot both be enabled")
    _integer(action_chunk_size, "action_chunk_size", 1)
    for name in ("num_trials_per_task", "num_open_loop_steps", "env_img_res", "num_images_in_input"):
        _integer(getattr(cfg, name), name, 1)
    _integer(cfg.num_steps_wait, "num_steps_wait", 0)
    _integer(cfg.seed, "seed", 0)
    if cfg.seed >= 2**32:
        raise ValueError("seed must be less than 2**32 for NumPy compatibility")
    if cfg.num_open_loop_steps > action_chunk_size:
        raise ValueError(f"num_open_loop_steps must be <= model action chunk size ({action_chunk_size})")
    if cfg.use_diffusion:
        _integer(cfg.num_diffusion_steps_train, "num_diffusion_steps_train", 1)
        _integer(cfg.num_diffusion_steps_inference, "num_diffusion_steps_inference", 1)


def select_action_chunk(actions, num_steps):
    """Take the earliest predicted actions, preserving their temporal order."""
    _integer(num_steps, "num_open_loop_steps", 1)
    chunk = list(islice(actions, num_steps))
    if not chunk:
        raise ValueError("Policy returned an empty action chunk")
    return chunk


def summarize_episodes(episodes):
    """Errors count as attempted failures; skipped demonstrations do not."""
    counts = Counter()
    tasks = {}
    seen = set()
    for episode in episodes:
        task_id, index = episode["task_id"], episode["episode_index"]
        _integer(task_id, "task_id", 0)
        _integer(index, "episode_index", 0)
        key = (task_id, index)
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}")
        seen.add(key)
        outcome = episode["outcome"]
        if outcome not in ("success", "failure", "error", "skipped"):
            raise ValueError(f"Invalid episode outcome: {outcome}")
        description = episode["task_description"]
        if not isinstance(description, str):
            raise ValueError("task_description must be a string")
        task = tasks.setdefault(task_id, {"task_id": task_id, "task_description": description, "counts": Counter()})
        if task["task_description"] != description:
            raise ValueError(f"Conflicting descriptions for task {task_id}")
        counts[outcome] += 1
        task["counts"][outcome] += 1

    def metrics(counter):
        completed = sum(counter[outcome] for outcome in ("success", "failure", "error"))
        return {"completed": completed, "successes": counter["success"],
                "failures": counter["failure"], "errors": counter["error"],
                "skipped": counter["skipped"],
                "success_rate": counter["success"] / completed if completed else None}

    return metrics(counts), [
        {"task_id": task_id, "task_description": task["task_description"], **metrics(task["counts"])}
        for task_id, task in sorted(tasks.items())
    ]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class EvaluationReport:
    """Atomically save progress after every episode, including skips and errors."""

    def __init__(self, path, run_id, cfg, *, data_kind="evaluation"):
        self.path = Path(path)
        config = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        self.payload = {
            "schema_version": 1, "data_kind": data_kind, "run_id": run_id,
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
            "episodes": [],
        }
        self.save()

    def save(self):
        summary, tasks = summarize_episodes(self.payload["episodes"])
        _atomic_json(self.path, {**self.payload, "summary": summary, "tasks": tasks})

    def record(self, task_id, episode_index, task_description, *, success=False, error=None, skipped=False):
        if self.payload["status"] != "running":
            raise ValueError("Cannot add episodes to a finished report")
        if not isinstance(success, bool) or not isinstance(skipped, bool):
            raise ValueError("success and skipped must be booleans")
        if (success and error is not None) or (skipped and (success or error is not None)):
            raise ValueError("Episode outcomes must be mutually exclusive")
        episode = {"task_id": task_id, "episode_index": episode_index,
                   "task_description": task_description,
                   "outcome": "skipped" if skipped else "error" if error is not None else "success" if success else "failure"}
        if error is not None:
            episode["error"] = str(error)
        # Validate before mutating the report, so rejected records leave it usable.
        summarize_episodes([*self.payload["episodes"], episode])
        self.payload["episodes"].append(episode)
        self.save()

    def finish(self, error=None):
        if self.payload["status"] != "running":
            raise ValueError("Report is already finished")
        self.payload["status"] = "completed" if error is None else "aborted"
        self.payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            self.payload["error"] = str(error)
        self.save()
