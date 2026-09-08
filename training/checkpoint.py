"""Versioned checkpoint I/O for the SAC trainer."""

from __future__ import annotations

import inspect
import platform
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sac import SAC

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    epoch: int
    history: dict[str, Any]
    metadata: dict[str, Any]


def _agent_state(agent: SAC) -> dict[str, Any]:
    return {
        "q1_function": agent.q1_function.state_dict(),
        "q2_function": agent.q2_function.state_dict(),
        "value_function": agent.value_function.state_dict(),
        "target_value_function": agent.target_value_function.state_dict(),
        "policy_function": agent.policy_function.state_dict(),
        "log_alpha": agent.log_alpha.detach(),
        "q1_optimizer": agent.q1_function_optimizer.state_dict(),
        "q2_optimizer": agent.q2_function_optimizer.state_dict(),
        "value_optimizer": agent.value_function_optimizer.state_dict(),
        "policy_optimizer": agent.policy_function_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),
    }


def _continuation_state(agent: SAC):
    buffer = agent.replay_buffer
    random_state = np.random.get_state()
    return {
        "replay": {
            "capacity": buffer.capacity,
            "batch_size": buffer.batch_size,
            "index": buffer.index,
            "current_size": len(buffer),
            "rng": buffer.rng.bit_generator.state,
            "arrays": {
                name: torch.from_numpy(getattr(buffer, name)[: len(buffer)])
                for name in ("S", "A", "R", "S_prime", "done")
            },
        },
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [
            random_state[0],
            torch.from_numpy(random_state[1].astype(np.int64)),
            *random_state[2:],
        ],
    }


def _restore_continuation(agent: SAC, state):
    saved = state["replay"]
    buffer = agent.replay_buffer
    if saved["capacity"] != buffer.capacity or saved["batch_size"] != buffer.batch_size:
        raise ValueError(
            "Resume requires the checkpoint's replay capacity and batch size"
        )
    size = int(saved["current_size"])
    if not 0 <= size <= buffer.capacity or not 0 <= saved["index"] < buffer.capacity:
        raise ValueError("Invalid replay checkpoint indices")
    for name, tensor in saved["arrays"].items():
        if name not in {"S", "A", "R", "S_prime", "done"}:
            raise ValueError("Invalid replay checkpoint array")
        target = getattr(buffer, name)
        if tensor.shape != target[:size].shape:
            raise ValueError(f"Replay checkpoint shape mismatch for {name}")
        target[:size] = tensor.cpu().numpy()
    buffer.index, buffer.current_size = int(saved["index"]), size
    buffer.rng.bit_generator.state = saved["rng"]
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    nr = state["numpy_rng"]
    np.random.set_state((nr[0], nr[1].cpu().numpy().astype(np.uint32), *nr[2:]))


def save_checkpoint(
    path: str | Path,
    agent: SAC,
    *,
    epoch: int,
    history: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer and run metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": "SAC",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        **dict(metadata or {}),
    }
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": checkpoint_metadata,
        "agent": _agent_state(agent),
        "training": {"epoch": int(epoch), "history": dict(history)},
        "continuation": _continuation_state(agent),
    }

    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        with temporary_path.open("wb") as checkpoint_file:
            torch.save(payload, checkpoint_file)
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint_path


def _torch_load(
    file_object: Any, map_location: torch.device | str, trust_legacy: bool = False
) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    # New checkpoints consist only of tensors and builtin data containers.
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = not trust_legacy
    return torch.load(file_object, **kwargs)


def _legacy_agent_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate checkpoints written by the original training script."""

    return {
        "q1_function": payload["q1_function_state_dict"],
        "q2_function": payload["q2_function_state_dict"],
        "value_function": payload["value_function_state_dict"],
        "target_value_function": payload.get("target_value_function_state_dict"),
        "policy_function": payload["policy_function_state_dict"],
        "log_alpha": payload["log_alpha"],
        "q1_optimizer": payload.get("q1_optimizer_state_dict"),
        "q2_optimizer": payload.get("q2_optimizer_state_dict"),
        "value_optimizer": payload.get("value_optimizer_state_dict"),
        "policy_optimizer": payload.get("policy_optimizer_state_dict"),
        "alpha_optimizer": payload.get("alpha_optimizer_state_dict"),
    }


def _restore_agent(
    agent: SAC, state: Mapping[str, Any], *, load_optimizers: bool
) -> None:
    agent.q1_function.load_state_dict(state["q1_function"])
    agent.q2_function.load_state_dict(state["q2_function"])
    agent.value_function.load_state_dict(state["value_function"])
    target_state = state.get("target_value_function")
    if target_state is None:
        target_state = state["value_function"]
    agent.target_value_function.load_state_dict(target_state)
    agent.policy_function.load_state_dict(state["policy_function"])
    with torch.no_grad():
        alpha = torch.as_tensor(
            state["log_alpha"], dtype=agent.log_alpha.dtype, device=agent.device
        )
        agent.log_alpha.copy_(alpha)

    if not load_optimizers:
        return
    optimizer_states = (
        (agent.q1_function_optimizer, state.get("q1_optimizer")),
        (agent.q2_function_optimizer, state.get("q2_optimizer")),
        (agent.value_function_optimizer, state.get("value_optimizer")),
        (agent.policy_function_optimizer, state.get("policy_optimizer")),
        (agent.alpha_optimizer, state.get("alpha_optimizer")),
    )
    for optimizer, optimizer_state in optimizer_states:
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)


def load_checkpoint(
    path: str | Path,
    agent: SAC,
    *,
    map_location: torch.device | str = "cpu",
    load_optimizers: bool = True,
    trust_legacy: bool = False,
    expected_config: Mapping[str, Any] | None = None,
) -> LoadedCheckpoint:
    """Load a current or legacy checkpoint into ``agent``.

    ``map_location`` is explicit so CPU evaluation can load checkpoints created
    on an accelerator.
    """

    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as checkpoint_file:
        payload = _torch_load(
            checkpoint_file, map_location=map_location, trust_legacy=trust_legacy
        )
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")

    if "agent" in payload:
        format_version = int(payload.get("format_version", 0))
        if format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {format_version}; "
                f"maximum supported is {CHECKPOINT_FORMAT_VERSION}."
            )
        state = payload["agent"]
        training = payload.get("training", {})
        metadata = dict(payload.get("metadata", {}))
        epoch = int(training.get("epoch", 0))
        history = dict(training.get("history", {}))
    else:
        state = _legacy_agent_state(payload)
        metadata = {"format_version": 0, "legacy": True, "algorithm": "SAC"}
        # Original checkpoints recorded a zero-based epoch index.
        epoch = int(payload.get("epoch", 0)) + 1
        history = dict(payload.get("history", {}))

    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint agent state must be a mapping.")
    if expected_config is not None and "config" in metadata:
        saved_config = metadata["config"]
        keys = [
            "env_id",
            "car_id",
            "reward_weights",
            "energy_factor",
            "speed_limit_positions",
            "speed_limits",
            "hidden_layer_sizes",
        ]
        if load_optimizers:
            keys.extend(
                (
                    "seed",
                    "num_steps_per_epoch",
                    "replay_buffer_capacity",
                    "optimization_batch",
                    "adam_lr",
                    "discount_factor_gamma",
                    "soft_update_factor_tau",
                )
            )
        for key in keys:
            if saved_config.get(key) != expected_config.get(key):
                raise ValueError(f"Checkpoint configuration mismatch for {key}")
    _restore_agent(agent, state, load_optimizers=load_optimizers)
    if load_optimizers and "continuation" in payload:
        _restore_continuation(agent, payload["continuation"])
    return LoadedCheckpoint(epoch=epoch, history=history, metadata=metadata)
