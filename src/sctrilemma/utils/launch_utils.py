"""Shared launch-time utilities for the official training entrypoint."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any, Dict

import torch
from omegaconf import DictConfig


def _summarize_git_status(status_text: str) -> str:
    counts = {"M": 0, "A": 0, "D": 0, "R": 0, "U": 0, "O": 0}
    for line in status_text.splitlines():
        if not line:
            continue
        code = line[:2]
        if code == "??":
            counts["U"] += 1
            continue
        if "R" in code:
            counts["R"] += 1
        elif "A" in code:
            counts["A"] += 1
        elif "D" in code:
            counts["D"] += 1
        elif "M" in code:
            counts["M"] += 1
        else:
            counts["O"] += 1
    return (
        f"M:{counts['M']} A:{counts['A']} D:{counts['D']} "
        f"R:{counts['R']} U:{counts['U']} O:{counts['O']}"
    )


def _get_git_code_state(repo_dir: str) -> Dict[str, Any]:
    def _run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()

    commit = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    diff_names = _run_git(["diff", "--name-status"])
    diff_hash = hashlib.sha256(diff_names.encode("utf-8")).hexdigest()

    return {
        "code.commit": commit,
        "code.is_dirty": bool(status),
        "code.status_summary": _summarize_git_status(status),
        "code.diff_hash": diff_hash,
        "code.state_available": True,
    }


def _format_step_tag(step: int) -> str:
    if step < 1000:
        return str(step)
    step_k = step / 1000.0
    if step % 1000 == 0:
        return f"{int(step_k)}K"
    return f"{step_k:.1f}".rstrip("0").rstrip(".") + "K"


def _short_experiment_name(experiment_name: str) -> str:
    return experiment_name.split("_", 1)[0]


def _build_run_name(
    experiment_name: str,
    resume_global_step: int | None = None,
    max_steps: int | None = None,
) -> str:
    short_name = _short_experiment_name(experiment_name)
    if resume_global_step is not None and resume_global_step > 0:
        target = max_steps if max_steps else resume_global_step
        return f"{short_name}-resume-{_format_step_tag(target)}"
    return short_name


def _apply_perf_flags(cfg: DictConfig, local_rank: int) -> None:
    """Apply optional perf flags. Defaults preserve legacy behavior.

    All toggles live under `cfg.perf.*`. If `cfg.perf` is absent or a field is
    missing, the legacy setting is kept.
    """
    perf = cfg.get("perf", {}) if hasattr(cfg, "get") else {}
    matmul_precision = perf.get("matmul_precision", "medium") if perf else "medium"
    torch.set_float32_matmul_precision(matmul_precision)

    if not perf:
        return

    flash = perf.get("enable_flash_sdp", None)
    if flash is not None:
        torch.backends.cuda.enable_flash_sdp(bool(flash))
    mem_eff = perf.get("enable_mem_efficient_sdp", None)
    if mem_eff is not None:
        torch.backends.cuda.enable_mem_efficient_sdp(bool(mem_eff))
    math_sdp = perf.get("enable_math_sdp", None)
    if math_sdp is not None:
        torch.backends.cuda.enable_math_sdp(bool(math_sdp))
    cudnn_sdp = perf.get("enable_cudnn_sdp", None)
    if cudnn_sdp is not None and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(bool(cudnn_sdp))

    if perf.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if perf.get("cudnn_allow_tf32", False):
        torch.backends.cudnn.allow_tf32 = True
    if perf.get("cuda_matmul_allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    if local_rank == 0:
        print(
            f"[PERF] matmul_precision={matmul_precision} "
            f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
            f"mem_eff_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
            f"math_sdp={torch.backends.cuda.math_sdp_enabled()} "
            f"cudnn_sdp={getattr(torch.backends.cuda, 'cudnn_sdp_enabled', lambda: 'n/a')()}"
        )
