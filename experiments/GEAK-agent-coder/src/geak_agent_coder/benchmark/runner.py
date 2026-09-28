"""Fixed engineer-only ToolAgentLoop runner with lazy MultiTune imports."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .models import ModelVariant
from .security import assert_no_held_out_leakage


class MultiTuneUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MultiTuneSymbols:
    ModelBackend: Any
    OpenAIModelBackend: Any
    ToolAgentLoop: Any
    GEAKStatefulTool: Any
    GEAKToolEnvironment: Any
    MultiTuneConfig: Any
    TrajectoryWriter: Any


def load_multitune_symbols(source_root: str | Path | None = None) -> MultiTuneSymbols:
    """Import runtime-heavy MultiTune only when a campaign is actually run."""

    try:
        with _temporary_import_path(source_root):
            runtime = importlib.import_module("multi_tune_agent.runtime")
            geak_tool = importlib.import_module("multi_tune_agent.geak_tool")
            config = importlib.import_module("multi_tune_agent.config")
            trajectory = importlib.import_module("multi_tune_agent.trajectory")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MultiTuneUnavailableError(
            "MultiTune runtime is unavailable; install it or set multitune_source_root"
        ) from exc
    required = {
        "ModelBackend": getattr(runtime, "ModelBackend", None),
        "OpenAIModelBackend": getattr(runtime, "OpenAIModelBackend", None),
        "ToolAgentLoop": getattr(runtime, "ToolAgentLoop", None),
        "GEAKStatefulTool": getattr(geak_tool, "GEAKStatefulTool", None),
        "GEAKToolEnvironment": getattr(geak_tool, "GEAKToolEnvironment", None),
        "MultiTuneConfig": getattr(config, "MultiTuneConfig", None),
        "TrajectoryWriter": getattr(trajectory, "TrajectoryWriter", None),
    }
    missing = sorted(name for name, symbol in required.items() if symbol is None)
    if missing:
        raise MultiTuneUnavailableError(
            "MultiTune is missing required symbols: " + ", ".join(missing)
        )
    return MultiTuneSymbols(**required)


class FixedEngineerRunner:
    """Run one variant/task using the same single-role agent implementation."""

    def __init__(
        self,
        *,
        multitune_config: Mapping[str, Any],
        output_root: str | Path,
        protected_sha256: Sequence[str],
        max_turns: int,
        decode: Mapping[str, Any],
        multitune_source_root: str | Path | None = None,
    ) -> None:
        self.multitune_config = dict(multitune_config)
        self.output_root = Path(output_root)
        self.protected_sha256 = tuple(protected_sha256)
        self.max_turns = int(max_turns)
        self.decode = dict(decode)
        self.multitune_source_root = multitune_source_root

    async def run_task(
        self,
        *,
        variant: ModelVariant,
        task_id: str,
        seed: int,
        model_visible_task: Mapping[str, Any],
    ) -> Path:
        assert_no_held_out_leakage(
            model_visible_task,
            protected_sha256=self.protected_sha256,
            held_out=True,
        )
        symbols = load_multitune_symbols(self.multitune_source_root)
        run_dir = self.output_root / variant.name / task_id / f"seed-{seed}"
        writer = symbols.TrajectoryWriter(run_dir)
        config_values = dict(self.multitune_config)
        # Every task/seed receives separate compilation and workspace state.
        config_values["trajectory_root"] = run_dir / "runtime"
        for key in (
            "geak_root",
            "cases_path",
            "trajectory_root",
            "aiter_root",
            "generated_template_root",
            "sft_dataset_root",
        ):
            if key in config_values:
                config_values[key] = Path(config_values[key]).expanduser()
        config = symbols.MultiTuneConfig(**config_values)
        environment = symbols.GEAKToolEnvironment(config, writer)
        tool = symbols.GEAKStatefulTool(environment)
        backend = symbols.OpenAIModelBackend(
            base_url=variant.base_url,
            model=variant.model,
            max_tokens=int(self.decode.get("max_tokens", 4096)),
            temperature=float(self.decode.get("temperature", 0.0)),
            timeout=float(self.decode.get("timeout_s", 600.0)),
        )
        # Runtime-checking the Protocol documents and enforces the fixed backend API
        # where supported, without importing MultiTune during analysis-only usage.
        model_backend = symbols.ModelBackend
        if getattr(model_backend, "_is_runtime_protocol", False) and not isinstance(
            backend, model_backend
        ):
            raise TypeError("OpenAIModelBackend does not satisfy ModelBackend")
        loop = symbols.ToolAgentLoop(
            backend,
            tool,
            max_assistant_turns=self.max_turns,
            # MultiTune computes final reward after its optional release step, while
            # GEAKStatefulTool rejects reward reads from a released session.
            retain_session=True,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the sole kernel engineer. Inspect, edit, and evaluate "
                    "through the GEAK tool. Do not request other roles or hidden data."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(model_visible_task, sort_keys=True),
            },
        ]
        output = await loop.run(
            messages,
            create_kwargs={
                "case_id": task_id,
                "role": "engineer",
                "establish_baseline": True,
            },
        )
        try:
            writer.append(
                "agent_loop",
                {
                    "task_id": task_id,
                    "variant": variant.name,
                    "seed": seed,
                    "checkpoint_identity": variant.checkpoint_identity,
                    "events": output.events,
                },
                role="engineer",
                phase="benchmark",
            )
            writer.finalize(
                {
                    "task_id": task_id,
                    "variant": variant.name,
                    "seed": seed,
                    "reward_score": output.reward_score,
                    "agent_role": "engineer",
                }
            )
        finally:
            tool.release(output.session_id)
        return writer.path


@contextmanager
def _temporary_import_path(source_root: str | Path | None) -> Iterator[None]:
    if source_root is None:
        yield
        return
    root = Path(source_root).resolve()
    candidate = root / "src" if (root / "src").is_dir() else root
    text = str(candidate)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:
            pass
