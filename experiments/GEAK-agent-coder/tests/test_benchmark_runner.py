import asyncio
import hashlib

import pytest

from geak_agent_coder.benchmark.models import ModelVariant
from geak_agent_coder.benchmark.runner import (
    FixedEngineerRunner,
    MultiTuneUnavailableError,
    load_multitune_symbols,
)
from geak_agent_coder.benchmark.security import HeldOutLeakageError


def test_multitune_import_is_lazy_and_has_clear_error(monkeypatch) -> None:
    def unavailable(_name):
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(
        "geak_agent_coder.benchmark.runner.importlib.import_module", unavailable
    )
    with pytest.raises(MultiTuneUnavailableError, match="unavailable"):
        load_multitune_symbols()


def test_runner_checks_leakage_before_loading_runtime(monkeypatch, tmp_path) -> None:
    protected = "hidden target"
    digest = hashlib.sha256(protected.encode()).hexdigest()
    runner = FixedEngineerRunner(
        multitune_config={},
        output_root=tmp_path,
        protected_sha256=[digest],
        max_turns=2,
        decode={},
    )

    def must_not_import(_root):
        raise AssertionError("runtime import happened before leakage gate")

    monkeypatch.setattr(
        "geak_agent_coder.benchmark.runner.load_multitune_symbols",
        must_not_import,
    )
    with pytest.raises(HeldOutLeakageError):
        asyncio.run(
            runner.run_task(
                variant=ModelVariant("base", "http://unused", "m", "checkpoint"),
                task_id="held-out",
                seed=1,
                model_visible_task={"context": protected},
            )
        )
