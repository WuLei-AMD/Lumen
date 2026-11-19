import hashlib

import pytest

from geak_agent_coder.benchmark.security import (
    HeldOutLeakageError,
    assert_no_held_out_leakage,
)


PROTECTED = "secret reference implementation"
DIGEST = hashlib.sha256(PROTECTED.encode()).hexdigest()


def test_accepts_contract_only_model_view() -> None:
    assert_no_held_out_leakage(
        {
            "task_id": "held-out-1",
            "contract": {"shape": [128, 128], "dtype": "bf16"},
            "reference_hash": DIGEST,
        },
        protected_sha256=[DIGEST],
    )

    assert_no_held_out_leakage(
        {
            "task_id": "held-out-2",
            "contract": {
                "target_kernel_functions": ["gemm_kernel"],
                "entry_point": "gemm",
            },
        },
        protected_sha256=[DIGEST],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"target": "patch"},
        {"nested": {"gold_answer": "code"}},
        {"reference_path": "/protected/reference.py"},
        {"message": '{"oracle": "hidden"}'},
    ],
)
def test_rejects_target_like_fields_recursively(payload) -> None:
    with pytest.raises(HeldOutLeakageError, match="target-like"):
        assert_no_held_out_leakage(payload, protected_sha256=[DIGEST])


def test_rejects_exact_protected_content_even_under_safe_key() -> None:
    with pytest.raises(HeldOutLeakageError, match="matches protected"):
        assert_no_held_out_leakage(
            {"context": PROTECTED},
            protected_sha256=[DIGEST],
        )


def test_missing_protection_catalog_fails_closed() -> None:
    with pytest.raises(HeldOutLeakageError, match="non-empty"):
        assert_no_held_out_leakage({"task_id": "x"}, protected_sha256=[])


def test_non_held_out_payload_is_not_restricted() -> None:
    assert_no_held_out_leakage(
        {"target": PROTECTED},
        protected_sha256=[],
        held_out=False,
    )
