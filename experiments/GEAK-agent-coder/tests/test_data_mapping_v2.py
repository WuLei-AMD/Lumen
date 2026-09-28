from __future__ import annotations

import pytest

from geak_agent_coder.data import DataContractError, map_row


def test_nested_mapping_source_const_pick_wrap_object_and_adapter() -> None:
    row = {
        "id": "  row-1  ",
        "payload": {"prompt": "Fix it", "patch": "PATCH"},
        "meta": {"language": "python"},
    }
    mapped = map_row(
        row,
        {
            "sample_id": {
                "adapter": "strip",
                "value": {"source": "id"},
            },
            "schema_version": {"const": "general_coding_replay_v1"},
            "input": {
                "object": {
                    "problem": {"source": "payload.prompt"},
                    "metadata": {
                        "wrap": {
                            "key": "language",
                            "value": {
                                "pick": [
                                    {"source": "missing"},
                                    {"source": "meta.language"},
                                ]
                            },
                        }
                    },
                }
            },
        },
    )
    assert mapped == {
        "sample_id": "row-1",
        "schema_version": "general_coding_replay_v1",
        "input": {
            "problem": "Fix it",
            "metadata": {"language": "python"},
        },
    }


def test_mapping_rejects_missing_fields_and_uncontrolled_plugins() -> None:
    with pytest.raises(DataContractError, match="missing source field"):
        map_row({}, {"sample_id": {"source": "missing"}})
    with pytest.raises(DataContractError, match="unknown adapter"):
        map_row(
            {"value": "x"},
            {
                "sample_id": {
                    "adapter": "os.system",
                    "value": {"source": "value"},
                }
            },
        )
    with pytest.raises(DataContractError, match="exactly one operator"):
        map_row({}, {"bad": {"source": "x", "const": "y"}})
    with pytest.raises(DataContractError, match="unsafe source path"):
        map_row({"__class__": "x"}, {"bad": {"source": "__class__"}})
