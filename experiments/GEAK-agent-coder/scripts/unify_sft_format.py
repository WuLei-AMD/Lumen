"""Unify all SFT data to messages chat format and upload to HF."""
import json
from pathlib import Path

EXISTING = Path("/home/danyzhan/geak_sft_dataset/phase1-production-wave-300-v2/processed/train.jsonl")
COMPLETE = Path("/home/danyzhan/geak_sft_dataset/complete_impl_v1.jsonl")
OUTPUT = Path("/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer optimizing Triton kernels "
    "for AMD MI300X (gfx942)."
)


def patch_to_messages(sample):
    """Convert geak_kernel_sft_v1 format to messages chat format."""
    inp = sample.get("input", {})
    tt = sample.get("task_type", "unknown")
    patch = sample.get("output", {}).get("patch", "")
    contract = inp.get("contract", {})
    op = contract.get("operator", "unknown")
    parent_src = inp.get("parent_source", {}).get("kernel.py", "")

    # Build user message based on task type
    if tt == "direction_conditioned":
        direction = inp.get("direction", {})
        instructions = direction.get("instructions", "Optimize this kernel.")
        user_content = (
            f"Optimize the following {op} Triton kernel for AMD MI300X (gfx942).\n\n"
            f"Direction: {instructions}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "cold_start":
        user_content = (
            f"Optimize the following {op} Triton kernel for AMD MI300X (gfx942).\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "error_recovery":
        ef = inp.get("error_feedback", {})
        vr = ef.get("verify_result", {}) if isinstance(ef, dict) else {}
        eval_info = vr.get("evaluation", {})
        compile_ok = eval_info.get("compile", {}).get("ok", "?")
        correct_ok = eval_info.get("correctness", {}).get("ok", "?")
        stderr = eval_info.get("compile", {}).get("stderr", "")
        if not stderr:
            stderr = eval_info.get("correctness", {}).get("stderr", "")
        error_msg = stderr[:500] if stderr else "Unknown error"

        user_content = (
            f"Fix the following {op} Triton kernel. "
            f"Compile: {compile_ok}, Correctness: {correct_ok}.\n\n"
            f"Error: {error_msg}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "profile_guided":
        profile = inp.get("profile", {})
        profile_str = json.dumps(profile, indent=2)[:500] if profile else "No profiling data"
        user_content = (
            f"Optimize the following {op} Triton kernel based on profiling data.\n\n"
            f"Profile:\n{profile_str}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "regression_balance":
        constraints = inp.get("regression_constraints", {})
        constraints_str = json.dumps(constraints, indent=2)[:300] if constraints else ""
        user_content = (
            f"Optimize the following {op} Triton kernel while avoiding regression.\n\n"
            f"Constraints: {constraints_str}\n\n"
            f"```python\n{parent_src}\n```"
        )
    else:
        user_content = f"Optimize this {op} kernel:\n\n```python\n{parent_src}\n```"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"```diff\n{patch}\n```"},
        ],
        "task_type": tt,
        "operator": op,
        "source_sample_id": sample.get("sample_id", ""),
        "data_source": "patch_v1",
        "schema_version": "messages_v1",
    }


def main():
    all_samples = []

    # Convert existing patch data
    with open(EXISTING) as f:
        for line in f:
            old = json.loads(line)
            new = patch_to_messages(old)
            all_samples.append(new)
    print(f"Converted {len(all_samples)} patch samples to messages format")

    # Add complete implementation data (already in messages format)
    impl_count = 0
    with open(COMPLETE) as f:
        for line in f:
            d = json.loads(line)
            d["data_source"] = "complete_impl_v1"
            d["schema_version"] = "messages_v1"
            all_samples.append(d)
            impl_count += 1
    print(f"Added {impl_count} complete implementation samples")

    # Stats
    print(f"\nTotal: {len(all_samples)} samples")
    by_type = {}
    by_source = {}
    for s in all_samples:
        tt = s.get("task_type", "?")
        src = s.get("data_source", "?")
        by_type[tt] = by_type.get(tt, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1

    print("\nBy task type:")
    for tt, count in sorted(by_type.items()):
        print(f"  {tt}: {count}")
    print("\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    # Verify all samples have messages format
    for s in all_samples:
        assert "messages" in s, f"Missing messages: {s.get('task_type')}"
        assert len(s["messages"]) >= 3, f"Too few messages: {len(s['messages'])}"
        assert s["messages"][0]["role"] == "system"
        assert s["messages"][1]["role"] == "user"
        assert s["messages"][2]["role"] == "assistant"
    print("\nFormat validation: ALL samples have messages format ✓")

    # Save
    with open(OUTPUT, "w") as f:
        for item in all_samples:
            f.write(json.dumps(item) + "\n")
    print(f"\nSaved to {OUTPUT}")

    # Upload
    print("\nUploading to HuggingFace...")
    import subprocess
    r = subprocess.run(
        ["huggingface-cli", "upload",
         "Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP",
         str(OUTPUT), "train.jsonl",
         "--repo-type", "dataset"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode == 0:
        print("Upload OK")
    else:
        print(f"Upload failed: {r.stderr[:200]}")


if __name__ == "__main__":
    main()
