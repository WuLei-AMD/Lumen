"""Convert ALL data files in the HF repo to unified messages format."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import subprocess

REPO = "Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP"
OUT_DIR = Path("/home/danyzhan/geak_sft_dataset/hf_unified")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer optimizing Triton kernels "
    "for AMD MI300X (gfx942)."
)


def convert_sample(sample):
    """Convert geak_kernel_sft_v1 to messages format."""
    inp = sample.get("input", {})
    tt = sample.get("task_type", "unknown")
    patch = sample.get("output", {}).get("patch", "")
    contract = inp.get("contract", {})
    op = contract.get("operator", "unknown")
    parent_src = inp.get("parent_source", {}).get("kernel.py", "")

    if tt == "direction_conditioned":
        direction = inp.get("direction", {})
        instructions = direction.get("instructions", "Optimize this kernel.") if isinstance(direction, dict) else str(direction)
        user_content = (
            f"Optimize the following {op} Triton kernel for AMD MI300X (gfx942).\n\n"
            f"Direction: {instructions}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "error_recovery":
        ef = inp.get("error_feedback", {})
        vr = ef.get("verify_result", {}) if isinstance(ef, dict) else {}
        eval_info = vr.get("evaluation", {}) if isinstance(vr, dict) else {}
        stderr = ""
        if isinstance(eval_info, dict):
            compile_info = eval_info.get("compile", {})
            if isinstance(compile_info, dict):
                stderr = compile_info.get("stderr", "")
            if not stderr:
                correct_info = eval_info.get("correctness", {})
                if isinstance(correct_info, dict):
                    stderr = correct_info.get("stderr", "")
        error_msg = str(stderr)[:500] if stderr else "Unknown error"
        user_content = (
            f"Fix the following {op} Triton kernel.\n\n"
            f"Error: {error_msg}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "profile_guided":
        profile = inp.get("profile")
        profile_str = json.dumps(profile, indent=2)[:500] if profile else "No profiling data"
        user_content = (
            f"Optimize the following {op} Triton kernel based on profiling data.\n\n"
            f"Profile:\n{profile_str}\n\n"
            f"```python\n{parent_src}\n```"
        )
    elif tt == "regression_balance":
        constraints = inp.get("regression_constraints")
        constraints_str = json.dumps(constraints, indent=2)[:300] if constraints else ""
        user_content = (
            f"Optimize the following {op} Triton kernel while avoiding regression.\n\n"
            f"Constraints: {constraints_str}\n\n"
            f"```python\n{parent_src}\n```"
        )
    else:
        user_content = (
            f"Optimize the following {op} Triton kernel for AMD MI300X (gfx942).\n\n"
            f"```python\n{parent_src}\n```"
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"```diff\n{patch}\n```"},
        ],
        "task_type": tt,
        "operator": op,
        "source_sample_id": sample.get("sample_id", ""),
        "schema_version": "messages_v1",
    }


def convert_file(filename):
    """Download, convert, save."""
    try:
        path = hf_hub_download(REPO, filename, repo_type="dataset")
    except Exception as e:
        print(f"  Skip {filename}: {e}")
        return None

    out_path = OUT_DIR / filename
    converted = 0
    skipped = 0

    with open(path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if "messages" in sample:
                fout.write(json.dumps(sample) + "\n")
                converted += 1
            elif "input" in sample and "output" in sample:
                new = convert_sample(sample)
                fout.write(json.dumps(new) + "\n")
                converted += 1
            else:
                skipped += 1

    print(f"  {filename}: {converted} converted, {skipped} skipped")
    return out_path


def main():
    files_to_convert = ["dev.jsonl", "samples.jsonl"]

    for fname in files_to_convert:
        print(f"Converting {fname}...")
        out_path = convert_file(fname)
        if out_path:
            print(f"  Uploading {fname}...")
            r = subprocess.run(
                ["huggingface-cli", "upload", REPO,
                 str(out_path), fname,
                 "--repo-type", "dataset"],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                print(f"  Upload OK")
            else:
                print(f"  Upload failed: {r.stderr[:200]}")

    # train.jsonl already unified (done in previous step)
    print("\ntrain.jsonl: already unified (429 samples)")
    print("Done!")


if __name__ == "__main__":
    main()
