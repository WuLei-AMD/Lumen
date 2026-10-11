"""Generate complete_implementation from ALL wave-2000 using robust patch apply."""
import json
import subprocess
import tempfile
import re
from pathlib import Path
from collections import defaultdict

SFT_DATA = Path("/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/train.jsonl")
OUTPUT = Path("/home/danyzhan/geak_sft_dataset/complete_impl_v2.jsonl")

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer. Write a high-performance "
    "Triton kernel for the specified operator on AMD MI300X (gfx942). "
    "Include @triton.autotune with appropriate configs."
)


def parse_hunks(patch_text):
    """Parse unified diff into hunks."""
    hunks = []
    current_hunk = None

    for line in patch_text.splitlines():
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {
                    "old_start": int(m.group(1)),
                    "old_count": int(m.group(2) or 1),
                    "new_start": int(m.group(3)),
                    "new_count": int(m.group(4) or 1),
                    "lines": [],
                }
        elif current_hunk is not None and not line.startswith("---") and not line.startswith("+++"):
            current_hunk["lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)
    return hunks


def fuzzy_apply(original: str, patch_text: str) -> str | None:
    """Apply patch with fuzzy line matching."""
    hunks = parse_hunks(patch_text)
    if not hunks:
        return None

    orig_lines = original.splitlines()
    result_lines = list(orig_lines)
    offset = 0

    for hunk in hunks:
        old_start = hunk["old_start"] - 1 + offset

        # Extract context/remove lines to find match position
        context_lines = []
        for l in hunk["lines"]:
            if l.startswith(" ") or l.startswith("-"):
                context_lines.append(l[1:] if len(l) > 1 else "")

        # Try exact position first
        best_pos = old_start
        best_score = 0

        # Search around expected position
        search_range = range(max(0, old_start - 50), min(len(result_lines), old_start + 50))
        for pos in search_range:
            score = 0
            for j, ctx in enumerate(context_lines):
                if pos + j < len(result_lines):
                    if result_lines[pos + j].rstrip() == ctx.rstrip():
                        score += 1
            if score > best_score:
                best_score = score
                best_pos = pos

        if best_score < max(1, len(context_lines) * 0.3):
            continue

        # Apply hunk at best_pos
        new_lines = []
        remove_count = 0
        for l in hunk["lines"]:
            if l.startswith("+"):
                new_lines.append(l[1:] if len(l) > 1 else "")
            elif l.startswith("-"):
                remove_count += 1
            elif l.startswith(" "):
                new_lines.append(l[1:] if len(l) > 1 else "")
                remove_count += 1

        result_lines[best_pos:best_pos + remove_count] = new_lines
        offset += len(new_lines) - remove_count

    result = "\n".join(result_lines)
    if not result.endswith("\n") and original.endswith("\n"):
        result += "\n"
    return result


def try_apply(parent: str, patch: str) -> str | None:
    """Try multiple methods to apply patch."""
    # Method 1: system patch --fuzz=3
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(parent)
            tf_path = tf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as pf:
            pf.write(patch)
            pf_path = pf.name

        for fuzz in [3, 99]:
            Path(tf_path).write_text(parent)
            r = subprocess.run(
                ["patch", f"--fuzz={fuzz}", "-s", tf_path, pf_path],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                result = Path(tf_path).read_text()
                try:
                    compile(result, "<test>", "exec")
                    return result
                except SyntaxError:
                    continue
    except Exception:
        pass
    finally:
        Path(tf_path).unlink(missing_ok=True)
        Path(pf_path).unlink(missing_ok=True)

    # Method 2: fuzzy Python apply
    result = fuzzy_apply(parent, patch)
    if result:
        try:
            compile(result, "<test>", "exec")
            return result
        except SyntaxError:
            pass

    # Method 3: if patch is a full replacement (starts at line 1, replaces everything)
    hunks = parse_hunks(patch)
    if hunks and hunks[0]["old_start"] == 1:
        new_lines = []
        for l in hunks[0]["lines"]:
            if l.startswith("+") or l.startswith(" "):
                new_lines.append(l[1:] if len(l) > 1 else "")

        # If patch covers most of the file, try using just + lines
        all_add_lines = []
        for h in hunks:
            for l in h["lines"]:
                if l.startswith("+"):
                    all_add_lines.append(l[1:] if len(l) > 1 else "")
                elif l.startswith(" "):
                    all_add_lines.append(l[1:] if len(l) > 1 else "")

        if all_add_lines:
            result = "\n".join(all_add_lines) + "\n"
            try:
                compile(result, "<test>", "exec")
                if len(result) > 50:
                    return result
            except SyntaxError:
                pass

    return None


def extract_spec(contract: dict) -> str:
    op = contract.get("operator", "unknown")
    arch = contract.get("architecture", "gfx942")
    lang = contract.get("language", "triton")
    shape_info = ""
    kc = contract.get("kernel_contract", {})
    if kc:
        shape = kc.get("shape", {})
        dtype = kc.get("input_dtype", "bf16")
        if shape:
            shape_info = f"\nShape: {json.dumps(shape)}\nDtype: {dtype}"
    return f"Operator: {op}\nArchitecture: {arch}\nLanguage: {lang}{shape_info}"


def main():
    samples = []
    with open(SFT_DATA) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    by_op = defaultdict(list)
    method_stats = {"patch_cmd": 0, "fuzzy_py": 0, "reconstruct": 0, "fail": 0}

    for i, sample in enumerate(samples):
        inp = sample.get("input", {})
        labels = sample.get("labels", {})

        if not labels.get("compile_pass") or not labels.get("correctness_pass"):
            continue

        parent = inp.get("parent_source", {}).get("kernel.py", "")
        patch = sample.get("output", {}).get("patch", "")
        contract = inp.get("contract", {})
        op = contract.get("operator", "unknown")

        if not parent or not patch:
            continue

        final_kernel = try_apply(parent, patch)
        if not final_kernel:
            method_stats["fail"] += 1
            continue

        spec = extract_spec(contract)
        new_sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Write a complete Triton kernel implementation for:\n\n{spec}\n\n"
                        f"The kernel must pass correctness checks and achieve high performance."
                },
                {"role": "assistant", "content": f"```python\n{final_kernel}\n```"},
            ],
            "task_type": "complete_implementation",
            "operator": op,
            "source_sample_id": sample.get("sample_id", ""),
            "data_source": "complete_impl_v2",
            "schema_version": "messages_v1",
        }
        by_op[op].append(new_sample)

        if (i + 1) % 200 == 0:
            total_so_far = sum(len(v) for v in by_op.values())
            print(f"  Processed {i+1}/{len(samples)}, generated {total_so_far}, failed {method_stats['fail']}")

    total = sum(len(v) for v in by_op.values())
    print(f"\nGenerated {total} complete_implementation samples")
    print(f"Failed to apply: {method_stats['fail']}")
    print(f"\nPer operator:")
    for op in sorted(by_op, key=lambda x: -len(by_op[x])):
        print(f"  {op}: {len(by_op[op])}")

    all_impl = []
    for op in sorted(by_op):
        all_impl.extend(by_op[op])

    with open(OUTPUT, "w") as f:
        for item in all_impl:
            f.write(json.dumps(item) + "\n")
    print(f"\nSaved {len(all_impl)} to {OUTPUT}")

    # Now rebuild train.jsonl = 2000 patch + all complete_impl
    MERGED = Path("/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl")

    # Read existing 2000 patch samples (already in messages format from previous rebuild)
    patch_samples = []
    with open(MERGED) as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_type") != "complete_implementation":
                patch_samples.append(d)

    print(f"\nRebuilding train.jsonl: {len(patch_samples)} patch + {len(all_impl)} complete_impl")

    with open(MERGED, "w") as f:
        for item in patch_samples:
            f.write(json.dumps(item) + "\n")
        for item in all_impl:
            f.write(json.dumps(item) + "\n")

    total_train = len(patch_samples) + len(all_impl)
    print(f"Total: {total_train} samples")

    # Upload
    print("\nUploading to HuggingFace...")
    import subprocess as sp
    r = sp.run(
        ["huggingface-cli", "upload",
         "Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP",
         str(MERGED), "train.jsonl",
         "--repo-type", "dataset"],
        capture_output=True, text=True, timeout=300,
    )
    print("Upload OK" if r.returncode == 0 else f"Upload failed: {r.stderr[:200]}")


if __name__ == "__main__":
    main()
