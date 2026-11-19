"""Tokenize every messages-format row without truncation and pin the result."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_ROOT = Path("/home/danyzhan/Lumen/experiments/GEAK-agent-coder")


def apply_chat_template_ids(tokenizer, messages, *, add_generation_prompt):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("chat template unexpectedly returned a batched result")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError("chat template did not return a flat token-id list")
    return encoded


def tokenize_messages_sample(tokenizer, sample, max_length):
    """Tokenize a messages-format sample with loss mask on assistant content."""
    messages = sample.get("messages")
    if not messages:
        return None

    # Full sequence
    full_ids = apply_chat_template_ids(
        tokenizer, messages, add_generation_prompt=False
    )

    # Prompt only (without assistant)
    prompt_messages = messages[:-1]
    prompt_ids = apply_chat_template_ids(
        tokenizer, prompt_messages, add_generation_prompt=True
    )

    if len(full_ids) > max_length:
        return None

    # Loss mask: 0 for prompt, 1 for assistant content + EOT
    prefix_len = len(prompt_ids)
    loss_mask = [0] * prefix_len + [1] * (len(full_ids) - prefix_len)

    assert len(loss_mask) == len(full_ids)

    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
        "attention_mask": [1] * len(full_ids),
    }


def tokenize_old_format_sample(tokenizer, sample, max_length):
    """Tokenize a geak_kernel_sft_v1 format sample."""
    inp = sample.get("input", {})
    output = sample.get("output", {})
    patch = output.get("patch", "")
    task_type = sample.get("task_type", "unknown")

    system_prompt = (
        "You are an expert GPU kernel engineer optimizing Triton kernels "
        "for AMD MI300X (gfx942)."
    )

    user_payload = json.dumps({"input": inp, "task_type": task_type}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
        {"role": "assistant", "content": patch},
    ]

    return tokenize_messages_sample(tokenizer, {"messages": messages}, max_length)


def process_file(tokenizer, input_path, output_path, max_length):
    tokenized = 0
    skipped = 0
    max_observed_length = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            if "messages" in sample:
                result = tokenize_messages_sample(tokenizer, sample, max_length)
            elif "input" in sample and "output" in sample:
                result = tokenize_old_format_sample(tokenizer, sample, max_length)
            else:
                skipped += 1
                continue

            if result is None:
                raise ValueError(
                    f"{input_path}: row {tokenized + skipped + 1} exceeds "
                    f"max_length={max_length}; full-coverage tokenization forbids truncation"
                )

            fout.write(json.dumps(result) + "\n")
            tokenized += 1
            max_observed_length = max(max_observed_length, len(result["input_ids"]))

    return tokenized, skipped, max_observed_length


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def path_sha256(path):
    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"tokenizer directory is empty: {target}")
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def scan_max_length(tokenizer, paths):
    maximum = 0
    rows = 0
    for input_path in paths:
        with open(input_path) as fin:
            for line_number, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "messages" in sample:
                    messages = sample["messages"]
                elif "input" in sample and "output" in sample:
                    converted = tokenize_old_format_sample(tokenizer, sample, 2**31 - 1)
                    maximum = max(maximum, len(converted["input_ids"]))
                    rows += 1
                    continue
                else:
                    raise ValueError(f"{input_path}:{line_number}: unsupported sample schema")
                ids = apply_chat_template_ids(
                    tokenizer, messages, add_generation_prompt=False
                )
                maximum = max(maximum, len(ids))
                rows += 1
    if rows == 0:
        raise ValueError("dataset contains no samples")
    return maximum


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_ROOT / "models/Qwen3-Coder-30B-A3B-Instruct",
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "data/build/qwen3_30b_a3b_sft_f_mi350",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to the observed maximum, rounded up to a multiple of 8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    observed_max = scan_max_length(tokenizer, (args.train, args.dev))
    max_length = args.max_length or int(math.ceil(observed_max / 8) * 8)
    if max_length < observed_max:
        raise ValueError(
            f"max_length={max_length} does not cover observed maximum={observed_max}"
        )
    print(f"Observed maximum: {observed_max}; configured maximum: {max_length}")

    # Train
    train_out = args.output_dir / "train.tokenized.jsonl"
    print(f"\nTokenizing train: {args.train}")
    train_tok, train_skip, train_max = process_file(
        tokenizer, args.train, train_out, max_length
    )
    print(f"  Tokenized: {train_tok}, Skipped (overflow): {train_skip}")

    # Dev
    dev_out = args.output_dir / "dev.tokenized.jsonl"
    print(f"\nTokenizing dev: {args.dev}")
    dev_tok, dev_skip, dev_max = process_file(
        tokenizer, args.dev, dev_out, max_length
    )
    print(f"  Tokenized: {dev_tok}, Skipped (overflow): {dev_skip}")

    # Token stats
    print("\nToken statistics:")
    for name, path in [("train", train_out), ("dev", dev_out)]:
        total_tokens = 0
        total_loss_tokens = 0
        count = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                ids = d["input_ids"]
                lm = d["loss_mask"]
                total_tokens += len(ids)
                total_loss_tokens += sum(lm)
                count += 1
        print(f"  {name}: {count} samples, {total_tokens} total tokens, "
              f"{total_loss_tokens} loss tokens, "
              f"avg {total_tokens/max(count,1):.0f} tokens/sample")

    # Manifest
    manifest = {
        "schema_version": "geak_training_data_manifest_v2",
        "artifacts": {
            "train": {
                "path": "train.tokenized.jsonl",
                "rows": train_tok,
                "sha256": file_sha256(train_out),
            },
            "dev": {
                "path": "dev.tokenized.jsonl",
                "rows": dev_tok,
                "sha256": file_sha256(dev_out),
            },
        },
        "source": {
            "train": {
                "path": str(args.train.resolve()),
                "sha256": file_sha256(args.train),
            },
            "dev": {
                "path": str(args.dev.resolve()),
                "sha256": file_sha256(args.dev),
            },
        },
        "max_length": max_length,
        "observed_max_length": max(train_max, dev_max),
        "split_max_lengths": {"train": train_max, "dev": dev_max},
        "overflow_rows": train_skip + dev_skip,
        "tokenizer": {
            "path": str(args.tokenizer.resolve()),
            "sha256": path_sha256(args.tokenizer),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Manifest SHA256: {file_sha256(manifest_path)}")


if __name__ == "__main__":
    main()
