"""Tokenize messages-format dataset into input_ids + loss_mask for SFT training.

Reads train.jsonl (messages format) and produces train.tokenized.jsonl
compatible with PretokenizedLossMaskDataset.
"""
import json
import hashlib
from pathlib import Path
from transformers import AutoTokenizer

TOKENIZER_PATH = "/home/danyzhan/Lumen/experiments/GEAK-agent-coder/models/Qwen3-Coder-30B-A3B-Instruct"
TRAIN_INPUT = Path("/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl")
DEV_INPUT = Path("/home/danyzhan/geak_sft_dataset/phase1-dev-wave-200-v1/processed/dev.jsonl")
OUTPUT_DIR = Path("/home/danyzhan/Lumen/experiments/GEAK-agent-coder/data/build/qwen3_30b_a3b_sft_f")
MAX_LENGTH = 12288

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_messages_sample(tokenizer, sample, max_length):
    """Tokenize a messages-format sample with loss mask on assistant content."""
    messages = sample.get("messages")
    if not messages:
        return None

    # Full sequence
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    # Prompt only (without assistant)
    prompt_messages = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
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
                skipped += 1
                continue

            fout.write(json.dumps(result) + "\n")
            tokenized += 1

    return tokenized, skipped


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Train
    train_out = OUTPUT_DIR / "train.tokenized.jsonl"
    print(f"\nTokenizing train: {TRAIN_INPUT}")
    train_tok, train_skip = process_file(tokenizer, TRAIN_INPUT, train_out, MAX_LENGTH)
    print(f"  Tokenized: {train_tok}, Skipped (overflow): {train_skip}")

    # Dev
    dev_out = OUTPUT_DIR / "dev.tokenized.jsonl"
    print(f"\nTokenizing dev: {DEV_INPUT}")
    dev_tok, dev_skip = process_file(tokenizer, DEV_INPUT, dev_out, MAX_LENGTH)
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
            "train": str(TRAIN_INPUT),
            "dev": str(DEV_INPUT),
        },
        "max_length": MAX_LENGTH,
        "tokenizer": TOKENIZER_PATH,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Manifest SHA256: {file_sha256(manifest_path)}")


if __name__ == "__main__":
    main()
