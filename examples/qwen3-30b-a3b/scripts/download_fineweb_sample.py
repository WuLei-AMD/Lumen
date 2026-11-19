#!/usr/bin/env python3
"""Stream a reproducible FineWeb sample into the JSONL pretraining format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=26624)
    parser.add_argument("--subset", default="sample-10BT")
    args = parser.parse_args()

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.subset,
        split="train",
        streaming=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    written = 0
    with temporary.open("w") as output:
        for row in dataset:
            text = row.get("text", "").strip()
            if not text:
                continue
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.documents:
                break
    if written != args.documents:
        raise RuntimeError(f"FineWeb ended after {written} usable documents")
    temporary.replace(args.output)
    print(f"Wrote {written} documents to {args.output}")


if __name__ == "__main__":
    main()
