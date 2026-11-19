#!/usr/bin/env python3
"""Export a merged SFT checkpoint as a pinned HF blockwise-FP8 candidate."""

import argparse
import json

from geak_agent_coder.export.exporter import export_fp8_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-arch", default="gfx950")
    parser.add_argument("--fp8-format", default="e4m3fn")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    manifest = export_fp8_checkpoint(
        args.source_dir,
        args.output_dir,
        fp8_format=args.fp8_format,
        target_arch=args.target_arch,
        block_size=args.block_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
