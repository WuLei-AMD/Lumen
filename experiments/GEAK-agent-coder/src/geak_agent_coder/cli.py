"""Command-line entry points for GEAK Agent Coder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import build_from_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geak-agent-coder")
    commands = parser.add_subparsers(dest="command", required=True)
    data_build = commands.add_parser(
        "data-build", help="build pinned local tokenized train/dev data"
    )
    data_build.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "data-build":
        manifest = build_from_config(args.config)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
