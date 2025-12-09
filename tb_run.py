"""
tb_run.py

Minimal CLI entry point to create a new TB "run" directory on D:\\TB_DATA\\runs.

Responsibilities:
- Generate or accept a run_id.
- Create D:\\TB_DATA\\runs\\<run_id>.
- Write a small run_meta.json with basic metadata.
- Initialize logging for this run (tb.log in the run directory).
- Print key information to stdout.

No trading logic here – this is just the run/artefact scaffold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tb_paths import (
    REPO_ROOT,
    DATA_ROOT,
    RUNS_ROOT,
    get_new_run_id,
    create_run_dir,
)
from tb_logging import get_tb_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TB: create a new run directory on D:\\TB_DATA\\runs"
    )
    parser.add_argument(
        "-d",
        "--description",
        type=str,
        default="",
        help="Optional short description for this run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional explicit run_id. "
            "If omitted, a timestamp-based run_id will be generated."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine run_id
    run_id = args.run_id or get_new_run_id()

    # Very simple validation: we don't allow path separators inside run_id.
    if any(sep in run_id for sep in ("/", "\\")):
        print("[TB][ERROR] run_id must not contain '/' or '\\'.", file=sys.stderr)
        return 1

    # Ensure run directory exists
    try:
        run_dir = create_run_dir(run_id)
    except RuntimeError as exc:
        print(f"[TB][ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # defensive catch-all
        print(
            f"[TB][ERROR] Unexpected error while creating run directory: {exc}",
            file=sys.stderr,
        )
        return 1

    # Build metadata payload
    now = datetime.now()
    meta = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "description": args.description,
        "paths": {
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "runs_root": str(RUNS_ROOT),
            "run_dir": str(run_dir),
        },
        "tb": {
            "component": "run_scaffold",
            "version": "0.2.0",  # bumped version due to logging integration
        },
    }

    # Write metadata file into the run directory
    meta_path = run_dir / "run_meta.json"
    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as exc:
        print(
            f"[TB][ERROR] Failed to write run_meta.json in {run_dir}: {exc}",
            file=sys.stderr,
        )
        return 1

    # --- Initialize logging for this run ---
    log_file = run_dir / "tb.log"
    try:
        logger = get_tb_logger("tb.run", log_file=log_file)
    except Exception as exc:
        # Logging should not prevent the run directory from existing, but we want to see the error.
        print(
            f"[TB][ERROR] Failed to initialize logger for run '{run_id}': {exc}",
            file=sys.stderr,
        )
        return 1

    # Log basic run information
    logger.info("New run created: run_id=%s", run_id)
    if args.description:
        logger.info("Description: %s", args.description)
    else:
        logger.info("Description: <empty>")

    logger.info(
        "Paths: repo_root=%s, data_root=%s, runs_root=%s, run_dir=%s",
        REPO_ROOT,
        DATA_ROOT,
        RUNS_ROOT,
        run_dir,
    )
    logger.info("Meta file written at: %s", meta_path)

    # Optional: warn if we are not executed from the repo root.
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        warning_msg = (
            "Current working directory is not equal to REPO_ROOT. "
            f"cwd={cwd} REPO_ROOT={REPO_ROOT}"
        )
        logger.warning(warning_msg)
        print("[TB][WARN] " + warning_msg)

    # Final status output to stdout (short summary)
    print(f"[TB] Created run '{run_id}'")
    print(f"[TB] Run directory: {run_dir}")
    print(f"[TB] Meta file    : {meta_path}")
    print(f"[TB] Log file     : {log_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
