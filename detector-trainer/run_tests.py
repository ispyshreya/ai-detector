#!/usr/bin/env python3
"""Unified test runner for the detector-trainer units.

Each unit's test file exposes a standalone runner (some predate pytest fixtures),
so we invoke them as subprocesses from the package root. Run:

    python3 run_tests.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = [
    "data/test_manifest.py",
    "models/test_resnet.py",
    "models/test_clip_head.py",
    "eval/test_harness.py",
]


def main() -> int:
    failed = []
    for suite in SUITES:
        print(f"\n=== {suite} ===")
        result = subprocess.run([sys.executable, suite], cwd=ROOT)
        if result.returncode != 0:
            failed.append(suite)
    print("\n" + "=" * 40)
    if failed:
        print("FAILED suites:\n  " + "\n  ".join(failed))
        return 1
    print(f"All {len(SUITES)} suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
