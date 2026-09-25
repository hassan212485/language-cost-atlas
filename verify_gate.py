#!/usr/bin/env python3
"""
verify_gate.py — run the gate exactly as the README documents it, and check
that it reproduces the three known cl100k_base -> o200k_base regressions.

This verifies the documented path, not the math in the abstract: if the README
command drifts from the code, this fails. Run it from a clean checkout:

    python3 verify_gate.py

Exit 0 when the documented command exits 1 and its report names exactly
sat_Olck, tzm_Tfng, taq_Tfng. Exit 1 otherwise.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {"sat_Olck", "tzm_Tfng", "taq_Tfng"}


def main():
    # The documented command (README "Reproduce the gate"): no arguments.
    proc = subprocess.run([sys.executable, "regression_gate.py"], cwd=HERE)
    if proc.returncode != 1:
        print(f"FAIL: documented command exited {proc.returncode}, expected 1 "
              "(a gate that finds a regression must fail)", file=sys.stderr)
        return 1

    report_path = os.path.join(HERE, "gate_report.json")
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    got = set(report["regressions"])
    if got != EXPECTED:
        print(f"FAIL: regressions {sorted(got)} != expected {sorted(EXPECTED)}",
              file=sys.stderr)
        return 1

    print(f"verify: documented command reproduced {sorted(EXPECTED)} (exit 1). green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
