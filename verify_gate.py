#!/usr/bin/env python3
"""
verify_gate.py — run the gate exactly as the README documents it, and check
that it reproduces the three known cl100k_base -> o200k_base regressions.

This verifies the documented path, not the math in the abstract. Three checks:
  1. the documented command (no arguments) exits 1;
  2. its report names exactly sat_Olck, tzm_Tfng, taq_Tfng;
  3. the README's sample output for those three matches the run (ratio,
     relative change, and CI bounds) — prose claims and printed numbers are
     drift too, and a guard that only checks exit codes will not catch them.
It still does not verify the statistical method, only that docs and code agree.
Run it from a clean checkout:

    python3 verify_gate.py

Exit 0 when all three pass. Exit 1 otherwise.
"""

import json
import os
import re
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

    # 3. The README's sample output must match this run, field for field.
    findings = {f["language"]: f for f in report["findings"]}
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    line_re = re.compile(
        r"^\s*(?P<lang>[a-z]{2,3}_[A-Za-z]{4})\s+(?P<ratio>\d+\.\d+)\s+"
        r"(?P<rel>[+-]\d+\.\d+)%\s+CI\[(?P<lo>[+-]\d+\.\d+),(?P<hi>[+-]\d+\.\d+)\]",
        re.M)
    documented = {m.group("lang"): m for m in line_re.finditer(readme)}
    problems = []
    for lang in sorted(EXPECTED):
        m = documented.get(lang)
        if m is None:
            problems.append(f"README has no sample line for {lang}")
            continue
        f = findings[lang]
        want = (f"{f['ratio_baseline']:.3f}", f"{f['relative_change_pct']:+.3f}%",
                f"CI[{f['ci_low']:+.4f},{f['ci_high']:+.4f}]")
        have = (m.group("ratio"), m.group("rel") + "%",
                f"CI[{m.group('lo')},{m.group('hi')}]")
        if have != want:
            problems.append(f"{lang}: README {have} != run {want}")
    if problems:
        print("FAIL: README sample output does not match the run:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"verify: documented command reproduced {sorted(EXPECTED)} (exit 1), "
          "README sample output matches the run. green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
