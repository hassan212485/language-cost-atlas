#!/usr/bin/env python3
"""
verify_gate.py — run the gate exactly as the README documents it, and check
that it reproduces the three known cl100k_base -> o200k_base regressions,
reading a report this script forced the run to produce.

This verifies the documented path, not the math in the abstract. Four checks:
  1. the documented command (no arguments) exits 2 (regressions). Any other
     exit code fails: a crash (exit 1) or a quiet pass (exit 0) must never
     read as "the gate found the three reds". A guard that only asks
     `returncode != 1` treats a crash as a pass.
  2. the report is one this run just wrote, not a committed copy: the
     committed gate_report.json is moved aside first, so a stale artifact
     cannot masquerade as evidence.
  3. that fresh report names exactly sat_Olck, tzm_Tfng, taq_Tfng.
  4. the README's sample output for those three matches the run (ratio,
     relative change, and CI bounds) — prose claims and printed numbers are
     drift too, and a guard that only checks exit codes will not catch them.
It also checks the committed gate_report.json still byte-matches this fresh
run, so the checked-in artifact cannot rot silently.

Run it from a clean checkout:

    python3 verify_gate.py

Exit 0 when every check passes. Exit 1 otherwise.
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {"sat_Olck", "tzm_Tfng", "taq_Tfng"}
REPORT = os.path.join(HERE, "gate_report.json")
EXIT_REGRESSION = 2


def main():
    # 2. Make the run produce the report. Move the committed copy aside so a
    # checked-in file cannot be mistaken for this run's evidence.
    committed = None
    if os.path.exists(REPORT):
        with open(REPORT, "rb") as fh:
            committed = fh.read()
        os.remove(REPORT)

    # 1. The documented command (README "Reproduce the gate"): no arguments.
    proc = subprocess.run([sys.executable, "regression_gate.py"], cwd=HERE)

    def restore():
        if committed is not None and not os.path.exists(REPORT):
            with open(REPORT, "wb") as fh:
                fh.write(committed)

    if proc.returncode != EXIT_REGRESSION:
        restore()
        if proc.returncode == 1:
            print("FAIL: documented command exited 1 (error), not 2 "
                  "(regressions). A crashing or misconfigured gate must not "
                  "read as a red.", file=sys.stderr)
        else:
            print(f"FAIL: documented command exited {proc.returncode}, expected 2 "
                  "(a gate that finds a regression must exit with the "
                  "regression code)", file=sys.stderr)
        return 1

    if not os.path.exists(REPORT):
        restore()
        print("FAIL: the documented command did not write gate_report.json; "
              "refusing to read a report this run did not produce.",
              file=sys.stderr)
        return 1

    with open(REPORT, encoding="utf-8") as fh:
        report = json.load(fh)

    problems = []

    # 3. Exactly the three known regressions.
    got = set(report["regressions"])
    if got != EXPECTED:
        problems.append(f"regressions {sorted(got)} != expected {sorted(EXPECTED)}")

    # 4. The README's sample output must match this run, field for field.
    findings = {f["language"]: f for f in report["findings"]}
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    line_re = re.compile(
        r"^\s*(?P<lang>[a-z]{2,3}_[A-Za-z]{4})\s+(?P<ratio>\d+\.\d+)\s+"
        r"(?P<rel>[+-]\d+\.\d+)%\s+CI\[(?P<lo>[+-]\d+\.\d+),(?P<hi>[+-]\d+\.\d+)\]",
        re.M)
    documented = {m.group("lang"): m for m in line_re.finditer(readme)}
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

    # The committed artifact must equal what the code now produces.
    if committed is not None:
        with open(REPORT, "rb") as fh:
            fresh = fh.read()
        if fresh != committed:
            problems.append("committed gate_report.json does not byte-match a "
                            "fresh documented run (the checked-in report is stale)")

    if problems:
        print("FAIL:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"verify: documented command reproduced {sorted(EXPECTED)} (exit "
          f"{EXIT_REGRESSION}), report freshly produced, README sample output "
          "and committed report match the run. green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
