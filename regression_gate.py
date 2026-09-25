#!/usr/bin/env python3
"""
regression_gate.py — fail a new vocabulary before it ships, if it makes any
language pay more tokens for the same meaning than the vocabulary it replaces.

The atlas (token_atlas.py) gives one aggregate ratio per language. That is the
wrong instrument for a gate: a flat threshold either misses small-but-real
regressions or fires on noise. A vocabulary swap is a paired experiment — the
same 1,012 sentences, tokenized twice — so the gate runs a paired test over
sentences, not a difference of two totals.

Criterion (per language L, pivot P):
    d_i = tokens_L^cand(i) / tokens_P^cand(i)
        - tokens_L^base(i) / tokens_P^base(i)
    A language REGRESSES if the one-sided 95% bootstrap CI of mean(d) lies
    entirely above 0 AND the relative change is at least --floor (default
    0.5%), so statistically-real but immaterial drift is not a failure.

Exit code is 1 when any language regresses, 0 otherwise. That is the gate.

Reproduces the known cl100k_base -> o200k_base case:
    python3 regression_gate.py
    -> REGRESS: sat_Olck, tzm_Tfng, taq_Tfng  (exit 1)
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_atlas import fetch_corpus, read_lines, PIVOT  # noqa: E402

BASELINE = "cl100k_base"
CANDIDATE = "o200k_base"


def per_sentence_counts(corpus, langs, vocab):
    """{lang: [tokens per sentence]} for one vocabulary, sentences aligned."""
    import tiktoken
    enc = tiktoken.get_encoding(vocab)
    devtest = os.path.join(corpus, "devtest")
    out = {}
    for lang in langs:
        lines = read_lines(os.path.join(devtest, f"{lang}.devtest"))
        out[lang] = [len(enc.encode(s)) for s in lines]
    return out


def bootstrap_ci(diffs, iters, rng, alpha):
    """One-sided lower bound and two-sided CI of the mean of paired diffs."""
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return lo, hi


def run_gate(corpus, baseline, candidate, pivot, iters, alpha, floor, seed):
    devtest = os.path.join(corpus, "devtest")
    langs = sorted(f[: -len(".devtest")] for f in os.listdir(devtest)
                   if f.endswith(".devtest"))
    if pivot not in langs:
        raise SystemExit(f"pivot {pivot} missing from corpus")

    base = per_sentence_counts(corpus, langs, baseline)
    cand = per_sentence_counts(corpus, langs, candidate)
    n = len(base[pivot])
    rng = random.Random(seed)

    findings = []
    for lang in langs:
        diffs = [cand[lang][i] / cand[pivot][i] - base[lang][i] / base[pivot][i]
                 for i in range(n)]
        mean = sum(diffs) / n
        old_ratio = sum(base[lang]) / sum(base[pivot])
        rel = 100.0 * mean / old_ratio if old_ratio else 0.0
        lo, hi = bootstrap_ci(diffs, iters, rng, alpha)
        regressed = lo > 0 and rel >= floor
        findings.append({
            "language": lang,
            "ratio_baseline": round(old_ratio, 4),
            "relative_change_pct": round(rel, 3),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "regressed": regressed,
        })

    regressions = sorted((f for f in findings if f["regressed"]),
                         key=lambda f: -f["relative_change_pct"])
    return {
        "baseline_vocab": baseline,
        "candidate_vocab": candidate,
        "pivot": pivot,
        "sentences": n,
        "languages": len(langs),
        "bootstrap_iters": iters,
        "alpha": alpha,
        "floor_pct": floor,
        "regressions": [f["language"] for f in regressions],
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--candidate", default=CANDIDATE)
    ap.add_argument("--pivot", default=PIVOT)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.5,
                    help="minimum relative %% change to count as material")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_report.json"))
    args = ap.parse_args()

    corpus = fetch_corpus(args.data_dir)
    report = run_gate(corpus, args.baseline, args.candidate, args.pivot,
                      args.iters, args.alpha, args.floor, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"gate: {args.baseline} -> {args.candidate} over {report['languages']} languages, "
          f"{report['sentences']} sentences")
    if report["regressions"]:
        print(f"REGRESS ({len(report['regressions'])}): " + ", ".join(report["regressions"]))
        for f in report["findings"]:
            if f["regressed"]:
                print(f"  {f['language']:12} {f['ratio_baseline']:8.3f}  "
                      f"{f['relative_change_pct']:+7.3f}%  CI[{f['ci_low']:+.4f},{f['ci_high']:+.4f}]")
        sys.exit(1)
    print("PASS: no language regressed")


if __name__ == "__main__":
    main()
