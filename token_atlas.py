#!/usr/bin/env python3
"""
token_atlas.py — what does the same meaning cost, in tokens, across languages?

Same 1,012 sentences (FLORES-200 devtest), one file per language. We tokenize
every sentence in every language with two OpenAI vocabulary generations
(cl100k_base, o200k_base) and count tokens. English is the yardstick.

Output: atlas.csv (one row per language) + summary.json.

The corpus is not committed (25 MB, third-party license). The script fetches it
on first run and verifies the file count, so the table is reproducible from a
clean checkout with:  python3 token_atlas.py
"""

import argparse
import csv
import json
import os
import sys
import tarfile
import urllib.request

CORPUS_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
EXPECTED_LANGS = 204
VOCABS = ["cl100k_base", "o200k_base"]
PIVOT = "eng_Latn"


def fetch_corpus(data_dir):
    """Download + extract FLORES-200 if not already present. Returns corpus dir."""
    corpus = os.path.join(data_dir, "flores200_dataset")
    devtest = os.path.join(corpus, "devtest")
    if os.path.isdir(devtest) and len(os.listdir(devtest)) == EXPECTED_LANGS:
        return corpus

    os.makedirs(data_dir, exist_ok=True)
    tarball = os.path.join(data_dir, "flores200_dataset.tar.gz")
    if not os.path.exists(tarball):
        print(f"downloading {CORPUS_URL}", file=sys.stderr)
        urllib.request.urlretrieve(CORPUS_URL, tarball)
    print("extracting", file=sys.stderr)
    with tarfile.open(tarball) as tf:
        tf.extractall(data_dir)

    n = len(os.listdir(devtest))
    if n != EXPECTED_LANGS:
        raise SystemExit(f"expected {EXPECTED_LANGS} language files, found {n}")
    return corpus


def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "atlas.csv"))
    args = ap.parse_args()

    corpus = fetch_corpus(args.data_dir)
    devtest = os.path.join(corpus, "devtest")

    import tiktoken

    encoders = {v: tiktoken.get_encoding(v) for v in VOCABS}
    langs = sorted(f[:-len(".devtest")] for f in os.listdir(devtest) if f.endswith(".devtest"))
    if PIVOT not in langs:
        raise SystemExit(f"pivot language {PIVOT} missing from corpus")

    # token counts[lang][vocab] = list of per-sentence token counts
    counts = {}
    chars = {}
    for lang in langs:
        lines = read_lines(os.path.join(devtest, f"{lang}.devtest"))
        chars[lang] = sum(len(s) for s in lines)
        counts[lang] = {v: [len(encoders[v].encode(s)) for s in lines] for v in VOCABS}

    pivot_totals = {v: sum(counts[PIVOT][v]) for v in VOCABS}
    pivot_chars = chars[PIVOT]

    rows = []
    for lang in langs:
        row = {"language": lang, "sentences": len(counts[lang][VOCABS[0]])}
        row["chars"] = chars[lang]
        for v in VOCABS:
            total = sum(counts[lang][v])
            row[f"tokens_{v}"] = total
            row[f"ratio_{v}"] = round(total / pivot_totals[v], 4)
            row[f"chars_per_token_{v}"] = round(chars[lang] / total, 3) if total else None
        rows.append(row)

    # sort by the older vocabulary's ratio: worst fragmentation first
    rows.sort(key=lambda r: r[f"ratio_{VOCABS[0]}"], reverse=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "corpus": "FLORES-200 devtest",
        "sentences_per_language": len(counts[PIVOT][VOCABS[0]]),
        "languages": len(langs),
        "vocabs": VOCABS,
        "pivot": PIVOT,
        "pivot_tokens": pivot_totals,
        "pivot_chars": pivot_chars,
        "mean_ratio": {v: round(sum(r[f"ratio_{v}"] for r in rows) / len(rows), 4) for v in VOCABS},
        "max_ratio": {v: max(rows, key=lambda r: r[f"ratio_{v}"])["language"] for v in VOCABS},
        "min_ratio": {v: min(rows, key=lambda r: r[f"ratio_{v}"])["language"] for v in VOCABS},
    }
    with open(os.path.join(os.path.dirname(args.out), "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"wrote {args.out} ({len(rows)} languages)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
