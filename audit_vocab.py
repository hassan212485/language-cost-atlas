#!/usr/bin/env python3
"""
audit_vocab.py — independent cost audit of one candidate vocabulary.

Given a candidate tokenizer (hf:/path/tokenizer.json, file:*.tiktoken, or a
tiktoken name), tokenize the same 1,012 FLORES-200 devtest sentences in every
one of the 204 languages and report:

  tokens (total), chars/token, and the tax ratio vs eng_Latn under
  (a) the candidate vocabulary and (b) cl100k_base as a fixed reference.

Also checks decode(encode(x)) == x per language, because a cheaper vocabulary
that does not round-trip is not cheaper, it is lossy.

The honest cross-vocabulary number is chars/token, not the ratio: a
language-specific vocabulary makes English itself more expensive, which
flatters every other language's ratio.

Usage:
  python3 audit_vocab.py --candidate=hf:/path/tokenizer.json --out=report.json
"""

import argparse
import json
import os
import sys

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "flores200_dataset", "devtest")
PIVOT = "eng_Latn"
FOCUS = ["eng_Latn", "urd_Arab", "snd_Arab", "kas_Arab", "pan_Guru", "hin_Deva",
         "ben_Beng", "pes_Arab", "arb_Arab", "prs_Arab", "azb_Arab", "uig_Arab"]


def load_encoder(spec):
    if spec.startswith("hf:"):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(spec[len("hf:"):])
    if spec.startswith("file:"):
        import tiktoken
        with open(spec[len("file:"):], "rb") as fh:
            return tiktoken.core.Encoding.from_tiktoken_model("cand", fh.read())
    import tiktoken
    return tiktoken.get_encoding(spec)


def encode(enc, text):
    if hasattr(enc, "encode_ordinary"):  # tiktoken
        return enc.encode_ordinary(text)
    return enc.encode(text).ids  # tokenizers.Tokenizer


def decode(enc, ids):
    if hasattr(enc, "decode"):
        return enc.decode(ids)
    return enc.decode(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", default="cl100k_base")
    ap.add_argument("--data-dir", default=DEV)
    ap.add_argument("--out", default="audit_report.json")
    args = ap.parse_args()

    cand = load_encoder(args.candidate)
    ref = load_encoder(args.reference)
    langs = sorted(f[:-len(".devtest")] for f in os.listdir(args.data_dir) if f.endswith(".devtest"))

    rows = {}
    for lang in langs:
        with open(os.path.join(args.data_dir, f"{lang}.devtest"), encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
        chars = sum(len(s) for s in lines)
        c_ids = [encode(cand, s) for s in lines]
        tok_c = sum(len(x) for x in c_ids)
        tok_r = sum(len(encode(ref, s)) for s in lines)
        exact = sum(1 for s, ids in zip(lines, c_ids) if decode(cand, ids) == s)
        rows[lang] = {"tokens_candidate": tok_c, "tokens_reference": tok_r, "chars": chars,
                      "chars_per_token_candidate": round(chars / tok_c, 3) if tok_c else None,
                      "chars_per_token_reference": round(chars / tok_r, 3) if tok_r else None,
                      "roundtrip_exact": exact, "sentences": len(lines)}

    for key in ("tokens_candidate", "tokens_reference"):
        piv = rows[PIVOT][key]
        for lang in langs:
            rows[lang]["ratio_vs_pivot_" + key.split("_")[1]] = round(rows[lang][key] / piv, 4)

    order = sorted(langs, key=lambda l: rows[l]["ratio_vs_pivot_candidate"], reverse=True)
    report = {
        "candidate": args.candidate,
        "reference": args.reference,
        "corpus": "FLORES-200 devtest, 1012 sentences per language, 204 languages",
        "pivot": PIVOT,
        "percent_fully_lossless": round(100 * sum(1 for l in langs if rows[l]["roundtrip_exact"] == rows[l]["sentences"]) / len(langs), 1),
        "mean_ratio_vs_pivot_candidate": round(sum(rows[l]["ratio_vs_pivot_candidate"] for l in langs) / len(langs), 4),
        "mean_ratio_vs_pivot_reference": round(sum(rows[l]["ratio_vs_pivot_reference"] for l in langs) / len(langs), 4),
        "pivot_chars_per_token_candidate": rows[PIVOT]["chars_per_token_candidate"],
        "pivot_chars_per_token_reference": rows[PIVOT]["chars_per_token_reference"],
        "worst_candidate": order[:8],
        "focus": FOCUS,
        "languages": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    print(f"lossless for all 1012 sentences: {report['percent_fully_lossless']}% of 204 languages")
    print(f"mean ratio vs eng_Latn: candidate {report['mean_ratio_vs_pivot_candidate']} vs {args.reference} {report['mean_ratio_vs_pivot_reference']}")
    print(f"eng_Latn chars/token: candidate {report['pivot_chars_per_token_candidate']} vs {args.reference} {report['pivot_chars_per_token_reference']}")
    print(f"{'language':<12} {'tok_cand':>9} {'c/t_cand':>9} {'ratio_cand':>11} {'ratio_ref':>10} {'exact':>7}")
    for lang in FOCUS + [l for l in order[:5] if l not in FOCUS]:
        r = rows[lang]
        print(f"{lang:<12} {r['tokens_candidate']:>9} {str(r['chars_per_token_candidate']):>9} "
              f"{r['ratio_vs_pivot_candidate']:>11} {r['ratio_vs_pivot_reference']:>10} "
              f"{r['roundtrip_exact']:>4}/{r['sentences']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
