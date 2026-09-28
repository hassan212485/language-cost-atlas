#!/usr/bin/env python3
"""
measure_tifinizer.py — put a THIRD vocabulary on the same 1,012 meanings.

token_atlas.py compares two OpenAI vocabulary generations (cl100k_base,
o200k_base). This script adds one purpose-built vocabulary trained for the
Tifinagh script (Tamazight/Tifinizer-Unigram-32K, SentencePiece / Unigram)
and measures it on the SAME FLORES-200 devtest sentences, with the SAME pivot
(eng_Latn) and the SAME counting rule as token_atlas.py.

Why Tifinagh: in atlas.csv the newer OpenAI vocabulary made the tail WORSE.
taq_Tfng 10.098x -> 10.2116x, tzm_Tfng 10.0323x -> 10.1475x. Both are
Tifinagh (North African). So: does a vocabulary trained for the script fix it?

Reproducible from a clean checkout:
    python3 measure_tifinizer.py
It downloads the SentencePiece model from Hugging Face on first run and
verifies the vocabulary size, then writes tifinizer_result.json.

Note on the ratio: tokens(lang)/tokens(eng_Latn), both counted with the SAME
vocabulary. A Tamazight-focused 32K vocabulary spends its budget on Tifinagh,
so English gets MORE expensive under it, which flatters the language side of
the ratio. The chars/token column is the pivot-free number; read them together.
"""
import json
import os
import sys
import unicodedata
import urllib.request

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "data", "flores200_dataset", "devtest")
MODEL_URL = ("https://huggingface.co/Tamazight/Tifinizer-Unigram-32K/"
             "resolve/main/tokenizer.model")
MODEL = os.path.join(HERE, "data", "tifinizer_unigram_32k.model")
EXPECTED_VOCAB = 32000
LANGS = ["eng_Latn", "taq_Tfng", "tzm_Tfng"]
PIVOT = "eng_Latn"


def fetch_model():
    if os.path.exists(MODEL):
        return MODEL
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print(f"downloading {MODEL_URL}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, MODEL)
    return MODEL


def read(lang):
    with open(os.path.join(DEV, f"{lang}.devtest"), encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def lost_chars(orig, back):
    """Characters present in orig but absent from back (pure length diff)."""
    from collections import Counter
    return Counter(orig) - Counter(back)


def main():
    if not os.path.isdir(DEV):
        raise SystemExit(f"corpus not found at {DEV}; run token_atlas.py first")
    path = fetch_model()
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    if sp.get_piece_size() != EXPECTED_VOCAB:
        raise SystemExit(f"expected vocab {EXPECTED_VOCAB}, got {sp.get_piece_size()}")

    out = {"tokenizer": "Tamazight/Tifinizer-Unigram-32K",
           "algorithm": "SentencePiece/Unigram", "vocab": sp.get_piece_size(),
           "corpus": "FLORES-200 devtest, 1012 sentences", "pivot": PIVOT,
           "languages": {}}
    for lang in LANGS:
        sents = read(lang)
        toks = sum(len(sp.encode(s, out_type=int)) for s in sents)
        chs = sum(len(s) for s in sents)
        exact = sum(1 for s in sents if sp.decode(sp.encode(s, out_type=int)) == s)
        lost = set()
        for s in sents:
            for ch in lost_chars(s, sp.decode(sp.encode(s, out_type=int))):
                lost.add(unicodedata.name(ch, f"U+{ord(ch):04X}"))
        out["languages"][lang] = {
            "sentences": len(sents), "chars": chs, "tokens": toks,
            "chars_per_token": round(chs / toks, 3),
            "roundtrip_exact": exact, "roundtrip_chars_lost": sorted(lost),
        }

    eng = out["languages"][PIVOT]["tokens"]
    for lang in LANGS:
        c = out["languages"][lang]
        c["ratio_vs_eng"] = round(c["tokens"] / eng, 4)

    with open(os.path.join(HERE, "tifinizer_result.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["languages"], indent=2))


if __name__ == "__main__":
    main()
