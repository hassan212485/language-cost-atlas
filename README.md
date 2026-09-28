# The language cost atlas

What does the same meaning cost, in tokens, across 204 languages?

Same 1,012 sentences (FLORES-200 devtest), one file per language, tokenized with
two OpenAI vocabulary generations. English is the yardstick. Every number here
comes from counting real tokens, not estimating.

## The headline

The average language costs **3.36x** English for the same meaning under
`cl100k_base` (GPT-3.5/4-era), and **2.12x** under `o200k_base` (GPT-4o era).
The newer vocabulary did not fix the world. It shrank the *average* gap and
left the *tail* alone.

- Median language: 2.25x (cl100k) -> 1.75x (o200k)
- Worst case: Shan, `shn_Mymr` — **14.98x** English in cl100k
- Only 1 of 204 languages sits under 1.2x in o200k. It is English.

## Where the newer vocabulary helped, and where it failed

Some scripts got transformed when the vocabulary improved:

| Language | cl100k | o200k |
|---|---|---|
| Armenian `hye_Armn` | 9.88x | 1.79x |
| Georgian `kat_Geor` | 9.75x | 1.79x |
| Burmese `mya_Mymr` | 11.62x | 3.16x |
| Odia `ory_Orya` | 12.40x | 4.99x |
| Tamil `tam_Taml` | 7.64x | 1.98x |

And some scripts got **worse** — three, to be exact:

| Language | cl100k | o200k |
|---|---|---|
| Santali, Ol Chiki `sat_Olck` | 12.74x | 13.70x |
| Tamahaq, Tifinagh `taq_Tfng` | 10.10x | 10.21x |
| Central Atlas Tamazight, Tifinagh `tzm_Tfng` | 10.03x | 10.15x |

A vocabulary generation is not monotone progress. It is a budget: money spent
on scripts someone trained a tokenizer for, taken from scripts someone didn't.
The two Tifinagh cases sit in North Africa, which is where I live, which is how
I noticed them at all.

## A third vocabulary: does a purpose-built one fix the tail?

The atlas only compares two OpenAI vocabularies. The obvious follow-up: put a
vocabulary built *for* Tifinagh on the same 1,012 sentences. `measure_tifinizer.py`
does exactly that with
[Tamazight/Tifinizer-Unigram-32K](https://huggingface.co/Tamazight/Tifinizer-Unigram-32K)
(SentencePiece / Unigram, 32,000 tokens, apache-2.0).

| Language | o200k | Tifinizer 32K | chars/token (o200k → 32K) |
|---|---|---|---|
| Tamahaq `taq_Tfng` | 10.21x | 2.14x | 0.45 → 1.49 |
| Central Atlas Tamazight `tzm_Tfng` | 10.15x | 0.96x | 0.43 → 3.17 |

`tzm_Tfng` costs *fewer* tokens than English under a vocabulary trained for it.
Read the ratio with care: this vocabulary spends its budget on Tifinagh, so
English gets *more* expensive under it (3.39 chars/token vs 4.91 under o200k),
which flatters the language side of the ratio. The pivot-free number is
chars/token, and there the same sentences move 0.43 → 3.17, a 7x swing.
Roundtrip: 991/1012 sentences decode exactly; the other 21 differ only by one
dropped space from SentencePiece normalization, with no glyphs lost.

So the tail is not a property of the language. It is a property of who the
vocabulary was built for.

## The metric, and what it actually measures

`ratio = tokens(meaning in language L) / tokens(same meaning in English)`,
over the same 1,012 sentences. A ratio of 3 means the language spends three
tokens where English spends one, for identical content.

What this does **not** say:

- Not a claim about model quality or fluency. This counts the cost of feeding
  the meaning in. What the model does with it is a different study.
- Not "some languages are inefficient." Fragmentation is a property of the
  tokenizer and the text encoding, not of the language or its speakers.
- Not universal across models. Only OpenAI vocabulary generations are tested
  here. `tiktoken` could not load the older open vocabularies in this
  environment; the conclusion is scoped to what actually ran.
- The `chars_per_token` column is the tell: below 1.0 means the tokenizer is
  splitting characters into byte fragments, which is where the cost comes from.

## Reproduce it

```bash
python3 token_atlas.py
```

The script downloads FLORES-200 (25 MB, third-party, not committed), verifies
204 language files, tokenizes every sentence with both vocabularies, and writes
`atlas.csv` and `summary.json`. Needs `tiktoken` only.

## The regression gate

A vocabulary generation can silently make a language more expensive. The gate
is a public checker that scores any candidate vocabulary against this atlas and
fails if a language's token tax regresses.

A flat threshold is the wrong instrument: two of the three known regressions
are ~1% moves, so a 2% cutoff would sleep through them. A vocabulary swap is a
paired experiment (the same 1,012 sentences, tokenized twice), so the gate runs
a paired test over sentences instead.

Per language, it computes the paired difference of the tax ratio against
`eng_Latn` and fails the language when the one-sided 95% lower confidence
bound of that mean clears 0 **and** the relative change clears a materiality
floor (default 0.5%). One-sided on purpose: a gate that stays quiet on a real
regression is worse than one that flags a marginal case. Exit codes are
distinct so a crash cannot pass for a red: **2** when a language regresses,
**1** on any error, **0** when clean.

Every `devtest` file must be line-aligned with the pivot, or the paired
comparison is meaningless; the gate exits with a named error instead of
crashing or silently truncating.

Reproduce the gate on the known `cl100k_base -> o200k_base` case:

```bash
python3 regression_gate.py
```

That prints the three reds and exits 2:

```
REGRESS (3): sat_Olck, tzm_Tfng, taq_Tfng
  sat_Olck       12.738   +7.568%  CI[+0.9370,+0.9915]
  taq_Tfng       10.098   +1.032%  CI[+0.0857,+0.1239]
  tzm_Tfng       10.032   +1.071%  CI[+0.0890,+0.1262]
```

### Score any vocabulary

`--baseline` and `--candidate` each take one of:

- a tiktoken encoding name, e.g. `o200k_base`
- `file:/path/to/vocab.tiktoken` — raw BPE ranks (`--pat-str` sets the split
  regex; defaults to the cl100k pattern)
- `hf:/path/to/tokenizer.json` — a HuggingFace tokenizers file (needs the
  `tokenizers` package)

```bash
python3 regression_gate.py --baseline cl100k_base --candidate file:/path/to/new.tiktoken
```

Tune with `--iters`, `--alpha`, `--floor`, `--pivot`, `--seed`. The report
lands in `gate_report.json` (one row per language, with the two-sided 95% CI
and the one-sided flag bound).

### Verify the documented path

```bash
python3 verify_gate.py
```

Runs the README command exactly as written and checks it reproduces
`sat_Olck`, `tzm_Tfng`, `taq_Tfng`. It moves the committed
`gate_report.json` aside first, so it can only read a report this run just
wrote (a stale artifact cannot masquerade as evidence), and it fails on any
exit code other than 2, so a gate that crashes reads as a failure, not as a
red. Green only if the docs, the committed report, and the code still agree.

## Files

- `token_atlas.py` — the whole pipeline, one file
- `regression_gate.py` — scores a candidate vocabulary, fails on regression
- `verify_gate.py` — runs the documented gate command and checks the result
- `measure_tifinizer.py` — a third vocabulary on the same 1,012 sentences
- `tifinizer_result.json` — that vocabulary's per-language numbers
- `gate_report.json` — the gate's last report, one row per language
- `atlas.csv` — one row per language: token totals, ratios, chars-per-token
- `summary.json` — corpus facts and the aggregate numbers
- `atlas-post.md` — the short public write-up

## Sources

FLORES-200 (NLLB team, Meta AI): https://github.com/facebookresearch/flores
Tokenizers: https://github.com/openai/tiktoken

## License

MIT. The FLORES-200 corpus is licensed by its authors, not by this repo.

## Auditing one candidate vocabulary

`audit_vocab.py` runs a single candidate tokenizer over the same 1,012 FLORES-200 devtest sentences in all 204 languages and reports, per language: total tokens, chars/token, the tax ratio vs `eng_Latn` under that vocabulary and under `cl100k_base`, and whether `decode(encode(x)) == x` holds for every sentence.

```
python3 audit_vocab.py --candidate=hf:/path/to/tokenizer.json --out=audit_report.json
```

Two things it refuses to hide:

- a vocabulary that does not round-trip is not cheaper, it is lossy, so losslessness is reported per language, not assumed;
- chars/token is the honest cross-vocabulary number. A vocabulary built for one language makes English itself more expensive, which flatters every other language's ratio. The script prints both so the flattery stays visible.

Worked example: **Hindko Tokenizer v1.0.0** (`junaid008/hindko-tokenizer`, 32K SentencePiece Unigram, Urdu script) — 100% of the 204 languages round-trip exactly across all 1,012 sentences; `urd_Arab` costs 0.5026x `eng_Latn`; `eng_Latn` is 1.883 chars/token vs 4.855 under `cl100k_base`; mean ratio vs English 2.16 (vs 3.36 under `cl100k_base`). FLORES has no `pan_Arab`, so the model's Shahmukhi Punjabi claim is untested here, not tested and failed.
