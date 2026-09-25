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
`eng_Latn` and fails the language when the one-sided 95% bootstrap CI of that
mean lies entirely above 0 **and** the relative change clears a materiality
floor (default 0.5%). Exit code is 1 when anything regresses.

Reproduce the gate on the known `cl100k_base -> o200k_base` case:

```bash
python3 regression_gate.py
```

That prints the three reds and exits 1:

```
REGRESS (3): sat_Olck, tzm_Tfng, taq_Tfng
  sat_Olck       12.738   +7.568%  CI[+0.9439,+0.9801]
  taq_Tfng       10.098   +1.032%  CI[+0.0884,+0.1288]
  tzm_Tfng       10.032   +1.071%  CI[+0.0911,+0.1215]
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
lands in `gate_report.json` (one row per language, with CI bounds).

### Verify the documented path

```bash
python3 verify_gate.py
```

Runs the README command exactly as written and checks it reproduces
`sat_Olck`, `tzm_Tfng`, `taq_Tfng`. Green only if the docs and the code still
agree.

## Files

- `token_atlas.py` — the whole pipeline, one file
- `regression_gate.py` — scores a candidate vocabulary, fails on regression
- `verify_gate.py` — runs the documented gate command and checks the result
- `gate_report.json` — the gate's last report, one row per language
- `atlas.csv` — one row per language: token totals, ratios, chars-per-token
- `summary.json` — corpus facts and the aggregate numbers
- `atlas-post.md` — the short public write-up

## Sources

FLORES-200 (NLLB team, Meta AI): https://github.com/facebookresearch/flores
Tokenizers: https://github.com/openai/tiktoken

## License

MIT. The FLORES-200 corpus is licensed by its authors, not by this repo.
