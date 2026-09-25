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

## Files

- `token_atlas.py` — the whole pipeline, one file
- `atlas.csv` — one row per language: token totals, ratios, chars-per-token
- `summary.json` — corpus facts and the aggregate numbers
- `atlas-post.md` — the short public write-up

## Sources

FLORES-200 (NLLB team, Meta AI): https://github.com/facebookresearch/flores
Tokenizers: https://github.com/openai/tiktoken

## License

MIT. The FLORES-200 corpus is licensed by its authors, not by this repo.
