# Same meaning, 204 languages, tokens counted

I counted. FLORES-200 devtest, 1,012 sentences, one file per language, run
through two OpenAI vocabularies.

English is the yardstick.

The average language costs **3.36x** English for the same meaning under the
old vocabulary (`cl100k_base`), and **2.12x** under the new one (`o200k_base`).

The newer vocabulary did not fix the world. It shrank the average and left the
tail.

Worst case: Shan, 14.98x. Armenian and Georgian dropped from ~9.8x to ~1.79x
when the vocabulary improved. Three scripts got *worse* instead: Santali in Ol
Chiki, and two Tifinagh scripts from North Africa. A vocabulary is a budget:
money spent on scripts someone trained a tokenizer for, taken from scripts
someone didn't.

One in 204 languages sits under 1.2x. It is English.

Code, table, and the limits of the measurement:
https://github.com/hassan212485/language-cost-atlas
