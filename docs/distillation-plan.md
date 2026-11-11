# Distilled corpus: plan

10 November 2026. Working plan, not a result. Not legal advice. Numbers marked interim will change.

The human-translated corpus (section 4 of [SPEC.md](../SPEC.md)) is about 299,000 pairs of assistant
commands, reviews and task dialogue. A model trained on it translates that kind of text well and
almost nothing else. This plan adds a second, larger corpus: English and Thai sentences from openly
licensed sources, translated once by a hosted teacher model, published as a versioned release like
the first.

## 1. Why

Scores on 10 November, chrF, greedy decoding, one scoring script (`train/evaluate.py`):

| System | Validation rows | Unfamiliar text | FLORES-200 en to th | FLORES-200 th to en |
|---|---:|---:|---:|---:|
| DeepSeek Flash (the teacher) | 46.72 | 61.50 | 59.08 | 63.81 |
| Typhoon Translate 1.5, 4B | | | 54.22 | 52.46 |
| This model, 75M, 8 epochs | 44.68 | 36.16 | 26.15 | 30.50 |
| This model, 75M, low-resource recipe | 43.22 | 35.11 | 25.88 | 28.52 |

- Validation rows: 2,000 held-out rows from the same sources as the training data, both directions.
- Unfamiliar text: 900 pairs from ALT (news), TALPCo (textbook) and Tatoeba (everyday), never trained on.
- FLORES-200: the 1,012 devtest sentences. Excluded from every training file.
- Typhoon Translate was run locally from its published weights with the rule-free prompt on its model
  card. Published scores for other systems are in section 8 and are not comparable without re-scoring.

The model is within two points of the teacher on text like its training data and 25 to 33 points
behind elsewhere. The gap is coverage. A bundle of low-resource training techniques (dropout 0.3,
label smoothing 0.2, subword sampling, word dropout) did not close it: it scored lower on all three.

**Target:** as close to 60 on FLORES-200 as possible in both directions. 60 is about where the teacher
scores on English to Thai, and a 150-sentence check against a larger commercial model scored 61.4, so
it looks like the practical ceiling for this measure. Milestones: 43 (a published Transformer-base
trained from scratch, section 8), 50, and 54 (Typhoon Translate 1.5).

Thai to English scores on FLORES need care. Its English side is original public text, and a large
model given the Thai can reproduce the English it has already read: the same check scored 95 in that
direction. English to Thai does not show this.

## 2. What the corpus is

Single-language sentences, each translated once by the teacher. One row is one chat utterance or one
sentence of prose, with its source, register, licence and, for dialogue, the line before it.

Every pair is used in both directions:

- **Forward** (real source, teacher's target): trains the direction the row was translated in.
- **Reversed** (teacher's text as source, real text as target), tagged so the model can tell.

A real English sentence therefore mostly helps English to Thai, and a real Thai sentence mostly helps
Thai to English (section 8, Koehn). The share of Thai-origin text matters for that reason.

## 3. Sources

**Rule: sources are in two tiers, and every row carries its tier.**

- **Tier 1** has a licence that allows commercial use, redistribution and derived works, granted by
  whoever holds the rights to the text.
- **Tier 2** is public Thai text with no stated licence, or with a licence that covers only the
  packaging. It was added on 11 November 2026 because tier 1 Thai ran out near 500,000 rows. It is
  kept apart so that the set can be built without it in one step, and whether it stays is decided
  before any release.

A source with an explicit term against this use stays out of both tiers. Each licence is read on the
source's own page and recorded with an attribution line in the manifest.

The pool today is 9,065,701 rows, and the set reduced from it for distillation is 8,731,390 rows:
3,700,902 English (all tier 1) and 5,030,488 Thai (501,476 tier 1 and 4,529,012 tier 2). Subtitle
dialogue from OpenSubtitles is 4,418,737 of the tier 2 rows and scores about 77 clean rows in 100,
under the bar of 85; it is in because it is the only large body of conversational Thai. The
[source record](distillation-sources.md) lists every source with its licence, tier, row count and
sample score.

Left out, and why:

| Kind | Examples | Reason |
|---|---|---|
| Web crawls | HPLT, FineWeb2, OSCAR, CC100, mC4, CulturaX, Mangosteen web | Quality: 47 to 60 clean rows in 100 after filtering. OSCAR and CulturaX also need an access agreement |
| Scraped news and reports | ThaiSum, HSE Thai Corpus, CMDF_VISTEC | Publishers' terms restrict reuse; ThaiSum also scored 75 |
| Mined parallel web text | ParaCrawl bonus (Koehn 2024), OPUS collections | Web-crawled; and distilled data did better for Thai (section 8) |
| Non-commercial | BEST, ORCHID, LOTUS, Thai National Corpus, most of WangchanThaiInstruct, Prachathai | Licence |
| Other people's posts | Pantip sets, tweet sets | Posted by people who did not grant a licence. Wisesight and Wongnai reviews are in as tier 2 |
| Outputs of other models | Hunyuan-MT, Claude, and any teacher whose terms forbid training on its output | Terms. The teacher is DeepSeek, whose terms allow it |

## 4. What the audit found

On 10 November, 100 random rows from each of 29 sources were read and labelled by four independent
reviewers: translate as is, usable but flawed, or should not be translated. About 62% of rows were
clean, and the reviewers would have paid to translate about 45% of the 5.6 million rows then in the
build. A share from 100 rows is good to about 9 points.

| Source | Clean | Main defect |
|---|---:|---|
| Thai Wikipedia | 53% | 30% of rows start mid-sentence: the splitter cut before a connective or a subjectless verb |
| Simple English Wikipedia | 62% | "It" and "This" with no referent; thin stub sentences |
| SODA | 87% | Clean, but the same few emotional scripts; 12 to 15% literal frames |
| Tatoeba English | 93% | Low information: 45% are six words or fewer |
| Task dialogue (five sets) | 40 to 53% | Slot-filled templates; about 40% need the previous turn |
| BYU Chit-Chat | 52% | 45% need the previous turn; 4% are injected nonsense sentences |
| English Wikinews | 63% | 11% have words missing where a link was stripped |
| Video captions (YODAS) | 38% | Caption line breaks cut sentences |
| Speech transcripts (LOTUSDIS) | 29% | Segments end before the predicate |
| Thai Wikibooks | 37% | Hard line wraps split words and sentences |

Most of the damage comes from where text is cut into rows, not from the text.

## 5. Cleaning rules

1. **Cut rows correctly, before any filter.**
   - Thai prose: never split before a clause that opens with a connective or a subjectless verb
     (ทำให้, ซึ่ง, จน, เพราะ, เนื่องจาก, แล้ว, โดย, ตลอดจน, ทั้งยัง, ขณะที่, เว้นแต่, ก่อนจะ and
     similar); attach it to the sentence before.
   - Wrapped text: rejoin lines and remove spaces inside words before splitting.
   - Captions and speech: join the pieces of one video or one speaker turn, then split into utterances.
   - Wikis: keep the text of links during extraction.
2. **Drop what is not a sentence.** Thai rows that end on a dangling word (แบบ, ว่า, คือ, เช่น, และ,
   ที่จะ, ดังนี้ and similar) or open an unfinished condition; instructions to a chatbot and chatbot
   answers; quiz option markers and "finish this sentence:" scaffolds; pointers ("See", "this page");
   English prose opening "It", "This", "These" with nothing to refer to ("He", "She", "They" stay,
   since Thai drops the pronoun); codes, usernames and address-like strings.
3. **Thin repetition.** Mask names, numbers and slot values, then cap rows per frame; cap SODA and
   Tatoeba English near 100,000 each; cap stub frames in the Wikipedias.
4. **Remove overlap** with the human corpus, the test suites, FLORES-200 and WMT24++, by exact match
   and by any shared run of eight words; then exact and near duplicates across all sources.
5. **Content.** Adult text is dropped. Rows that mention the Thai royal family are dropped outside
   encyclopaedia and news prose.
6. **Acceptance.** On a fresh sample of 100 rows per source, at least 85 complete and natural and at
   most 3 that should not be translated. Spoken sources are judged as utterances, not as textbook
   sentences. The check is repeated by a reviewer who did not write the rules.

Rules worth copying from other Thai corpus builders: keep a document only if at least half its
characters are Thai, because standard language detectors fail on Thai (Mangosteen); filter gambling
and obscene text (Mangosteen); collapse characters repeated more than three times and keep spaces as
tokens, since spaces mark Thai sentence boundaries (WangchanBERTa).

## 6. Enrichment

- **Context.** A dialogue row carries the previous turn of the same conversation. The teacher is
  shown it and asked to translate only the current line, and the model is trained with its `<ctx>`
  channel. Rows that need a previous turn and have none are dropped.
- **Teacher flags.** The teacher returns a fixed marker in place of a translation when the source is
  not a complete utterance. This catches cut-off rows that no rule can see.
- **Provenance.** Each row keeps its article, video or dialogue id, its position, and which repairs
  were applied.

## 7. Buying and filtering

- **Teacher:** DeepSeek Flash, thinking off, one output per row. It scored within 1.5 points of a much
  larger model on English to Thai, so a stronger teacher would buy little on this kind of text.
- **Price:** not yet measured on this set. A pilot on shorter rows measured 50 input and 20 output
  tokens a row; scaled by length, the current build is roughly $95 to $100 off-peak.
- **Order:** 1,000 rows to measure price and look at quality; 250,000 rows and a training run on the
  small probe model; then further steps only while the score still climbs. Distilled Thai-to-English
  quality in one study rose 22.8, 23.5, 24.1, 24.2 as data grew, then levelled off (section 8).
- **Filter before training:** drop flagged rows, length-ratio outliers, wrong-language outputs and
  the lowest-scoring pairs by an estimated-quality measure. Hunyuan-MT filters its synthetic pairs the
  same way.
- **Record:** model name as served, prompt, settings and date go in the release, with the translated
  rows. The teacher is not deterministic, so a rebuild reproduces the source sentences, not the
  translations.

## 8. What other work shows

Read from the papers and model cards themselves. Scores are theirs and are not on our scoring script.

| Work | Finding | Use here |
|---|---|---|
| Koehn, WMT 2024, [Neural Methods for Aligning Large-Scale Parallel Corpora](https://aclanthology.org/2024.wmt-1.132) | Small Marian models, Thai to English, FLORES-200 spBLEU: mined web pairs 19.3; with OPUS 20.4; pairs distilled from NLLB 25.3; distilled plus OPUS 23.5. Real Thai translated by the teacher 23.9; real English translated and reversed 18.4; both 25.3. Student 25.3 against teacher 31.5 | Distil instead of mining; use both directions; Thai-origin rows matter; expect to land below the teacher |
| Shurtz and others, MRL 2025, [When Scripts Diverge](https://aclanthology.org/2025.mrl-main.22) | Transformer-base from scratch on 1,080,329 cleaned pairs (2,175,880 before cleaning), FLORES+ devtest chrF++: Thai to English 39.4, English to Thai 42.6; best variant 43.2 and 43.9 | The same-class baseline to beat. Half of mined pairs do not survive cleaning |
| Xu and others, [ALMA](https://arxiv.org/abs/2309.11674) | Fine-tuning on single-language text before 58,000 human pairs: 22.17 BLEU without the first stage, 29.89 with it | A pre-training stage on the source sentences before translation training |
| Tencent, [Hunyuan-MT](https://arxiv.org/abs/2509.05209) | About 3M pairs including DeepSeek-synthesised ones, filtered by estimated quality and an LLM judge; then 268,000 curated pairs; then reinforcement learning with metric rewards | Broad then curated; filter synthetic pairs. Its licence forbids training other models on its output |
| SCB 10X, [Typhoon 2](https://arxiv.org/abs/2412.13702) | Dropped Thai translated from English instruction data because it introduced errors; Thai to English 3:7 worked best in its fine-tuning | Prefer Thai written in Thai |
| SCB 10X, [Typhoon Translate](https://huggingface.co/scb10x/typhoon-translate-4b) | A 4B Thai-English translator trained on translations from larger models with human spot checks. Reports no FLORES, BLEU or chrF score | The laptop-sized system to compare with; scored here in section 1 |
| SCB 10X, [Typhoon](https://arxiv.org/abs/2312.13951) | Typhoon-7B-Instruct, FLORES-200 English to Thai: BLEU 31.14, chrF 46.62 | Published reference point |
| Jasmine and Upstage, [JAI-1](https://arxiv.org/abs/2510.08620) | FLORES English to Thai BLEU: GPT-4o 46.5, GPT-4 41.3, a 10.7B translator fine-tuned on 0.5M pairs 41.6. Public Thai text by size: web crawls 58 billion tokens, clean sources about 0.3 billion | What a pretrained base buys; how little clean Thai exists |
| VISTEC, [WangchanBERTa](https://arxiv.org/abs/2101.09635) | 94% of its 78.5 GB was social and forum text supplied privately and not released. Its sentence splitter scored F1 0.69 to 0.96 by domain | Informal Thai at scale is not public; Thai sentence splitting is imperfect everywhere |
| VISTEC and others, [Mangosteen](https://arxiv.org/abs/2507.14664) | A Thai pipeline cut Common Crawl from 202M to 25M documents and raised a small model's score | Thai-specific filters, section 5 |
| AIResearch, [SCB-MT-EN-TH-2020](https://airesearch.in.th/releases/machine-translation-models/) | A 74M Transformer on 1,001,752 pairs: IWSLT 2015 BLEU Thai to English 17.15, English to Thai 12.95; with 3.3M OPUS pairs 28.33 and 17.77 | An earlier model of the same size, to score on the same tests |
| VISTEC, [WangchanThaiInstruct](https://arxiv.org/abs/2508.15239) | BLEU and ROUGE agreed with human judges at Kendall 0.03 and 0.005; an LLM judge at 0.13 | Overlap scores are a rough guide. Claims about chat need judged or human ratings |

## 9. Still to add

Clean supply of Thai tops out well below the English. Leads, none verified, each needing its licence
read at the source:

- Restaurant reviews released by Wongnai (LGPL-3.0): informal written Thai from the platform itself.
- [Thai local language translation dataset](https://huggingface.co/datasets/pythainlp/thai-local-language-translation-dataset):
  15,247 pairs of local Thai with Central Thai, shop and customer-service chat. Probably only the
  Korat part is usable. Translating the Central side gives dialect-to-English pairs.
- US Embassy Bangkok news in Thai and English (human-translated pairs, better as training pairs or a test set).
- Thai statutes (outside copyright), capped; Thai-language journals under CC BY; the open Isan
  conversational speech corpus; phrasebooks on Wikivoyage.
- The spoken Thai sources dropped during cleaning, once they are judged as utterances.
- LST20, with human-marked sentence and clause boundaries, as a test for the Thai splitter. Free with
  sign-up and no named licence, so for testing only.

Known gaps no source fills yet: informal Thai, regional dialect, everyday transactions, health and money.

## 10. Reproducing it

`train/distill_sources.py` builds the source sentences from pinned downloads and writes a manifest
with every count, licence and checksum. `train/teacher.py` translates a file and can resume.

To do before the first release: pin every dependency by exact version; mirror inputs that cannot be
fetched again later (dated Wikimedia dumps, the weekly Tatoeba export); confirm a rebuild is
byte-identical on Linux; add a workflow beside `corpus-release.yml` that builds the sources, attaches
the recorded translations and publishes both, CC BY-SA 4.0, with the attribution notice.

## 11. Open questions

- How far a repaired Thai splitter raises the Thai share, measured against human-marked boundaries.
- Whether pre-training on the source sentences, one model per direction, or noise for look-alike Thai
  characters helps. Each is a short run on the 19M probe model.
- Whether a 75M model has the capacity for the full set, or a larger one is needed.
- Which estimated-quality measure to filter with: the common one carries a non-commercial licence.
