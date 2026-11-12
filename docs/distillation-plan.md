# Distilled corpus: plan

First written 10 November 2026, corrected to the state of 12 November 2026. Working plan, not a
result. Not legal advice. Counts are as of 12 November and will change.

The human-translated corpus (section 4 of [SPEC.md](../SPEC.md)) is about 299,000 pairs of assistant
commands, reviews and task dialogue. A model trained on it translates that kind of text well and
almost nothing else. This plan adds a second, much larger corpus: a billion clean words of English
and Thai text in one language, each row translated once by a hosted teacher model. Nothing has been
translated yet. The [source record](distillation-sources.md) holds the detail of every source and rule.

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

**Target, restated on 11 November.** The best translator for informal one-to-one chat between Thai
and English in a model under 300 million parameters. The proof is Thai speakers preferring its
output, blind, to DeepSeek's, Typhoon Translate's and Google's on chat messages none of them has seen.
FLORES-200 is a secondary line: match or beat Typhoon Translate 1.5 (54.22 and 52.46 above).

This replaces the target of 10 November, "as close to 60 on FLORES-200 as possible", and its
milestones of 43, 50 and 54. An adversarial review on 11 November judged the top FLORES score out of
reach for a model trained from random weights on DeepSeek's translations, and estimated 46 to 55 chrF
for the student against the teacher's 59 and 64. The wins it judged realistic are per size and on chat.

On English to Thai, a 150-sentence check of a larger hosted model scored 61.4, so about 60 looks like
the practical ceiling for this measure. Thai to English scores on FLORES need care. Its English side
is original public text, and a large model given the Thai can reproduce the English it has already
read: the same check scored 95 in that direction. English to Thai does not show this.

## 2. What the corpus is

Text in one language, each row translated once by the teacher. One row is one chat utterance, or one
or more whole sentences of prose up to a whole paragraph, with its source, register, licence, licence
tier and, where there is one, the line before it.

**Size.** One billion clean words of source text: 600 to 700 million Thai words first, then English.
Progress is counted in words, not rows, because rows now run from a short subtitle line to a
700-character paragraph. The owner accepted that reaching this may take weeks; the November date for a
public result no longer binds the corpus work.

A pair can be used in both directions:

- **Forward** (real source, teacher's target): trains the direction the row was translated in.
- **Reversed** (teacher's text as source, real text as target). Reversed pairs must be tagged so the
  model can tell: untagged, they cost up to 12.1 BLEU on natural input (section 8, Marie). Rows marked
  `forward_only` are never reversed.

**Mix, from a literature review on 12 November.** 60 to 70% Thai-origin by words, never below half and
never all of it. Thai-origin rows put natural Thai on one side: they are the forward pairs for Thai to
English, and the only natural Thai targets for English to Thai. Both origins together beat either
alone (section 8: Koehn; Wu; Vamvas). The English is chosen for register: dialogue, forum and fiction,
with encyclopaedic text a minority. The final mix is set at training time, not at collection.

An earlier note in this plan held Thai near 30% on the strength of Typhoon 2's ratio of 3 Thai to 7
English. That was wrong: the ratio describes instruction-tuning examples for a pretrained language
model, not a translation corpus.

## 3. Sources

**Rule: sources are in two tiers, and every row carries its tier.**

- **Tier 1** has a licence that allows commercial use, redistribution and derived works, granted by
  whoever holds the rights to the text.
- **Tier 2** is public Thai text with no stated licence, or with a licence that covers only the
  packaging. It was added on 11 November 2026 because tier 1 Thai ran out near 500,000 rows. It is
  kept apart so that the set can be built without it in one step, and whether it stays is decided
  before any release.

Out of both tiers: a dataset whose own licence forbids this use (non-commercial, no-derivatives,
no-training), a source behind an access agreement, and the output of any model whose terms forbid
training on it. Each licence is read on the source's own page and recorded with an attribution line in
the manifest.

The web-crawl text is in tier 2 on the strength of the collection's licence, which covers the
packaging and not the pages. The terms of the individual sites have not been cleared. Tier 2 is for
experiments, and none of it is released before that is settled.

**Where the Thai stands on 12 November.** 899.7 million Thai words have been collected that pass the
local rules. About 684 million of them are counted clean, at the shares measured on judged samples.

| Source | Words passing the rules, millions | Counted clean, millions |
|---|---:|---:|
| FineWeb2 Thai web crawl, short paragraphs (up to 330 characters) | 424.3 | 314.0 |
| FineWeb2 Thai web crawl, long paragraphs (331 to 700 characters, marked long) | 353.9 | 276.1 |
| The reduced set of 11 November (subtitles, Wikipedia, news, law, court judgments, reviews) | 53.2 | 43.2 |
| Thai financial news and forum | 49.1 | 37.7 |
| Government Public Relations Department news | 10.6 | 6.4 |
| Pantip and krathu-500 forum posts | 6.2 | 4.7 |
| Thai Wikipedia discussion pages | 2.3 | 1.8 |

The crawl is 18 files of FineWeb2, read behind a pinned list of 174 domains, whole paragraphs only.
By kind, the clean words are about 283 million general web, 280 million news, 73 million fiction,
27 million subtitles, 11 million forum and conversational, 7 million encyclopaedic and 3 million legal.
About 22 million clean words are tier 1 and about 662 million tier 2.

**Conversational Thai is under 2% of the total and is the known gap.**

Subtitle dialogue (OpenSubtitles Thai, tier 2) is 4.67 million rows in the pool. No licence is stated,
and the site carries an "ai-train=no" signal, which the manifest records. After film-level and
structural fixes about 77 rows in 100 are clean by strict review. It was kept as the only large body of
conversational Thai, and is now a small share of the total by words.

**English** is unchanged at about 3.7 million rows, roughly 54 million words, from 14 tier 1 sources.
English collection resumes after the Thai is settled.

Left out, and why:

| Kind | Examples | Reason |
|---|---|---|
| Web crawls, other than FineWeb2 | HPLT v2, Mangosteen web, MADLAD-400, OSCAR, CC100, mC4, CulturaX | HPLT: its server now refuses downloads, and a sample was 53 clean in 100 by rules. Mangosteen web: no page addresses, 37 in 100. MADLAD-400 Thai: no page addresses, waits for a filter. OSCAR and CulturaX need an access agreement |
| Scraped news | ThaiSum, HSE Thai Corpus | Publishers' terms restrict reuse; ThaiSum also scored 75 |
| Mined parallel web text | ParaCrawl bonus (Koehn 2024), OPUS collections | Distilled data did better for Thai (section 8) |
| Non-commercial or research-only | BEST, ORCHID, LOTUS, Thai National Corpus, most of WangchanThaiInstruct, Prachatai (taken off the domain list), openbase.in.th, GigaSpeech 2 Thai and its relabelling | Licence |
| Scanned or machine-captioned text | Royal Gazette OCR sets, auto-captioned YouTube text | Hard-wrapped lines and mis-encoded vowels; no sentence boundaries |
| Failed on quality | Thai Wikibooks, Wikisource, Wikiquote and Wiktionary examples | Too few clean rows |
| Built but not taken | iApp, YODAS, LOTUSDIS, Thai Wikipedia whole paragraphs from the first extraction | Under the bar on samples |
| Held as candidates | Company annual reports | Hard-wrapped and tabular |
| Tweet sets | | Posted by people who did not grant a licence. Pantip posts, Wisesight and Wongnai reviews are in as tier 2 |
| Outputs of other models | Hunyuan-MT, and any model whose terms forbid training on its output | Terms. The teacher is DeepSeek, whose terms allow it |

## 4. What the audit found

This is the first audit, kept as the record of why the rules in section 5 exist. The source list has
changed since, and current sample scores are in the [source record](distillation-sources.md).

On 10 November, 100 random rows from each of 29 sources were read and labelled by four independent
model reviewers: translate as is, usable but flawed, or should not be translated. About 62% of rows
were clean, and the reviewers would have paid to translate about 45% of the 5.6 million rows then in
the build. A share from 100 rows is good to about 9 points.

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

Rows are cleaned in four stages, cheapest first: rules, local filters, the teacher as a judge, and
checks after translation. The rules come first in this section and the other stages after them.

1. **Cut rows correctly, before any filter.**
   - Web crawl and the other bulk Thai sources: whole paragraphs only. Rows cut out of longer
     paragraphs scored 54 clean in 100 and are no longer taken.
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
   sentences. The check is repeated by a reviewer who did not write the rules. A tier 1 source under
   the bar is not taken. Since 11 November a Thai source under it may be kept and marked, for the
   filters below and the reduce stage to decide.
7. **Web crawl.** A pinned list of domains, and on each page sentence-boundary, headline, label,
   garble and contact-detail rules, then exact and near-duplicate removal. Rules took the crawl from
   64 to about 74 clean rows in 100 and then levelled off.

Rules worth copying from other Thai corpus builders: keep a document only if at least half its
characters are Thai, because standard language detectors fail on Thai (Mangosteen); filter gambling
and obscene text (Mangosteen); collapse characters repeated more than three times and keep spaces as
tokens, since spaces mark Thai sentence boundaries (WangchanBERTa).

### Local filters

Three small classifiers (XLM-R base, fine-tuned on the laptop) score rows keep-or-drop: one for prose,
one for forum and conversational text, one for subtitles. They were trained on 295,000 rows that
larger hosted models labelled on 11 and 12 November. Those labels agreed with a strict human-style
review on 88 to 92 rows in 100, and rows such a judge keeps were 92 to 98% clean by the strict
standard. The judging models' output is used only as labels, to train the filters and to measure
quality. None of their text is training data.

Held-out results at the middling threshold (p >= 0.8):

| Filter | Rows kept | Clean among kept | Clean before |
|---|---:|---:|---:|
| Prose, first version (crawl paragraphs) | 77% | 88% | 74% |
| Forum and conversational | 73% | 91% | 75% |
| Subtitles | 74% | 87% | 74% |

Scores are written to side files beside the rows. Nothing is dropped at collection, and the threshold
is chosen at the reduce stage. As of 12 November all subtitle, forum, discussion, financial and
government-news rows are scored and the crawl is being scored. An improved prose filter, trained on
142,700 labels, is still training.

**The filters cannot yet be rebuilt from this repository.** Their training scripts, labels and weights
are outside it. This is an open item (section 10).

### The teacher as a judge

DeepSeek Flash was tried as a row judge on 19,699 labelled rows. It kept 68% of rows at 91% clean,
agreed with the labels 82% of the time, and cost about $8 per million rows. It is not in use: the
owner ruled out any paid AI use until the billion words are collected.

### After translation

The teacher's flags and automatic checks (length ratio, untranslated output), sections 6 and 7.

## 6. Enrichment

- **Context.** A dialogue row carries the previous turn of the same conversation, and a Thai prose
  row the sentence before it. The teacher is shown it and asked to translate only the current line
  (`train/teacher.py` does this). The model is to be trained with its `<ctx>` channel; the trainer does
  not read context yet. Rows that need a previous turn and have none are dropped.
- **Teacher flags.** Planned, not yet in `train/teacher.py`: the teacher returns a fixed marker in
  place of a translation when the source is not a complete utterance. This catches cut-off rows that
  no rule can see.
- **Provenance.** Each row keeps its article, video or dialogue id, its position, and which repairs
  were applied.

## 7. Buying and filtering

Nothing has been bought beyond trials costing about $6 in total, and no paid AI use is allowed until
the billion words are collected.

- **Teacher:** DeepSeek Flash, thinking off, one output per row. It scored within 1.5 points of a
  larger hosted model on English to Thai, so a stronger teacher would buy little on this kind of text.
- **Price:** not measured. A trial on short rows measured 50 input and 20 output tokens a row. The
  estimate of $95 to $100 made on 10 November was for that day's much smaller build and no longer
  applies. A priced pilot of 1,000 rows comes first and has not been run.
- **Order:**
  1. Collect and filter the Thai.
  2. Collect English to 30 to 40% of the total.
  3. The priced 1,000-row pilot, to measure price and look at quality.
  4. Distil in steps, training and scoring after each, and go on only while the score still climbs.
     Distilled Thai-to-English quality in one study rose 22.8, 23.5, 24.1, 24.2 as data grew, and
     levelled off near a billion words (section 8).
  5. Training that can pass the teacher on chat: the best of several teacher outputs per row,
     human-translated chat pairs weighted at the end, and reinforcement learning with a reward. The
     reward needs human judgement of conversational Thai.
  6. Proof: a chat test set of 500 to 1,000 messages each way, human-translated, never published or
     trained on, and a blind ranking by Thai speakers.
- **Filter before training:** drop flagged rows, length-ratio outliers, wrong-language and
  untranslated outputs, and the lowest-scoring pairs by an estimated-quality measure. Hunyuan-MT
  filters its synthetic pairs the same way.
- **Record:** model name as served, prompt, settings and date go in the release, with the translated
  rows. The teacher is not deterministic, so a rebuild reproduces the source sentences, not the
  translations.
- **Trainer.** `train/train.py` reads the teacher's output (`--distilled`), teaches a `forward_only`
  pair in one direction, opens a reversed pair's source with `<sep>` so machine-written input is
  marked, and puts a context line between two `<ctx>` markers. A side may be up to 256 pieces. None
  of this has been run on distilled pairs yet, because none exist. All pairs are held in memory,
  which will not hold for the full set.
- **Scoring** is still the project's own chrF with greedy decoding. Moving to sacreBLEU chrF and
  chrF++ with signatures, and to beam search, is an open item.

## 8. What other work shows

Read from the papers and model cards themselves. Scores are theirs and are not on our scoring script.

| Work | Finding | Use here |
|---|---|---|
| Koehn, WMT 2024, [Neural Methods for Aligning Large-Scale Parallel Corpora](https://aclanthology.org/2024.wmt-1.132) | Small Marian models, Thai to English, FLORES-200 spBLEU: mined web pairs 19.3; with OPUS 20.4; pairs distilled from NLLB 25.3; distilled plus OPUS 23.5. Real Thai translated by the teacher 23.9; real English translated and reversed 18.4; both 25.3. Student 25.3 against teacher 31.5. Gains level off near a billion words | Distil instead of mining; use both origins; Thai-origin rows matter most for Thai to English; expect to land below the teacher; the size target |
| Wu and others, EMNLP 2019 | At the same total, equal parts forward-translated and back-translated data beat either alone by 2.6 and 3.5 BLEU | Keep both origins in the mix |
| Vamvas and others 2026, [arXiv 2603.25489](https://arxiv.org/abs/2603.25489) | With an LLM teacher, text that originated in the low-resource language won in both directions: 42.5 against 26.6 BLEU into the low-resource language | Thai-origin text as the larger share |
| Marie and others 2020 | Untagged reversed pairs cost up to 12.1 BLEU on natural input | Tag reversed pairs |
| Shurtz and others, MRL 2025, [When Scripts Diverge](https://aclanthology.org/2025.mrl-main.22) | Transformer-base from scratch on 1,080,329 cleaned pairs (2,175,880 before cleaning), FLORES+ devtest chrF++: Thai to English 39.4, English to Thai 42.6; best variant 43.2 and 43.9 | The same-class baseline to beat. Half of mined pairs do not survive cleaning |
| Xu and others, [ALMA](https://arxiv.org/abs/2309.11674) | Fine-tuning on single-language text before 58,000 human pairs: 22.17 BLEU without the first stage, 29.89 with it | A pre-training stage on the source sentences before translation training |
| Tencent, [Hunyuan-MT](https://arxiv.org/abs/2509.05209) | About 3M pairs including DeepSeek-synthesised ones, filtered by estimated quality and an LLM judge; then 268,000 curated pairs; then reinforcement learning with metric rewards | Broad then curated; filter synthetic pairs. Its licence forbids training other models on its output |
| SCB 10X, [Typhoon 2](https://arxiv.org/abs/2412.13702) | Dropped Thai translated from English instruction data because it introduced errors; Thai to English 3:7 worked best in its fine-tuning | Prefer Thai written in Thai. The 3:7 ratio is for instruction-tuning a pretrained model and says nothing about the Thai share of a translation corpus |
| SCB 10X, [Typhoon Translate](https://huggingface.co/scb10x/typhoon-translate-4b) | A 4B Thai-English translator built on Qwen3, Apache-2.0, trained on translations from larger models with human spot checks. Reports no FLORES, BLEU or chrF score | The laptop-sized system to compare with; scored here in section 1 |
| SCB 10X, [Typhoon](https://arxiv.org/abs/2312.13951) | Typhoon-7B-Instruct, FLORES-200 English to Thai: BLEU 31.14, chrF 46.62 | Published reference point |
| Jasmine and Upstage, [JAI-1](https://arxiv.org/abs/2510.08620) | FLORES English to Thai BLEU: GPT-4o 46.5, GPT-4 41.3, a 10.7B translator fine-tuned on 0.5M pairs 41.6. Public Thai text by size: web crawls 58 billion tokens, clean sources about 0.3 billion | What a pretrained base buys; how little clean Thai exists |
| VISTEC, [WangchanBERTa](https://arxiv.org/abs/2101.09635) | 94% of its 78.5 GB was social and forum text supplied privately and not released. Its sentence splitter scored F1 0.69 to 0.96 by domain | Informal Thai at scale is not public; Thai sentence splitting is imperfect everywhere |
| VISTEC and others, [Mangosteen](https://arxiv.org/abs/2507.14664) | A Thai pipeline cut Common Crawl from 202M to 25M documents and raised a small model's score | Thai-specific filters, section 5 |
| AIResearch, [SCB-MT-EN-TH-2020](https://airesearch.in.th/releases/machine-translation-models/) | A 74M Transformer on 1,001,752 pairs: IWSLT 2015 BLEU Thai to English 17.15, English to Thai 12.95; with 3.3M OPUS pairs 28.33 and 17.77 | An earlier model of the same size, to score on the same tests |
| VISTEC, [WangchanThaiInstruct](https://arxiv.org/abs/2508.15239) | BLEU and ROUGE agreed with human judges at Kendall 0.03 and 0.005; an LLM judge at 0.13 | Overlap scores are a rough guide. Claims about chat need judged or human ratings |

## 9. Still to add

The Thai word count is near its target, but it is mostly web prose and news. What is missing is
conversational Thai, under 2% of the total. Of the leads listed here on 10 November, the Wongnai
reviews and the Thai statutes are now in, and the Thai local language translation dataset was examined
and not built (its Central Thai side is a few templates). Leads still open, none verified, each needing
its licence read at the source:

- US Embassy Bangkok news in Thai and English (human-translated pairs, better as training pairs or a test set).
- Thai-language journals under CC BY; the open Isan conversational speech corpus; phrasebooks on Wikivoyage.
- LST20, with human-marked sentence and clause boundaries, as a test for the Thai splitter. Free with
  sign-up and no named licence, so for testing only.

English collection resumes once the Thai is settled, to 30 to 40% of the total, chosen for register:
dialogue, forum and fiction first, encyclopaedic text a minority. More than half of the present
English rows are encyclopaedic.

Known gaps no source fills yet: informal Thai chat, regional dialect, everyday transactions, health and money.

## 10. Reproducing it

What is true on 12 November:

- **Reproduced.** `train/distill_sources.py` builds the pool and the reduced set of 11 November from
  pinned downloads and writes a manifest with every count, licence and checksum. That set was rebuilt
  byte for byte from the same downloads, on macOS only.
- **Not yet reproducible from this repository.** The crawl collection (`train/crawl.py` and its
  domain list), the filter scores and the filters themselves were added on 11 and 12 November. The
  crawl reader is not yet committed, the crawl files are not pinned by checksum, and the filters'
  training scripts, labels and weights are outside the repository.
- **Too big for a standard runner.** The pool has outgrown the disk of a standard GitHub Actions
  runner.
- **No workflow.** `corpus-release.yml` builds only the human-translated corpus and predates all of
  this.
- `train/teacher.py` translates a file and can resume. It does not yet record the prompt and date
  with its output, or ask for the flag of section 6.

To do before the first release: bring the crawl, the scores and the filters into the repository with
their inputs pinned; pin every dependency by exact version; mirror inputs that cannot be fetched again
later (dated Wikimedia dumps, the weekly Tatoeba export); confirm a rebuild is byte-identical on Linux;
decide where a build of this size runs; add a workflow that builds the sources, attaches the recorded
translations and publishes both with the attribution notice. Whether tier 2 stays is decided before
any release, and the release's licence depends on that decision: a CC BY-SA 4.0 release can hold
tier 1 only.

## 11. Open questions

- Whether tier 2 text stays in a release, and under what terms the rest is then published.
- Where enough conversational Thai comes from.
- Which filter thresholds to reduce at, once the crawl is scored and the improved prose filter is trained.
- Moving scoring to sacreBLEU chrF and chrF++ with signatures, and to beam search.
- Who gives the human judgement of conversational Thai that the chat test set, the blind ranking and
  the reward all need.
- Whether pre-training on the source sentences, one model per direction, or noise for look-alike Thai
  characters helps. Each is a short run on the 19M probe model.
- Whether a 75M model has the capacity for the full set, or the 297M one is needed.
- Which estimated-quality measure to filter with: the common one carries a non-commercial licence.
