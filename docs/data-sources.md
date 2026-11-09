# English-Thai parallel data for a ~300M-parameter from-scratch bilingual Transformer

Research date 2026-11-09. Not legal advice.

Labels used below: **[sourced]** read on the cited page. **[estimate]** my own number or inference.

## 1. Bottom line

- Clean-licence, human or crowd translated data totals about 495k pairs: 26,157 (CC BY local) + 468,568 (SCB human/crowd, CC BY-SA).
- That is about 54x below the 26.9M rows behind quickmt, the closest comparable model (195M parameters).
- Reaching millions of pairs means using web crawl (HPLT) and subtitles (OpenSubtitles). Both have unclear rights in the underlying text.
- Nearly every curated source is **English-origin**. Thai-origin text is the biggest quality gap for the th->en direction and is not fixed by any public dataset found.

## 2. Local data (manifest.json)

`alt` 19,896, `talpco` 1,366, `tatoeba` 4,895 = **26,157 unique pairs**. `train_rows` 51,330 + `validation_rows` 984 = 52,314 = 26,157 x 2, so the 51k rows are the same pairs in both directions, not 51k pairs.

Contribution: about 0.9% of the core mix below, and 5% of the clean human-translated pool.

## 3. Source by source

### SCB-MT-EN-TH-2020
- Card: https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020
- Paper: https://arxiv.org/abs/2007.03541, full text read at https://ar5iv.labs.arxiv.org/html/2007.03541
- Licence: **CC BY-SA 4.0** (card: "The center releases the dataset and baseline models under CC-BY-SA 4.0"). Code repo is Apache-2.0 (https://github.com/vistec-AI/thai2nmt) but that covers code, not data. The paper itself states no licence.
- Size: 1,001,752 pairs. Card splits: train 801,402, validation 100,173, test 100,177.
- Filtering: per-dataset handcrafted rules (token min/max, character share, EN/TH token ratio, exact duplicates) plus Universal Sentence Encoder cosine similarity: web crawl threshold 0.7, Thai government 0.5, others not listed. Mean similarity per subset from 0.58 (nus_sms) to 0.82 (msr_paraphrase), paper Table 4. No Cohere-style score exists in the paper or card; I found no such score for this dataset.

| Subset (paper Table 2) | Pairs | Original language | Method | Conversational? |
|---|---:|---|---|---|
| task_master_1 | 222,733 | English | Professional translation | Yes, task dialogs |
| product_review_translator | 133,330 | English (CTRL-generated) | Professional translation | Moderate, informal |
| product_review_yn | 280,208 | English (CTRL-generated) | **Google Translation API output, human accept/reject** | Moderate |
| product_review_crowd | 24,587 | English (CTRL-generated) | Crowd | Moderate |
| nus_sms | 43,750 | English (Singlish SMS) | Crowd | Yes, but lowest similarity 0.58 |
| mozilla_common_voice | 33,797 | English | Crowd | Yes, short spoken sentences |
| msr_paraphrase | 10,371 | English news | Crowd | No |
| thai_websites | 120,280 | Mixed | Auto-aligned | Low |
| paracrawl | 60,039 | Mixed | Auto-aligned | Low |
| wikipedia | 33,756 | Mixed | Auto-aligned | No |
| assorted_government | 25,398 | Mixed | Auto-aligned | No |
| apdf | 13,503 | Mixed | Auto-aligned | No |

Totals: professional/crowd 468,568; Google-output 280,208; auto-aligned 252,976; sum 1,001,752.

Discrepancies: card lists mozilla_common_voice as 2,451 train rows and a wrong-looking "4.05%". Paper text and Table 2 also differ on review counts (43,374 vs 133,330 professional; 307,570 vs 280,208 Google), possibly reviews vs segments. Count the `subdataset` column after download before finalising.

Direction: all human/crowd subsets are English-origin. Thai-origin text can only be in the auto-aligned 252,976, share unknown.

**Google output risk (280,208 pairs):** an older third-party copy of Google's Service Specific Terms bars using Cloud Translation "to create, train, or improve ... any other machine translation engine" (https://static.carahsoft.com/concrete/files/7615/1311/1617/Google_Service_Specific_Terms.pdf). The current page (https://cloud.google.com/terms/service-terms) has no such line in section 25, but section 17 bars using generated output to create models similar to a Google model; whether Translation counts as an AI/ML service is not stated, and I read only the first 100k of 144k characters. Treat this subset as excluded for an open release.

### OPUS en-th inventory (API listing)
https://opus.nlpl.eu/opusapi/?source=en&target=th&preprocessing=moses&version=latest

| Corpus | Version | Pairs | Licence found | Note |
|---|---|---:|---|---|
| OpenSubtitles | v2024 | 24,831,103 | **None stated**; asks for link to opensubtitles.org and citation (https://opus.nlpl.eu/legacy/OpenSubtitles-v2018.php) | Most conversational bulk; noisy |
| CCAligned | v1 | 10,746,372 | No IP claim; "We do not own any of the text" (https://opus.nlpl.eu/legacy/CCAligned.php) | Web; overlaps HPLT/ParaCrawl |
| ParaCrawl-Bonus | v9 | 2,175,923 | Not verified | Web |
| XLEnt | v1.2 | 1,236,145 | No licence named (https://opus.nlpl.eu/legacy/XLEnt.php) | Entity pairs, not sentences |
| SCB_MT_EN_TH | v1.0 | 988,260 | CC BY-SA | Same data as SCB |
| QED | v2.0a | 264,677 | **"RESEARCH purpose only"** (https://opus.nlpl.eu/legacy/QED.php) | Excluded |
| TED2020 | v1 | 160,762 | **CC BY-NC-ND** (TED policy) | Excluded |
| NeuLab-TedTalks | v1 | 102,773 | **CC BY-NC-ND** (TED policy) | Excluded |
| bible-uedin | v1 | 124,386 | Not verified | Off-register |
| Tanzil | v1 | 93,540 | Not verified | Off-register |
| KDE4 | v2 | 70,634 | Not verified | Software strings |
| wikimedia | v20260327 | 65,759 | Wikipedia content terms expected, not verified | Encyclopedic |
| translatewiki | v2026-07-01 | 31,431 | Not verified | UI strings |
| ALT | v20191206 | 18,088 | CC BY 4.0 | Already local |
| Tatoeba | v2026-07-08 | 1,195 | Per sentence | Already local (local count 4,895 is higher) |
| ELRC, GNOME, Ubuntu, tldr-pages | various | under 3k each | Not checked | Skip |

TED policy: https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy

OpenSubtitles quality: general literature reports many misaligned or badly translated pairs and alignment-score pruning (https://nr.no/publikasjon/1650877/). No Thai-specific error rate found. Direction [estimate]: mostly English-origin films with Thai subtitles.

### HPLT v2
https://arxiv.org/html/2503.10267v3 Table 5: `tha_Thai` **4,088,354** deduplicated pairs. Project page (https://hplt-project.org/datasets/v2.0) licenses packaging as CC0 and disclaims the underlying text, with takedown by request, stated for monolingual data; I did not find the parallel-data licence on that page. An HPLT paper reports CC0 for its parallel releases (https://aclanthology.org/2025.mtsummit-2.21). Web-crawled, register unverified, some web pages are themselves MT.

### ALT Parallel Corpus
https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html 20,106 sentences, 1,893 articles, English Wikinews translated by NICT. Parallel corpus CC BY 4.0. News register, not conversational. Standard split 18,088 / 1,000 / 1,018. Thai is one of 13 languages, no Thai-specific notes. Overlap with OPUS ALT (same data).

### ChindaMT-Grounded
https://huggingface.co/datasets/iapp/ChindaMT-Grounded and paper https://arxiv.org/html/2609.34770
- 1,973,358 records (1,172,061 en-th, 801,297 th-en), from 1,206,946 unique source pairs. Alpaca instruction format.
- **Outputs are machine-generated.** Constraints are extracted from reference translations, then targets are regenerated by Qwen3.5-35B-A3B-FP8 and kept only if all constraints pass. Authors report a reference-free audit rating regenerated targets at least as good as references. Human evaluation was 3 raters on 100 outputs of the *model*, not an audit of the dataset.
- Qwen: Apache-2.0 per the Base model card (https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base); I did not read the official instruct card or the FP8 quantisation's card. Apache would allow training.
- Sources: hplt 660,898, paracrawl 649,378, xlent 326,134, scb_2020 239,871, opensubtitles 63,048, bible 17,326, wikimedia 8,212, en-th-texts 8,035, tatoeba 392, elrc 64. Selected for instruction-following difficulty from a 17.85M-record pool.
- Licence CC BY-SA 4.0.
- Glossary: no glossary data. A mandated glossary is explicitly out of scope; "specific terms" constraints come from the reference only.
- Verdict: duplicates HPLT/ParaCrawl/SCB, built for constraint-following, Bible-heavy preview rows. Do not use for the first run. Sample before any later use.

### LORELEI Thai LDC2023T08
https://catalog.ldc.upenn.edu/LDC2023T08, README https://catalog.ldc.upenn.edu/docs/LDC2023T08/README.txt
- Monolingual 39,182,511 words. Found parallel 2,855,245 words (news 2,273,823, weblog 581,422). Translated: FromEng 141,212 words, ToEng 124,511 words. No segment counts. Roughly 130k-190k found-parallel segments [estimate at 15-22 words each].
- Price not shown ("Login for the applicable fee"). 2023 newsletter said "Non-members may license this data for a fee" (https://ldc-upenn.blogspot.com/2023/07/), while other announcements said members-only (https://linguistlist.org/issues/34/2237/). Catalog lists both.
- Non-members "cannot use LDC data to develop or test products for commercialization" (https://www.ldc.upenn.edu/data-management/using/licensing). Model release is not addressed; the non-member PDF could not be read here.
- Verdict: excluded. Small, news-heavy, restricted. ToEng 124k words is the only genuinely Thai-origin human set found, but too small and restricted.

### FLORES+
https://huggingface.co/datasets/openlanguagedata/flores_plus Thai listed (`tha`). dev 997, devtest 1,012 sentences per language. CC BY-SA 4.0. Gated: login and accept terms; do not redistribute publicly unless protected from scraping. Card states it "should not be used as training data". Blind test set is not in this repository. Fine as a held-out benchmark. Keep it out of every training source; check overlap by hash since crawl data could contain it [estimate].

### Other
- **quickmt-train.th-en** https://huggingface.co/datasets/quickmt/quickmt-train.th-en: 26,934,996 rows, deduplicated aggregate via mtdata from CCAligned, TED, ELRC, OPUS collections, HPLT v2, OpenSubtitles (2016, 2018, 2024), QED, SCB, TED2020, Tanzil, Tatoeba, Bible, WMT24++ Thai and others. No licence stated. Contains the non-commercial TED and research-only QED data, and WMT24++ which would contaminate that benchmark. Useful as a size reference only.
- **FineTranslations** https://www.infoq.com/news/2026/01/huggingface-fine-translations/ and https://huggingface.co/datasets/HuggingFaceFW/finetranslations: English output is Gemma 3 27B translation of FineWeb2. ODC-By. Thai not confirmed. **Gemma terms (https://ai.google.dev/gemma/terms) define a model trained on Gemma's synthetic outputs as a Model Derivative**, and distribution requires passing on Gemma use restrictions, so plain Apache-2.0 release would not be clean. Excluded.
- **LEXiTRON**: about 53k TH-EN and 83k EN-TH entries [docstring, unverified], licence "other"/LEXiTRON License (https://huggingface.co/datasets/SEACrowd/lexitron/blob/main/lexitron.py, https://opend-portal.nectec.or.th/dataset/lexitron-2-0). Terms not read. Skip until NECTEC confirms.
- **ParaNames** CC BY 4.0 (https://arxiv.org/abs/2202.14035v3). Names only; Thai count not found. Later glossary or name-transliteration use only.
- **NLLB-Seed**: could not verify whether Thai is included or its licence (PDF not parseable here). Treat as absent.
- **Mozilla Common Voice**: already in SCB (33,797 per paper). No separate source needed.
- **Commercial data vendors**: found no public en-th dataset price list.

## 4. Ratings

Scale: H / M / L. Accuracy = translation accuracy, Nat = naturalness, Conv = conversational relevance, Origin = Thai-origin coverage. These are my judgements from the method descriptions, not measured.

| Source | Accuracy | Nat | Conv | Origin | Overlap | Licence | Usable pairs | Cost |
|---|---|---|---|---|---|---|---:|---|
| ALT | H | M | L | None | With OPUS copy only | CC BY 4.0 | 19,896 | Free |
| TALPCo | H | M | M | Mixed, unverified | None | CC BY 4.0 | 1,366 | Free |
| Tatoeba | M-H | M | H | Some Thai-origin, unverified | Small | Per sentence | 4,895 | Free |
| SCB human/crowd | H/M | M | H for dialogs, SMS, voice | None | SCB copies | CC BY-SA | 468,568 | Free |
| SCB auto-aligned | M | M | L | Some, share unknown | HPLT/ParaCrawl | CC BY-SA | up to 252,976 | Free |
| SCB Google-output | M | L-M | M | None | None | CC BY-SA + Google terms risk | 0 for release | Free |
| HPLT v2 | M-L | M | L | Some, share unknown | CCAligned, ParaCrawl, SCB web | Packaging CC0 per paper, underlying web text unowned | ~2.0M after filtering [estimate] of 4.09M | Free |
| OpenSubtitles v2024 | L-M | M-H | H | Mostly English-origin | Prior OPUS versions | No licence stated | ~3M after filtering [estimate] of 24.8M | Free, legal risk |
| CCAligned | L-M | M | L | Some | HPLT, ParaCrawl | No licence stated | Not used | Free |
| Wikimedia | M | L | L | Some | SCB wikipedia | Unverified | ~50k [estimate] | Free |
| QED, TED | H | H | M | None | With each other | NC/research | 0 for release | n/a |
| LDC LORELEI | H for translated | L | L | Yes, 124k words | Web crawls | Restricted | 0 for release | Fee unknown |
| ChindaMT-Grounded | Unknown | Unknown | L | n/a | HPLT, ParaCrawl, SCB | CC BY-SA | Not used | Free |
| FLORES+ | H | M | L | n/a | Benchmark only | CC BY-SA, gated | 0 training | Free |

## 5. Licence implications

- **CC BY 4.0** (ALT, TALPCo): attribution only.
- **CC BY-SA 4.0** (SCB, ChindaMT, FLORES): ShareAlike on model weights is legally unsettled. If weights count as an adaptation, weights must be CC BY-SA, not Apache-2.0. Precedent only, not clearance: ChindaMT weights are Apache-2.0 (with Qwen terms) while its data is CC BY-SA; quickmt weights are CC BY 4.0. Lowest-risk option if SCB is included: publish weights under CC BY-SA 4.0.
- **Non-commercial / research-only** (TED, QED, LDC non-member): excluded for an open release.
- **No stated licence** (OpenSubtitles, CCAligned, XLEnt): a licence on the weights cannot fix unknown rights in the source text. Business decision.
- **Model-output terms**: Gemma output trains into a Model Derivative; Google Translate output has a possible restriction.

## 6. Pair-count evidence

| Evidence | Pairs | Model | Result |
|---|---:|---|---|
| SCB paper, SCB only | 1,001,752 | Transformer BASE | th->en BLEU 17.15 (Google 14.19); en->th 12.95 (Google 15.36) on IWSLT 2015 |
| SCB paper, SCB + OPUS | +3,318,153 OPUS after filter | Transformer BASE | th->en 28.33; en->th 17.77; beats Google both ways |
| quickmt | 26,934,996 rows | 195M parameters | FLORES devtest th->en BLEU 29.32, COMET22 87.15, about equal to NLLB-1.3B; en->th BLEU 9.55, chrF2 54.77, COMET22 84.52, above NLLB-1.3B (6.29, 46.91, 83.25) |
| Sennrich & Zhang 2019 https://ar5iv.arxiv.org/html/1905.11901 | 159k de-en | Tuned RNN | BLEU 32.8; ko-en 94k sentences 10.37 |
| Koehn & Knowles 2017 https://ar5iv.labs.arxiv.org/html/1706.03872 | Words | RNN | en-es NMT passes SMT at about 15M words; 25.7 BLEU at 24.1M words, 31.1 at 385.7M |
| ChindaMT | 17.85M records pool across both directions | Fine-tuned Qwen, not from scratch | Size reference only |

SCB BLEU caveat [estimate]: IWSLT test sets are TED talks, OPUS includes QED educational subtitles, so test-train overlap may flatter the OPUS gain.

Reading: about 500k pairs is workable for a smaller model but not for 300M from scratch; roughly 3-6M is where evidence shows usable quality; 25M+ is what the closest comparable used. Pair counts are not directly comparable between models or test sets. I found no published learning curve for Thai.

Shortfall: clean human set 495k is 2% of 26.9M. Core mix 2.8M is 10%. With subtitles 5.8M is 22%.

## 7. Recommended first mix

| Part | Pairs | Status |
|---|---:|---|
| ALT + TALPCo + Tatoeba | 26,157 | Local, CC BY; oversample |
| SCB human/crowd | 468,568 | CC BY-SA; oversample dialogs, SMS, voice |
| SCB auto-aligned | ~250k | CC BY-SA, after dedupe |
| HPLT v2 en-th, top half by filter | ~2.0M | [estimate] |
| Wikimedia | ~50k | [estimate], verify licence |
| **Core total** | **~2.8M** | |
| OpenSubtitles v2024, filtered | ~3M | [estimate], owner decision |
| **With subtitles** | **~5.8M** | |

Excluded: SCB Google-output, TED, QED, LDC, ChindaMT, FineTranslations, FLORES+ (benchmark only), quickmt aggregate, Bible/Tanzil, software strings.

Thai-origin gap: commission Thai-written conversational sentences translated into English, and hold some out as a th->en test set.

## 8. Commissioned review and translation prices

| Source | Rate | Notes |
|---|---|---|
| Gengo https://gengo.com/pricing-languages/ | From $0.09 Standard, $0.18 Advanced | Thai pair not shown; older Gengo pages show $0.06/$0.12 (https://gengo.com/cheap-translation/) |
| GoTranscript https://gotranscript.com/thai-translation-services | From $0.07/word | Advertised |
| ProZ profiles e.g. https://www.proz.com/profile/1297922 | $0.06-0.12/word, $22-40/hour | Self-reported |
| Thai-market guide https://www.thailandstarterkit.com/business/thai-translator/ | 1-1.25 baht/word, 400-500 baht/page | Undated |
| MTPE general https://translated.com/resources/translation-pricing-trends-2026 and others | About half of full rate, $0.05-0.15/word | Not Thai-specific |

Per-sentence [estimate], 10-word conversational sentence: full translation about $0.30 (Thai-market, about 33 THB/USD) to $1.20 (international); review/post-edit about half.

Per 10,000 sentences (100k words) [estimate]: translation $3,000-$12,000; review $1,500-$6,000. Thai has no spaces, so agree whether billing is on English source or Thai characters. One source reports Thai text about 15% longer.

Not found: Appen, TELUS, Toloka per-sentence rates; Upwork Thai hourly rates.

## 9. Not verified

- LDC2023T08 fee, the non-member agreement text, and whether model release is allowed.
- NLLB-Seed Thai presence and licence.
- OpenSubtitles.org terms and any upstream rights.
- HPLT parallel-data licence on the project page; only the paper snippet and monolingual terms were seen.
- FineTranslations Thai presence.
- ParaCrawl-Bonus, Wikimedia, Bible, Tanzil, KDE4, translatewiki licence terms.
- LEXiTRON terms.
- Google Cloud service terms beyond the first 100k characters.
- Filter yields for HPLT and OpenSubtitles are estimates; quality ratings are from method descriptions, no data sampled.
- TALPCo and Tatoeba licences taken from your brief, not re-read.
- Your own chat logs are not assessed or counted.
