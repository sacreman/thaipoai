# Distillation source sentences

Draft, 11 November 2026. Not legal advice.

This is the record of how the source sentences for distillation are built: where each one comes from,
under what licence, how text is cut into rows, which filters a row passed, and what a rebuild needs. The
sentences are English and Thai text that a teacher model translates once. The translation step is not
covered here. It is not deterministic and is recorded separately.

## 1. What the set is

`data/distill/sources.jsonl` holds 9,065,701 rows: 3,701,301 English and
5,364,400 Thai. It is a pool, not yet the set that will be translated: the owner will reduce it to a
high-quality set in one later pass. Each row is one chat utterance, or one or more whole sentences of prose:

```json
{"id": "sgd-train001-12-3", "text": "I need the car on March 8th.", "lang": "en", "source": "sgd", "register": "task dialogue", "licence": "CC BY-SA 4.0", "context": "What day would you like to pick it up?"}
```

- `context` is what stands just before the row: on dialogue rows the turn before in the same conversation,
  on Thai prose rows the sentence before in the same article. It is there so the translator knows what is
  being answered, or who an unstated subject is. Only `text` is to be translated. 4,234,128 rows carry one.
- `repairs`, when present, lists what was changed in the text: a leading connective removed, a label or
  option marker removed, a bracket of Chinese, Japanese or Korean removed, a doubled Thai sara e corrected, a trailing connective
  removed from a spoken row.
- `tier` is the row's licence tier, set on its source. **1**: a clear licence allowing commercial reuse and
  redistribution, from whoever owns the text. **2**: text that is publicly downloadable where the licence is
  unclear: none is stated, or it covers only the packaging while the text is third parties', or it is an
  odd but permissive one. Tier 2 is allowed for Thai sources only, and never where a term speaks against the
  use (non-commercial, no-derivatives, no scraping or model training, an access agreement). 4,961,503 rows are
  tier 2. They can be left out in one step: see section 7. The manifest counts rows `by_tier` and records,
  for each source, its tier, what the licence check found and the page it was read on.
- `pool`, when present, is `"candidate"`: the row's source scored under the quality bar of section 5 on a
  sample, by this build's reviewers or by an independent review. Its rows are kept and marked, so the later
  reduction can decide. 4,928,852 rows carry it, from seven Thai sources: OpenSubtitles, Thai Wikipedia's
  whole paragraphs, the Common Voice collector file, YODAS, wisesight, the iApp questions and LOTUSDIS. The
  manifest gives each source's latest scores as `last_score`. Only structural junk is dropped at this stage.
- `use`, when present, is `"forward_only"`: the pair made from this row may be trained on only as written,
  the row's text as source and its translation as target, and never reversed. All 271,515 Thai Wikipedia
  rows carry it, and no others. A row without the field may be used in both directions. The manifest gives
  the count under `by_use` and each source's setting.
- `id` holds the source's own id and the row's position in it.

`data/distill/manifest.json` records, for this exact file: its sha256, every source's counts at each
filter, licence and attribution line, and every input file's address, sha256, size and whether it can
be fetched again.

Lengths, measured with the project's tokenizer: 63.1 characters on average
(172 at the 95th percentile) and 17.3 tokenizer pieces
(50 at the 95th percentile). English averages 86.1 characters, Thai 47.6:
subtitle lines are short.

`data/distill/films.json` gives, for each film in OpenSubtitles: the subtitle lines read, the rows in the
set, the share of lines dropped, the share joined to the line before, and the number of lines that mix rude
and polite pronouns or say "ผมนั้น", "ฉันนั้น" (a cheap sign of machine translation). Nothing acts on these yet.

Everything is built by one module, `train/distill_sources.py`. Its tests are in `tests/test_distill_sources.py`.

## 2. Sources

Every source's licence was read on its own page (the repository's licence file, the dataset card, or the
site itself) on 10 or 11 November 2026. A tier 1 source is used only if that licence allows commercial
reuse and redistribution, and can be combined into a CC BY-SA 4.0 release. Tier 2 sources (see `tier` in
section 1) do not meet that test and would have to be left out of such a release, or cleared first.

| Source | Dataset | Language | Register | Licence | Tier | Pool | Rows | Attribution |
|---|---|---|---|---|---:|---|---:|---|
| `byu_chitchat` | BYU Chit-Chat Dataset | English | informal chat | [MIT](https://opensource.org/license/mit) | 1 | passed | 183,594 | Copyright (c) 2019 Perception, Control and Cognition Lab, Brigham Young University |
| `abcd` | Action-Based Conversations Dataset | English | customer-service chat | [MIT](https://opensource.org/license/mit) | 1 | passed | 19,857 | Copyright (c) 2021 ASAPP Research |
| `multiwoz` | MultiWOZ 2.2 | English | task dialogue | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 1 | passed | 42,224 | Budzianowski and others 2018; Zang and others 2020. Also offered under MIT in the GitHub repository |
| `taskmaster2` | Taskmaster-2 | English | spoken task dialogue | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 1 | passed | 48,484 | Byrne, Krishnamoorthi, Ganesh, Dubey, Kim and Cedilnik, Google LLC |
| `taskmaster3` | Taskmaster-3 | English | task dialogue | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 1 | passed | 80,720 | Byrne, Krishnamoorthi and Ganesh, Google LLC |
| `sgd` | Schema-Guided Dialogue | English | task dialogue | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | passed | 114,693 | Rastogi, Zang, Sunkara, Gupta and Khaitan, Google LLC |
| `dolly` | databricks-dolly-15k | English | questions and answers | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | 1 | passed | 5,630 | Databricks, Inc. |
| `oasst2_th` | OpenAssistant Conversations (OASST2), Thai | Thai | questions typed to an assistant | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 1 | passed | 577 | OpenAssistant contributors, LAION |
| `commonvoice_th` | Common Voice sentences, Thai | Thai | everyday sentences | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | passed | 147 | None required |
| `wangchan` | WangchanThaiInstruct, instructions of the CC BY-SA rows | Thai | questions typed to an assistant | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | passed | 1,470 | VISTEC and AIResearch.in.th. Each row's id holds the dataset's own ID |
| `tatoeba_th` | Tatoeba, Thai | Thai | everyday sentences | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) | 1 | passed | 1,935 | Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/`<number>` |
| `tatoeba_en` | Tatoeba, English | English | everyday sentences | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) | 1 | passed | 100,000 | Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/`<number>` |
| `soda` | SODA | English | social chat, written by a language model | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 1 | passed | 100,000 | Kim and others, Allen Institute for AI, 2023 |
| `globalvoices` | Global Voices, English (OPUS GlobalVoices 2018Q4) | English | news features | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 1 | passed | 227,979 | Global Voices authors, each row's id holds the article's date and address, https://globalvoices.org/`<yyyy>`/`<mm>`/`<dd>`/`<address>`/ |
| `enwikinews` | English Wikinews | English | news | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 1 | passed | 217,015 | Wikinews contributors, each row's id holds the page id, https://en.wikinews.org/?curid=`<page id>` |
| `enwikivoyage` | English Wikivoyage | English | travel | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | passed | 479,071 | Wikivoyage contributors, each row's id holds the page id, https://en.wikivoyage.org/?curid=`<page id>` |
| `blognone` | Blognone posts, January 2020 to April 2023 | Thai | technology news | [CC BY 3.0 TH](https://creativecommons.org/licenses/by/3.0/th/) | 1 | passed | 37,461 | Blognone and the post's author. Each row's id holds the post's row number in Noxturnix/blognone-20230430, which gives its author, title and date |
| `blognone_para` | Blognone posts, whole paragraphs | Thai | technology news | [CC BY 3.0 TH](https://creativecommons.org/licenses/by/3.0/th/) | 1 | passed | 9,610 | As for blognone |
| `voathai` | Voice of America Thai, articles (Multilingual Open Text 1.11) | Thai | news | [Public domain](https://www.voanews.com/p/5338.html) | 1 | passed | 59,553 | Voice of America, each row's id holds the article number, https://www.voathai.com/a/`<number>`.html. Collected by Palen-Michel, Kim and Lignos, Multilingual Open Text, whose collection is CC BY 4.0 |
| `voathai_para` | Voice of America Thai, whole paragraphs | Thai | news | [Public domain](https://www.voanews.com/p/5338.html) | 1 | passed | 13,641 | As for voathai |
| `thailaw` | Thai Law Dataset v1.0, sections of Acts and decrees | Thai | legal | [Public domain (Thai Copyright Act, section 7)](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | passed | 6,155 | Collected by PyThaiNLP (Wannaphong Phatthiyaphaibun), released CC0. Each row's id holds the document's number in the dataset |
| `thailaw_clauses` | Thai Law Dataset v1.0, clauses of announcements and regulations | Thai | legal | [Public domain (Thai Copyright Act, section 7)](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | passed | 22,501 | As for thailaw |
| `court` | Thai Supreme Court judgments (deka-bulk) | Thai | legal | [Public domain (Thai Copyright Act, section 7)](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | passed | 77,075 | Supreme Court of Thailand, collected by Korakot Chaovavanich. Each row's id holds the judgment's docid |
| `thwiki` | Thai Wikipedia | Thai | encyclopaedic | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | passed | 271,515 | Wikipedia contributors, each row's id holds the page id, https://th.wikipedia.org/?curid=`<page id>` |
| `simplewiki` | Simple English Wikipedia | English | plain-language encyclopaedic | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | passed | 1,124,167 | Wikipedia contributors, each row's id holds the page id, https://simple.wikipedia.org/?curid=`<page id>` |
| `commonvoice_wiki` | Common Voice sentences taken from English Wikipedia | English | encyclopaedic | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | passed | 957,867 | None required |
| `iapp` | iApp Wiki QA, questions | Thai | questions and answers | [MIT](https://opensource.org/license/mit) | 1 | candidate | 5,367 | Copyright (c) 2021 iApp Technology Co., Ltd. |
| `lotusdis` | LOTUSDIS | Thai | spoken conversation | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | candidate | 1,436 | NECTEC, National Science and Technology Development Agency, Thailand |
| `commonvoice_th_collector` | Common Voice sentences, Thai, sentence collector file | Thai | everyday sentences | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 1 | candidate | 18,962 | None required |
| `thwiki_para` | Thai Wikipedia, whole paragraphs | Thai | encyclopaedic | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 | candidate | 42,914 | As for thwiki |
| `truevoice` | TrueVoice intent, customer utterances | Thai | call-centre requests | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 2 | passed | 8,977 | TrueVoice, published by Kobkrit Viriyayudhakorn |
| `wongnai` | Wongnai review corpus, review lines | Thai | restaurant reviews | [LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html) | 2 | passed | 92,353 | Wongnai and the reviews' authors. Each row's id holds the file and the review's row number |
| `yodas` | YODAS, Thai uploader-supplied captions (th000) | Thai | spoken, video captions | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 2 | candidate | 15,693 | The uploader of each video. Each row's id holds the video id, https://www.youtube.com/watch?v=`<id>`. Collected by Li and others, YODAS, 2023 |
| `opensubtitles_th` | OpenSubtitles Thai (OPUS OpenSubtitles v2018) | Thai | film and television dialogue | [Unstated](https://opus.nlpl.eu/legacy/OpenSubtitles-v2018.php) | 2 | candidate | 4,667,199 | OpenSubtitles, http://www.opensubtitles.org/, and the subtitles' volunteer translators. Each row's id holds the film's IMDb number and the subtitle file's number. Collected by Lison and Tiedemann, OPUS |
| `wisesight` | Wisesight Sentiment Corpus | Thai | social media posts | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 2 | candidate | 9,859 | Wisesight (Thailand) Co., Ltd. and the posts' authors |

Notes:

- **Thai Wikipedia is forward-only.** Its rows are for Thai to English only. On labelled samples 83 to 85
  rows in 100 are clean and none is unusable: as input to a translator the flaws matter little, but as a
  target they would be learned. Each row carries `"use": "forward_only"` so that later stages do not turn
  the pair round.

- **Row ids carry the attribution.** A Wikimedia row's id holds the page id, a Tatoeba row's the sentence
  number, a Voice of America row's the article number, a Global Voices row's the article's date and address,
  a Blognone row's the post's row number in its dataset (which gives author, title and date, but no address).
- **SODA is the one source written by a language model.** Everything else was written by people.
- **MultiWOZ** is MIT in its GitHub repository and Apache-2.0 on its author's dataset card. Cite both.
- **Voice of America Thai.** VOA's terms say text it produces itself is in the public domain, and that
  material from Reuters, AP and AFP on its sites is not. Only articles are read, and any article that names
  one of those agencies anywhere in its text is left out whole. An article that drew on agency copy without
  saying so cannot be detected, so some risk remains.
- **Global Voices** is used without author names, which its archive does not carry.
- **Thai law.** Thai statutes and official orders are not subject to copyright (Copyright Act B.E. 2537,
  section 7). The collection is released as CC0 by PyThaiNLP.
- **Supreme Court judgments.** Judgments are not copyright works (section 7(4)). These are real people's
  cases, so any paragraph that names a private person is left out: a title or rank followed by a name, the
  name of one of the record's own litigants, a child even by initial, or a paragraph that opens with a
  person's initial. So is one with a house address, a land deed's number or a company named as a party. A record whose headnote the Thai Bar wrote is left out whole.
  Judgments from before B.E. 2500 are not read: they are in older spelling.
- **OpenSubtitles Thai (tier 2).** No licence is stated. OPUS, which distributes the collection, asks only
  "Please, add a link to http://www.opensubtitles.org/". The site's robots.txt carries
  "Content-Signal: search=yes,ai-train=no" (read 11 November 2026). The subtitles are volunteers' translations
  of films and programmes. The owner decided to take it as tier 2, removable in one step.
- **Wongnai (tier 2).** The corpus is published by Wongnai itself under LGPL-3.0, a software licence. The
  reviews were written by its users. Wongnai's terms of service forbid users to collect reviews for any
  purpose "เว้นแต่จะได้รับอนุญาตจากทางบริษัท" (unless the company permits); this corpus is the company's own release.
- **wisesight (tier 2).** The compiler waived all rights (CC0 1.0); the posts are third parties'.
- **TrueVoice (tier 2).** Apache-2.0, stated in an individual's repository. TrueVoice's own grant was not
  found, and the speakers are its customers.
- **WangchanThaiInstruct.** Each row has its own licence. Only the 5,014 CC BY-SA 4.0 rows are read; the
  other 34,986 are CC BY-NC 4.0 and are not.
- **OASST2.** Only what people typed to the assistant is taken, not the assistant's replies.
- **Common Voice** sentence text comes from the project's GitHub repository, where it is released as CC0.
- **Tatoeba.** Sentences their authors released as CC0 are left out: as a set they did not pass the quality check.
- **BYU Chit-Chat** is real students' chat. First names and chat handles can appear, and no rule here finds them.
- **Benchmarks** (FLORES-200 and WMT24++ English-Thai) are downloaded only to exclude their sentences.
  FLORES+ is access-controlled and was not used: FLORES-200 holds the same sentences before later corrections.

Sources that were built and then left out because too few of their rows passed the quality check: Common
Voice English collector sentences, Thai and English Wikibooks, OASST2 English, Aya, the answers and input-bound
instructions of WangchanThaiInstruct, and the Tatoeba CC0 sentences. Thai sources that fell under the bar are
no longer left out: they are the candidate pool (`pool` in section 1).

Sources that were examined and not built:

- **Thai Local Language Translation Dataset** (CC BY-SA 4.0). Its Central Thai side is a few sentence
  templates about selling goods with the product, unit and price swapped, and carries dialect spellings.
- **Thai Wikinews.** Under 250 usable rows, many of them headlines.
- **Caption and transcript sets that store a whole video as one run of text**, where utterances cannot be found again.

## 3. How text is cut into rows

- **English prose** is split after `.`, `!` or `?` before a capital, with a list of abbreviations that do not end a sentence.
- **Thai prose.** Thai has no sentence punctuation, and writers put a space between clauses as readily as
  between sentences. A paragraph is cut only where the cut is certain: at a space before a word that opens a
  new sentence and nothing else (a short list of discourse markers such as ต่อมา, นอกจากนี้, อย่างไรก็ตาม), with
  brackets and quotes closed. A Thai row is therefore one sentence or several whole ones, never part of one.
  Merging comes before dropping: a sentence that would be dropped for leaning on the one before it, or for
  being very short, is joined to that sentence again when the two fit in one row. Every row carries the
  sentence before it in the article as `context`, when there is one of at most 400 characters.
- **Whole paragraphs (`voathai_para`, `blognone_para`, and `thwiki_para`, which is built but not taken).**
  Almost half of Thai Wikipedia's paragraphs, and many in the news sources, have no certain cut and are over
  the 250-character row limit. They are not split. Those of at
  most 600 characters and 80 tokenizer pieces go in whole, under source names of their own so they can be
  priced and checked apart. The trainer takes at most 128 pieces a side. On the corpus's own pairs whose Thai
  side is 40 pieces or more (1,231 pairs), English takes 1.2 times the pieces of its Thai on average and
  1.66 times for one pair in twenty. By that measure 3% of paragraphs of up to 80 Thai pieces would have an
  English side over 128; with a limit of 96 it would be 13%. A trained sentence splitter (PyThaiNLP's crfcut)
  and a wider list of openers were both tried on these paragraphs and labelled: 54 and 71 rows in 100 were
  complete, against the 85 needed, so neither is used.
- **Wiki pages.** Templates, tables, files, references, headings and list items are removed. Templates that
  carry words of the sentence keep them (`{{lang}}`, `{{langx}}`, `{{nowrap}}`, `{{convert}}`, `{{ill}}` and the
  like). English wikis give the first 20 or 40 sentences of each article; Wikinews and Thai Wikipedia are read whole.
- **Thai Wikipedia, more carefully**, because Thai has no full stop to show that something is missing:
  - A reference or a line break becomes a space, so two sentences do not fuse.
  - A formula, or a template that leaves no plain text, leaves a mark, and any sentence holding the mark is
    dropped and is not used as context. Notes and pronunciation templates leave nothing and no mark.
  - A paragraph that leads into a list (it ends in ได้แก่, ดังนี้, คือ, ประกอบด้วย or a colon) is dropped
    with everything after it up to the next heading. The lines under a `;` glossary line are dropped.
  - An article that carries a maintenance template is skipped whole: เก็บกวาด (1,558 articles), ปรับภาษา
    (1,315), issues (1,153), ยังไม่ได้แปล (378), แก้ภาษา (350), ไม่เป็นสารานุกรม (349), โฆษณา (74) and a few
    English-named ones, 5,033 of 189,458 articles in all.
- **Task dialogues.** Only the user's or customer's side is taken, with the turn before as context. The other
  side is written from templates.
- **Informal chat (BYU).** Each message is a row, with the message just before it in the same conversation
  as context. The source lists messages in the order they were sent. Two people often type at once, so the
  message before is sometimes the sender's own or belongs to another thread of the talk: it is carried as it is.
  - People split one sentence over two messages. A message is joined to the same sender's message just before
    it when that one ends in a comma or a dangling word (but, and, more, the, a, to, like), or has no closing
    mark and this one opens with a lowercase joining word (and, but, that, out, from, to, of, or, so, than,
    which, because). Joins chain while the row stays within the length limit.
  - A reply of five words or fewer made only of pronouns and auxiliaries ("you do huh?") is kept only when
    the message before it is the other person's question.
  - Messages spelled out a letter at a time are skipped.
- **SODA** is read at one dialogue in four, each turn with the turn before as context. Turns holding a stage
  direction in brackets or asterisks, and turns of eight words or fewer that only greet or thank, are skipped.
- **Global Voices.** Quoted blog posts are left out, and so are posts with fewer than six sentences of their own
  and lines under twelve words. So are captions and credits ("Photo by", "used under", "CC BY", "Original
  post", a line ending in an abbreviated month), lines with a link's language tag such as (RUS), and lines
  opening "In it", "They are" or "Check out".
- **Questions typed to an assistant (OASST2 Thai, WangchanThaiInstruct).** Each one-line message is a row,
  whole. From WangchanThaiInstruct only instructions with no input passage are taken, and not those that ask
  about a case, article or study that is not in the row or that open with a writing-task template.
- **Thai law, clauses.** As for sections of Acts, but the clauses ("ข้อ 5") of announcements, regulations and
  orders, which stop at the signature block. A clause that still holds a signature line, or that has none
  of the words a provision is made with (ให้, ต้อง, ห้าม, อาจ, จะ and the like), is skipped.
- **Supreme Court judgments.** Each paragraph is cut like other Thai prose, with the sentence before it in
  the same judgment as context. Some records are typed with a line break every hundred characters or so:
  their lines are joined into one paragraph first. Docket references "(อันดับ ...)" are removed. Closing orders ("พิพากษายืน", "ยกฟ้อง ...") and paragraphs typed with a space
  between every word are skipped.
- **Subtitles (OpenSubtitles).** One subtitle sentence a row, from the files that keep each film apart and
  each line's times. The row id holds the film's IMDb number, the subtitle file's number and the line's
  number. A file whose Thai was stored in one encoding and read in another (every letter turned into "เธ" or
  "เน" and one more) is read properly again, or left out if that fails: 33 files repaired, 19 left out. A
  sentence is often split across two screens, so a line is joined to the one before it, in the same file and
  within two seconds, when that one hangs (it ends in "...", a comma or a dangling word, or opens a clause
  that needs another) or this one opens with "..." or a joining word, and the two fit a row. A join never
  crosses a change of speaker: not after a line that has finished (it ends in a final particle, "?" or "!"),
  not onto a line the file marks with a dialogue dash, and not between ค่ะ and ครับ. A row carries the row
  before it as context when at most two seconds separate them and no line between them was left out. Sound
  and music tags, formatting marks and a leading speaker dash are removed. Skipped: song lyrics (a music mark
  at either end of the raw line, and the lines between an opening mark and its closing one), credit and
  fansubber lines ("subtitles by", "แปลโดย", "ซับ", web addresses, season announcements), a line where "/"
  runs two speakers together, a line that only calls somebody (a title or honorific with a name), a line
  under 20 Thai letters with no word that shows something is being said (a bare name or noun phrase), and a
  half sentence that found nothing to join. The reader also counts, for each subtitle file, its Thai lines,
  those with a particle of speech and its possessives ("ของฉัน", "ของคุณ"); `films.json` holds the rates for
  the reduction. There is no cap for the source or a film.
- **Video captions (YODAS).** A caption line is a screenful, not a sentence. One video's lines are joined
  until a line closes with a polite particle (ครับ, ค่ะ) not preceded by นะ, or with closing punctuation, or
  with a looser closing word just before a pause of two seconds. Lines cut off by a pause or by a change of
  speaker are thrown away. Videos that are not mostly Thai or that advertise gambling are skipped.
- **Conversation transcripts (LOTUSDIS).** What one person says before another speaks is one turn; its
  pieces are joined, and a row ends where a piece closes on a final particle.
- **Common Voice collector file.** One sentence a row. About a third is machine translation from English;
  rows with the written pronouns that such translation overuses are skipped, and much of it remains.
- **Social media posts (wisesight).** One post a row. Skipped: a post with no verb (a product name or a
  tag); one with two or more quoted phrases (a news headline); advertising and news copy, which shows in
  promotion words or in a brand named twice with no "I", "you" or question in the post; and a post whose
  letters are garbled (marks stacked as no Thai word has them).
- **Restaurant reviews (Wongnai).** A review is typed as several lines and each line is a row. Lines that
  are a dish, a price or a rating are skipped: bullets and numbered items, lines with no verb (for a line
  under 40 Thai letters, no verb of eating or visiting and no word of judgement), lines under 20 characters,
  and lines where lost emoji left "??".
- **Call-centre requests (TrueVoice).** One utterance a row. The publisher blanked some numbers; an
  utterance with such a hole ("เช็ต . บาท", or a lone "G" left of "3G") is skipped, as is a bare topic: no
  verb, or under 30 characters with no particle, pronoun or question word.
- **Questions (Dolly).** Only the questions people wrote are taken, from records that have no reference
  text, and not those typed without a capital or that point at a text that is not in the row.

## 4. Filters, in order

Each row is counted in the manifest under the first filter that removed it.

1. **Cleaning and repairs.** `train.corpus.clean` (Unicode NFC, zero-width characters removed, one spelling of
   sara am, single spaces), then the repairs listed in section 1.
2. **Contact details and codes.** Email addresses, handles, phone-shaped and long numbers, booking and promo
   codes, usernames, street addresses, postcodes, a reply that is only a name, and task-dialogue lines that
   give or ask for a customer's name or account details. The row is dropped, not masked.
3. **Not a sentence.** A link; code or table characters; under 70% of the characters in the row's own
   script; an unclosed quotation or bracket; a line of mostly capitalised words. English prose must start
   with a capital and end in a full stop, question or exclamation mark, and not end at an abbreviation. Thai
   prose must start with a Thai letter and contain a common verb word outside brackets, and must not be a
   note (หมายเหตุ), a timeline entry that opens with a year, or a labelled line ("ภาษาอังกฤษ : ..."). Nor may
   it show the signature of a lost word ("ที่ใน" before a name, "รายการ ของ", "จาก ใน"), an editor's note
   ("ควรตรวจสอบ"), or a pointer to a numbered item such as "(2)". Voice of America's files have lost some
   Latin-script names and programme titles: they are already missing in the published collection, so such
   rows are dropped, not repaired. In spoken sources: captions typed
   a word at a time (a space for every ten characters or more), numbered steps, speaker labels and subtitle credits.
4. **Adult content.** The project's English word list and a short Thai one.
5. **Royal family.** A row that mentions the Thai royal family is dropped unless the source is encyclopaedia
   or news prose (Thai Wikipedia, Simple English Wikipedia, Wikinews, Voice of America). Detection is a list of
   royal titles, vocabulary and names, broad on purpose.
6. **Length.** English: at least 3 words, at most 300 characters and 100 tokenizer pieces. Thai: at least
   8 characters (40 for prose), at most 250 characters and 64 pieces.
7. **Unfinished.** A Thai row that ends on a dangling word (ว่า, คือ, เช่น, และ, ได้แก่, หลังจาก, เพราะ, ซึ่ง, ดังนั้น and similar), or
   opens a condition and never reaches its result. In Voice of America, a row that stops within 20
   characters of a "said that" (ว่า) or a "but" (แต่): it was cut there.
8. **Leans on the row before.** English prose that opens by pointing back ("It", "This", "These", "That",
   "Such", "Therefore", "Instead" and similar). Thai that opens with a joining word. Thai prose that opens
   with a verb whose subject is in the sentence before is counted apart ("subject in the row before") and is
   kept when it carries that sentence as context. Spoken rows may open with "but", "so", "then" or "because" and may
   leave out the subject, but not open with any other joining word. In sources of loose single sentences,
   a demonstrative anywhere.
9. **Pointers and gaps.** "See ...", "this page", "listed below"; a space before punctuation where a word was lost.
10. **Source-specific.** Instructions and chatbot voice in the question sets; four in five of the
    encyclopaedia's stock sentences ("was born in", "is a town in"), three in four of Simple English
    Wikipedia's biography leads ("X (1915 – 1998) was an American cartoonist.") and half of its footballer lines; quoted speech and archaic words in Tatoeba; doubled words and closings
    ("thank you, that will be all") in the task dialogues.
11. **Needs the turn before.** A dialogue line of under five words, or one that opens with "yes", "it", "that"
    and the like, is kept only if it has a context line. A context line with contact details, markup, adult
    words, a royal mention or more than 200 characters is not carried. Nor is the sentence before a Thai
    prose row when it ends on a dangling word, shows a lost word, or begins like the row itself. In ABCD a line after the first is
    dropped when its context cannot be carried, or when the context asks for a name, address, phone, email,
    username, pin, zip code, order id or account id: the answer is an invented identity.
12. **Already in the project.** Anything in the published corpus (train, validation, test), the older
    public pairs, the evaluation suites or the synthetic chat set.
13. **Benchmark overlap.** A row that matches a benchmark sentence, or for English shares any run of eight
    consecutive words with one.
14. **Duplicate.** Two rows are the same if they match after case-folding, replacing digits with 0, removing
    punctuation and spaces, and treating Tatoeba's Tom and Mary as one name. Sources are read in a fixed
    order, conversation first and encyclopaedias last, so the scarcer source keeps the row.
15. **Repetition.** These save money, not quality: the rows differ only in a name, a place or a date.
    - At most 300 SODA rows and 10 Tatoeba rows may begin with the same three words.
    - At most 2,000 Tatoeba rows may begin with the same name (Tom, Yanni, Ziri and the rest), which is 2% of
      what Tatoeba gives. A first word counts as a name when it is capitalised and not a common opening word.
    - In task dialogues a line's frame is the line lower-cased with these masked: the slot values the dataset
      marks in it, any slot value the dataset marks anywhere (so "to Boston" is masked in a line nobody
      annotated), capitalised names, numbers and number words, months, weekdays and times. At most 5 rows
      may share a frame.
    - In task dialogues at most 2 rows may answer the very same previous line.
16. **Near duplicate.** Over the whole pool: a row of 50 or more letters is dropped when its first 40 or
    last 40 letters match an earlier kept row and Python's `difflib` rates the two 0.8 similar or more.

## 5. Quality check

A source passes if a random sample of 100 of its rows, labelled row by row, has at least 85 good rows and at
most 3 that should not be translated. A source well under the bar is not taken. A Thai source a little
under it is kept as the candidate pool (`pool` in section 1), for the later reduction to decide. There are two bars:

- **Prose bar**, for written text: a good row is complete, self-contained and natural.
- **Utterance bar**, for speech, chat, reviews and questions typed to an assistant (OASST2 Thai, Common Voice
  Thai, Wongnai, TrueVoice, WangchanThaiInstruct, OpenSubtitles, wisesight): a good row is a complete
  utterance a person would say and a translator could translate. Fillers and dropped subjects are allowed.

Samples were labelled by model reviewers working from a written rubric, not by Thai speakers, and reviewers
differ by up to 15 points on the same source. Rows with a `context` were judged with it in view. The table
gives the builder's own latest sample of each source. Three independent, stricter reviews were also made of
earlier builds, and the set was changed after each. The last of them passed ABCD (94), Voice of America
(88) and its whole paragraphs (90), and Blognone's whole paragraphs (98), and failed Thai Wikipedia, its
whole paragraphs and YODAS, which was then taken out.

**Thai Wikipedia's sentences are under the bar on the latest sample (82 good rows in 100, none unusable)
and are kept for one direction only**, Thai to English: see `use` in section 1. Its whole paragraphs scored 67
under a stricter reading that counts machine-translated wording as a flaw, and are not taken. What is
wrong in both is mostly the articles' own writing (translated wording, typing errors, missing words), which
no extraction rule removes. ABCD (78 here, 94 in the independent review) and Schema-Guided Dialogue (83
here, 99 independently) are flagged.

| Source | Bar | Good | Flawed | Not to translate |
|---|---|---:|---:|---:|
| `byu_chitchat` | prose | 85 | 12 | 3 |
| `abcd` | prose | 78 | 22 | 0 |
| `multiwoz` | prose | 91 | 8 | 1 |
| `taskmaster2` | prose | 97 | 3 | 0 |
| `taskmaster3` | prose | 97 | 3 | 0 |
| `sgd` | prose | 83 | 17 | 0 |
| `dolly` | prose | 87 | 13 | 0 |
| `oasst2_th` | utterance | 100 | 0 | 0 |
| `commonvoice_th` | utterance | 100 | 0 | 0 |
| `truevoice` | utterance | 89 | 11 | 0 |
| `wongnai` | utterance | 94 | 6 | 0 |
| `wangchan` | utterance | 94 | 6 | 0 |
| `tatoeba_th` | prose | 100 | 0 | 0 |
| `tatoeba_en` | prose | 96 | 4 | 0 |
| `soda` | prose | 97 | 3 | 0 |
| `globalvoices` | prose | 88 | 9 | 3 |
| `enwikinews` | prose | 91 | 9 | 0 |
| `enwikivoyage` | prose | 88 | 12 | 0 |
| `blognone` | prose | 89 | 11 | 0 |
| `blognone_para` | prose | 88 | 12 | 0 |
| `voathai` | prose | 89 | 11 | 0 |
| `voathai_para` | prose | 88 | 12 | 0 |
| `thailaw` | prose | 93 | 7 | 0 |
| `thailaw_clauses` | prose | 94 | 5 | 1 |
| `court` | prose | 95 | 5 | 0 |
| `thwiki` | prose | 82 | 18 | 0 |
| `simplewiki` | prose | 92 | 7 | 1 |
| `commonvoice_wiki` | prose | 95 | 5 | 0 |

Rows of the English dialogue and chat sources were judged with the turn before in view.

The candidate pool, with its latest scores (this build's reviewers; then an independent review where one was made):

| Source | Bar | Good | Flawed | Not to translate | Independent |
|---|---|---:|---:|---:|---|
| `opensubtitles_th` | utterance | 77 | 15 | 7 | 232/46/22 of 300 rows of the reduced set, labelled strictly; 81/19/0 on a 100-row sample before |
| `wisesight` | utterance | 94 | 3 | 3 | 82/13/5, before the advertising rules |
| `yodas` | utterance | 81 | 18 | 1 | 72/22/6 |
| `commonvoice_th_collector` | utterance | 80 | 19 | 1 | none |
| `lotusdis` | utterance | 78 | 22 | 0 | none |
| `iapp` | prose | 76 | 24 | 0 | 82/18/0 |
| `thwiki_para` | prose | 67 | 33 | 0 | 79/21/0 |

## 6. The mix

Each source has a `take`: the number of rows it gives, or `None` for every usable row. Rows are chosen by a
fixed pseudo-random order of their ids (a hash), so a partial take is a spread sample and comes out the same
every time. There is no target total. Capped sources: SODA and Tatoeba English 100,000 each (both repeat
themselves). Every other source gives all its usable rows; no Thai source is capped. Sources are read in
this order: those that passed the bar before the candidate pool, and tier 1 before tier 2. So no row is
dropped as the duplicate of one that the later reduction, or leaving tier 2 out, may remove.

### The reduced set

The pool is not what gets translated. A last stage, `reduce`, reads `sources.jsonl` and writes
`data/distill/distill_set.jsonl`, the set that would be translated, with `distill_set.json` beside it: its
sha256, its rows by source, language, tier and register, what each rule removed from each source, the
look-alike groups, and the rules' settings. The pool is left as it is, so it can be reduced again with other
rules. A row is counted under the first rule that removes it, in this order:

1. **Tier.** `reduce_tier1` leaves every tier 2 row out.
2. **Candidate pool.** A source marked `candidate` stays in the pool only. Two are admitted all the same:
   wisesight, which reached the bar after its fixes (94 good rows in 100), and OpenSubtitles Thai, which is
   close to the bar but under it (232 good rows in 300 in the reduced set and 22 unusable, most of those half sentences). It is in by the
   owner's decision, because it is the only large body of conversational Thai.
3. **Subtitle films in the worst band.** A film that lost 70% or more of its subtitle lines to the filters,
   or where 1% or more of lines mix rude and polite pronouns, is left out whole: 231 of 5,767 films. So is a subtitle file that reads as raw machine translation: one with
   50 Thai lines or more whose share of lines with a particle of speech (ครับ, นะ, สิ, ไหม and the like), less
   its possessives per line, is under 0.04. People who subtitle by hand write those particles; the machine
   translation of the time wrote almost none and kept every English "my" and "your". 180 files of 166 films.
4. **Junk a rule can find.** Rows that are mostly digits; Thai rows that are mostly Latin letters; a
   character repeated six times or more ("55555555", "soooooo"); leftover markup; credit lines.
5. **Too short to teach.** Thai rows under 8 letters once final particles are taken off, and English rows
   under 3 words. A short row that has a context line is a reply ("ใช่ครับ", "ขอบคุณ") and stays, up to 20
   rows of each.
6. **Look-alike duplicates.** Two rows are look-alikes when they are the same after Unicode NFKC folding,
   lower-casing, removing punctuation and spaces, replacing every digit with 0, and for Thai taking off the
   final particles (ครับ, ค่ะ, คะ, นะ, จ้ะ, จ้า, เลย, สิ, ซิ, หรอก, น่ะ, อ่ะ, ล่ะ, เถอะ, ด้วย). One row of each group
   is kept: a tier 1 row before a tier 2 one, then one with context, then the longer text, then the earlier.
   A Thai sentence may stay once in each politeness (closing with ครับ, with ค่ะ or คะ, or with neither),
   because a chat translator has to learn all three.
7. **Near-duplicates.** The step of filter 16 is run again over everything that is left, across sources:
   rows of 50 letters or more whose first or last 40 letters match and that `difflib` rates 0.8 similar. The
   pool has already been through it once, so it finds few.

The stage adds no quality cap and no balance between registers or languages.

The reduced set built from the present pool holds 8,731,390 rows: 3,700,902 English and 5,030,488 Thai
(tier 1 4,202,378, tier 2 4,529,012). Removed: 84,372 rows of candidate sources not admitted; 118,477 of
machine-translated subtitle files; 72,186 of subtitle films in the worst band; 48,917 look-alike duplicates
(51,505 groups, with 23,306 rows kept as politeness variants); 8,541 too short and 11 short replies over the
cap; 1,662 with repeated characters; 127 credit lines; 13 near-duplicates; 5 of markup or numbers.

## 7. Commands

```
python -m train.distill_sources fetch       # download what is missing and check every sha256
python -m train.distill_sources             # candidates, near, mix
python -m train.distill_sources near mix    # redo near-duplicate removal and the mix
python -m train.distill_sources mix         # re-balance after changing a take
python -m train.distill_sources mix_tier1   # the same mix with every tier 2 source left out
python -m train.distill_sources reduce      # from the pool to distill_set.jsonl, the set to translate
python -m train.distill_sources reduce_tier1  # the same without tier 2
```

`candidates` reads the downloads and writes `filtered.jsonl`. `near` writes `candidates.jsonl`. `mix`
writes `sources.jsonl` and `manifest.json`. `reduce` writes `distill_set.jsonl` and `distill_set.json`, in about five minutes. Nothing else is needed: no hand edits, no login, no other script.

Before the build, the same checkout needs the published corpus and its tokenizer
(`python -m train.release fetch`, `python -m train.corpus`, `python -m train.tokenizer`, or the corpus
release unpacked). The build stops if `models/tokenizer-v1/tokenizer.model` is not the pinned one
(sha256 `94f70f275460bc1ebc59fa2f8a2fbd2eaa21f490560d0292f43ddcf2d4265d75`), because the length limits were measured with it.

## 8. Pinned inputs

- **Every download is pinned by address and sha256** in `PINS`: 247 files, 4.2 GB. A file that no longer matches
  stops the build. Hugging Face and GitHub files are addressed by commit or release, so the address itself is fixed.
- **Python and libraries, exact versions used:** Python 3.12.13, `pyarrow` 26.0.1, `sentencepiece` 0.2.3.
  These are the versions the corpus release workflow installs. No other package is needed. The exact
  Python version matters: the filters use Python's Unicode tables.
- **No model files are downloaded at run time.** The only model is the project's own tokenizer.
- **One input is cut out of larger files.** LOTUSDIS is published as audio with its transcripts. `fetch` reads
  only the transcript columns, by HTTP range requests, and writes them to `lotusdis/transcripts.jsonl`. The
  sha256 pinned is of that file. This takes about 18 minutes and needs `pyarrow`.

### Inputs that cannot be fetched again in the same form

The manifest marks these `"upstream": "not kept"`. Wikimedia deletes dated dumps after a few months,
Tatoeba rebuilds its exports weekly, and GitHub does not promise that a commit's archive keeps the same bytes.

| File | Size | Fetched from |
|---|---:|---|
| `tatoeba/eng_sentences_detailed.tsv` | 199.0 MB | https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2 |
| `tatoeba/tha_sentences_detailed.tsv` | 0.9 MB | https://downloads.tatoeba.org/exports/per_language/tha/tha_sentences_detailed.tsv.bz2 |
| `tatoeba/eng_sentences_CC0.tsv.bz2` | 1.3 MB | https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2 |
| `sgd/sgd.zip` | 37.0 MB | https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/zip/e852981ae34990f4358979625854259302feaa78 |
| `simplewiki/simplewiki-20261101-pages-articles.xml.bz2` | 360.1 MB | https://dumps.wikimedia.org/simplewiki/20261101/simplewiki-20261101-pages-articles.xml.bz2 |
| `thwiki/thwiki-20261101-pages-articles.xml.bz2` | 511.5 MB | https://dumps.wikimedia.org/thwiki/20261101/thwiki-20261101-pages-articles.xml.bz2 |
| `enwikivoyage/enwikivoyage-20261101-pages-articles.xml.bz2` | 132.6 MB | https://dumps.wikimedia.org/enwikivoyage/20261101/enwikivoyage-20261101-pages-articles.xml.bz2 |
| `enwikinews/enwikinews-20261101-pages-articles1.xml-p1p1500000.bz2` | 38.1 MB | https://dumps.wikimedia.org/enwikinews/20261101/enwikinews-20261101-pages-articles1.xml-p1p1500000.bz2 |
| `enwikinews/enwikinews-20261101-pages-articles1.xml-p1500001p3000000.bz2` | 6.5 MB | https://dumps.wikimedia.org/enwikinews/20261101/enwikinews-20261101-pages-articles1.xml-p1500001p3000000.bz2 |
| `enwikinews/enwikinews-20261101-pages-articles1.xml-p3000001p3095843.bz2` | 2.9 MB | https://dumps.wikimedia.org/enwikinews/20261101/enwikinews-20261101-pages-articles1.xml-p3000001p3095843.bz2 |

Mirroring them would take 1.29 GB in 10 files. The largest is 511 MB, well under GitHub's 2 GB
limit for a release asset. All may be re-hosted: the Wikimedia dumps are CC BY-SA 4.0 (Wikinews CC BY 4.0),
Tatoeba is CC BY 2.0 FR with a CC0 part, and Schema-Guided Dialogue is CC BY-SA 4.0. Each needs its licence
and attribution kept beside it. Nothing has been uploaded. The two Tatoeba sentence files are stored
unpacked, as they were first downloaded for the project.

Until they are mirrored, `data/distill/raw` on the build machine is the only copy, and `fetch` cannot
rebuild it from an empty checkout once the publishers have moved on.

## 9. What a GitHub Actions job still needs

No workflow exists yet. Open items:

1. **Mirror the inputs above** and point their addresses at the mirror.
2. **The older public pairs.** `data/public/train.jsonl` and `data/public/validation.jsonl` are read for
   the exclusion steps and are not built by any script in the repository. Either add their build, or
   exclude their sources directly from pinned downloads.
3. **Build the corpus and tokenizer first** in the same job, or download the corpus release.
4. **Confirm on Linux.** The build has only run on macOS (Apple silicon).

### Determinism

The build uses no random numbers, no threads of its own and no clock. Files and archive members are read in
sorted order, hashes are BLAKE2 and SHA-256 and do not depend on Python's hash seed, and sets are used only
for membership tests, never for output order. Output is written as UTF-8 with `\n` line ends. The present version was run twice from the same downloads and gave
the same `sources.jsonl`, `candidates.jsonl`, `manifest.json` and `films.json`, byte for byte.

What could still differ on another platform, and has not been tested:

- **Python version.** Case-folding, `\w`, `\d` and NFC normalisation follow the Unicode tables of the
  interpreter. Use exactly 3.12.13.
- **SentencePiece.** Piece counts decide the length filter. The same version should segment the same way
  on Linux, but that is assumed, not shown.
- **pyarrow and expat.** They only read data. A different version should not change the text read.
- **Line ends and locale.** Every file is opened as UTF-8 explicitly. Windows is not supported.

### Runner fit

Measured on an Apple M4 Max laptop:

| Step | Wall time | Peak memory |
|---|---:|---:|
| `fetch`, all files (3.6 GB) | not timed as one run | not measured |
| `candidates`, `near` and `mix` together | 45 minutes | 5.3 GB |

Peak disk is about 16 GB: 4.2 GB of downloads, 4.1 GB `filtered.jsonl`, 4.0 GB `candidates.jsonl` and 3.4 GB
`sources.jsonl`, plus the corpus. A standard GitHub-hosted Linux runner has 4 cores, 16 GB of memory,
about 14 GB of free disk and a 6-hour limit, so memory fits and disk no longer does without deleting
`filtered.jsonl` after `near`, or a larger runner. The build is single-threaded and the
runner's cores are slower: allow an hour and set the job's timeout above the corpus workflow's 30 minutes.
`sources.jsonl` should be compressed for a release asset.
