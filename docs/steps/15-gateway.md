# Step 15: the gateway

9 November 2026. `cmd/gateway` is the translation API. It takes `POST /v1/translate`, renders
the same prompt the model is trained on, calls any OpenAI-compatible server, runs the guards
and repairs once. It serves `/healthz` and `/metrics`, and logs only status and milliseconds.
A test proves message text never reaches the log.

## It ran for real

Gateway on the Mac, in front of `serve/mps_server.py` running the base Gemma 4 E4B:

| Request | Result | Time |
|---|---|---|
| English to Thai, female speaker | `ok`, ending in ค่ะ | 9.6 s (first request after loading) |
| English to Thai, with chat history | `repaired`, first try wrote both gender particles | 1.1 s |
| Thai to English, 10 โมง | `ok`, "at 10 AM" | 0.7 s |
| English to Thai, 1450 baht, male speaker | `ok`, "1,450 บาท", ending in ครับ | 0.7 s |
| English to French | refused with a 400 | |

Four requests prove the path works. They say nothing about load, which waits for vLLM on real
GPUs in step 18.

## What the guards are worth

I ran every guard over 1,320 saved translations that Opus had already scored for meaning.
129 of them were misleading.

| Guard | Fired | Of those, misleading | Decision |
|---|---|---|---|
| `gender_hedge` (ครับ/ค่ะ together) | 43 | 0 | Repair. The rubric ignores style, but a reader sees it |
| `negation` | 29 | 6 | Report only. Wrong 79% of the time |
| `times` | 3 | 3 | Repair |
| `no_thai` | 2 | 2 | Repair |
| `numbers` | 0 | 0 | Repair |

**The guards do not protect meaning.** Together the repairing guards fired on 48 translations,
3.6% of the total, and only 5 of the 129 misleading ones were among them. Most real errors are
a wrong actor or a lost idiom, and a regular expression cannot see those. The guards catch
format faults and a narrow set of factual slips. Meaning has to come from the model.

Two guards changed because of this measurement:

- `numbers` fired 11 times with no real errors, all on texting shorthand such as `l8 2`, `new 1`
  and `2gether`. It now compares only numbers of two or more digits, so a changed single digit
  ("3 nights" becoming "4 nights") goes uncaught.
- `times` is new. The real number errors were mostly clock times, which the old guard skipped.
  สองทุ่ม is 8 pm, and models wrote 7 pm or 10:30 pm. It covers evening hours only.

## What this does not show

- Both changes were tuned on the same 1,320 translations they are measured on. The `times`
  result in particular comes from only three cases.
- The `gender` guard needs the speaker's gender, which the saved suites do not record, so it
  was tested only by unit tests.
- Whether a repair improves the meaning is unmeasured. The gateway reports first-try and final
  failures separately so the benchmark in step 14 can measure it.

Approved by the owner on 9 November 2026.
