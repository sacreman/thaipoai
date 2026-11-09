# Step 3: what is in the archive?

8 November 2026. `cmd/dataset-projector` replayed ThaiPo's archive and counted it. It decodes
metadata fields only and reports totals, and a test proves its report cannot carry message
text or chat IDs. Every chat-day was checked against its manifest sha256 first.

## Result

14,465 chat-days from 3,116 chats. No damaged or unreadable days. The run took 139 seconds.

| Filter | Rows left |
|---|---|
| All translations | 212,389 |
| Typed text only | 211,066 |
| Literal translation path only | 139,965 |
| English-Thai pair | 99,519 |
| Excluding one industry pack | **93,005** |

Of the 93,005:

| Split | Rows |
|---|---|
| Thai to English | 66,462 |
| English to Thai | 26,543 |
| Group chats | 91,206 |
| One-to-one chats | 1,774 |
| Label already written by DeepSeek | 41,560 |
| Label written by another provider, so it needs relabelling | 51,445 |

## What it means

- The estimate before this count was 100,000 to 500,000 rows. The real figure is at the low end,
  and further filters will reduce it: flagged rows, explicit content and consent.
- Thai to English has two and a half times the data of English to Thai. It is also the direction
  that needed the most improvement in step 2.
- 98% of usable rows come from group chats, so consent has to cover everyone who wrote in them.
- Rows from the Native Reply path are the largest single loss, about 71,000 rows.

At the 3,607 rows an hour measured in step 1, one pass of 93,005 rows would take about 26 hours
on the Mac. Real rows may be longer than the test rows.

## Limits

Filters that need the message text, such as explicit content and personal details, were not
applied. Consent has not been applied either.
