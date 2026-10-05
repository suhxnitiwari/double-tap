# Double Tap

**Live:** https://suhxnitiwari.github.io/double-tap/

Six years of my Instagram (2020–2026), parsed from Meta's data export: what kinds of reels I like, what I mark "not interested," how fast I decide, when I scroll, and how my taste changed.

The site looks like an Instagram profile. Each highlight is one question, and the charts live inside the stories.

## Pipeline

```
Instagram Data/raw/{main,spam}/   Meta's HTML exports (not in this repo)
        │
pipeline/parse.py        12 export files → tidy parquet tables (likes, not interested, saves, comments…)
pipeline/parse_dms.py    reels shared in DMs: who sent what, when; message text is never kept
pipeline/enrich.py       timezone fix + post time decoded from each reel's shortcode
pipeline/embed.py        one row per reel, caption embeddings (local multilingual MiniLM)
pipeline/classify.py     topic labels: keyword rules → logistic regression → creator majority
pipeline/analyze.py      every number on the site → site/data.js
        │
site/index.html          the page (vanilla JS + SVG, no build step)
```

Run everything:

```bash
./pipeline/run.sh
```

## Things worth knowing

- **Timezone.** Meta stamps the HTML export in a fixed UTC−8 with no daylight saving. I proved it by decoding each reel's creation time from its shortcode (base64 of a media ID whose top bits are milliseconds since Instagram's epoch). In every month, the earliest likes land exactly 8.0 hours "before" the reel existed.
- **Watch history.** Instagram keeps only about a week of `videos_watched`, and most of it is sponsored. Likes, saves, sends and rejections stand in for "what I watch."
- **Topic accuracy.** The classifier agrees with the keyword rules on 73% of held-out captions. A hand check of 40 random labels found about 30 correct. Treat topics as trends, not per-reel truth.
- **Privacy.** DM threads are stored as hashed IDs. Friends' handles never reach the site, and no caption or message was sent to any API.
