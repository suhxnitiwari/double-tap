"""Turn Instagram's HTML export into tidy tables.

Every export file is a list of `main > div` rows; each row is a nested set of
label/value tables. We pull the reel/post URL, caption, owner username and the
row's own timestamp (the last date in the row), then write one parquet per source.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW = Path.home() / "Desktop/Instagram Data/raw"
OUT = Path(__file__).resolve().parent.parent / "data"

DATE = re.compile(r"[A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} ?[ap]m")
MEDIA = re.compile(r"instagram\.com/(reel|reels|p|tv)/([A-Za-z0-9_-]+)")

SOURCES = {
    "liked": "your_instagram_activity/likes/liked_posts.html",
    "not_interested": "ads_information/ads_and_topics/posts_you're_not_interested_in.html",
    "watched": "ads_information/ads_and_topics/videos_watched.html",
    "viewed": "ads_information/ads_and_topics/posts_viewed.html",
    "saved": "your_instagram_activity/saved/saved_posts.html",
    "story_likes": "your_instagram_activity/story_interactions/story_likes.html",
    "stories_viewed": "your_instagram_activity/story_interactions/stories_viewed.html",
    "comments": "your_instagram_activity/comments/post_comments_1.html",
    "reel_comments": "your_instagram_activity/comments/reels_comments.html",
    "searches": "logged_information/recent_searches/word_or_phrase_searches.html",
    "profile_searches": "logged_information/recent_searches/profile_searches.html",
    "interests": "your_instagram_activity/ai/interest_categories.html",
}


def parse_time(s):
    s = re.sub(r"\s*([ap]m)$", r" \1", s)
    return datetime.strptime(s, "%b %d, %Y %I:%M %p")


def pairs(row):
    """(label, value) for every two-cell table row inside this export row."""
    out = []
    for tr in row.select("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) == 2:
            out.append((tds[0].get_text(strip=True), tds[1].get_text(" ", strip=True)))
        elif len(tds) == 1 and tds[0].find("div") and not tds[0].find("table"):
            # single cell: label as bare text, value in a nested div (comments use this)
            label = next((s.strip() for s in tds[0].find_all(string=True, recursive=False) if s.strip()), "")
            out.append((label, tds[0].div.get_text(" ", strip=True)))
    return out


def first(ps, label):
    return next((v for k, v in ps if k == label), None)


def parse_row(row):
    text = row.get_text(" ", strip=True)
    dates = DATE.findall(text)
    ps = pairs(row)
    m = MEDIA.search(str(row))
    surface = first(ps, "Surface")
    return {
        "time": parse_time(dates[-1]) if dates else None,
        "kind": ("reel" if m and m.group(1) in ("reel", "reels", "tv") else "post") if m else None,
        "shortcode": m.group(2) if m else None,
        "owner": first(ps, "Username"),
        "caption": first(ps, "Caption"),
        "comment": first(ps, "Comment"),
        "media_owner": first(ps, "Media Owner"),
        "search": first(ps, "Search"),
        "interest": first(ps, "Interest"),
        "surface": surface,
        "hashtags": [d.get_text(strip=True) for d in row.select("h2 + div div._a6-p")
                     if d.find_parent("div").find_previous("h2").get_text(strip=True) == "Hashtags"]
        if row.find("h2", string=re.compile("Hashtags")) else [],
    }


def parse_file(path):
    soup = BeautifulSoup(path.read_text(), "lxml")
    return [parse_row(r) for r in soup.select("main > div")]


def main():
    OUT.mkdir(exist_ok=True)
    for name, rel in SOURCES.items():
        frames = []
        for account in ("main", "spam"):
            p = RAW / account / rel
            if not p.exists():
                continue
            df = pd.DataFrame(parse_file(p))
            df["account"] = account
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(axis=1, how="all")
        df.to_parquet(OUT / f"{name}.parquet")
        print(f"{name:16} {len(df):>7} rows  {df['time'].min()} -> {df['time'].max()}", file=sys.stderr)


if __name__ == "__main__":
    main()
