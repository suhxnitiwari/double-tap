"""Reels shared in DMs: who sent which reel, when. Message text is never kept.

Friends are stored as a hashed thread id, so nothing personal leaves this script.
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from parse import DATE, MEDIA, OUT, RAW, parse_time


# my own display names on both accounts. A thread where nobody else ever speaks
# (main <-> spam, or a chat with myself) is a bookmark folder, not a share.
ME = {"Suhani 💗", "Hani 💕"}


def thread_rows(path):
    soup = BeautifulSoup(path.read_text(), "lxml")
    thread = hashlib.sha1(path.parent.name.encode()).hexdigest()[:8]
    senders = set(h.get_text(strip=True) for h in soup.select("main > div > h2"))
    if senders <= ME:
        return
    people = len(senders)
    for row in soup.select("main > div"):
        m = MEDIA.search(str(row))
        if not m:
            continue
        stamp = row.select_one("div._a6-o")
        # attachment box is [caption, owner, <div><a href=reel></div>]
        link = row.find("a", href=MEDIA)
        box = link.parent.parent.find_all("div", recursive=False) if link else []
        bits = [d.get_text(" ", strip=True) for d in box]
        yield {
            "time": parse_time(DATE.search(stamp.get_text()).group()) if stamp else None,
            "sender": row.h2.get_text(strip=True) if row.h2 else None,
            "thread": thread,
            "group": people > 2,
            "kind": "reel" if m.group(1) in ("reel", "reels", "tv") else "post",
            "shortcode": m.group(2),
            "caption": bits[-3] or None if len(bits) >= 3 else None,
            "owner": bits[-2] or None if len(bits) >= 2 else None,
        }


def main():
    frames = []
    for account in ("main", "spam"):
        files = sorted((RAW / account / "your_instagram_activity/messages/inbox").glob("*/message_*.html"))
        df = pd.DataFrame([r for f in files for r in thread_rows(f)])
        # "me" is the one sender present in nearly every thread
        me = Counter(s for t, g in df.groupby("thread") for s in g.sender.unique()).most_common(1)[0][0]
        df["from_me"] = df.sender == me
        df = df.drop(columns="sender")
        df["account"] = account
        frames.append(df)
        print(account, len(files), "files", len(df), "shares", f"me={me!r}", df.from_me.mean().round(2))
    pd.concat(frames, ignore_index=True).to_parquet(OUT / "dm_shares.parquet")


if __name__ == "__main__":
    main()
