"""One row per unique reel/post across every source, with a caption embedding.

Captions are cleaned of links, @mentions and the reach-bait hashtags that every
reel carries (#explorepage, #fyp...), which otherwise swamp any topic signal.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from enrich import load

DATA = Path(__file__).resolve().parent.parent / "data"

BAIT = re.compile(
    r"#(explore\w*|fy\w*|foryou\w*|viral\w*|trend\w*|reel\w*|insta\w*|tiktok\w*|"
    r"reelsinstagram|reelitfeelit|reelkarofeelkaro|explorepage\w*|aesthetic|pov|real|"
    r"memes?|meme\w*|relatable\w*|funny\w*|comedy|content\w*|like\w*|follow\w*|share\w*)\b"
)


def clean(caption):
    if not isinstance(caption, str):
        return ""
    c = re.sub(r"https?://\S+|@[\w.]+", " ", caption.lower())
    c = BAIT.sub(" ", c)
    return re.sub(r"\s+", " ", c).strip()[:400]


def media_table():
    rows = []
    for name in ("liked", "not_interested", "saved"):
        d = load(name).dropna(subset=["shortcode"])
        rows.append(d[["shortcode", "kind", "caption", "owner", "posted_utc"]])
    dm = pd.read_parquet(DATA / "dm_shares.parquet")
    rows.append(dm[["shortcode", "kind", "caption", "owner"]])
    m = pd.concat(rows).sort_values("caption", na_position="last")
    m = m.groupby("shortcode").first().reset_index()
    m["text"] = m.caption.map(clean)
    return m


def main():
    m = media_table()
    has = m.text.str.len() >= 15
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb = np.zeros((len(m), 384), dtype=np.float32)
    emb[has.values] = model.encode(m.text[has].tolist(), batch_size=128, show_progress_bar=True,
                                   normalize_embeddings=True)
    m["has_text"] = has
    m.to_parquet(DATA / "media.parquet")
    np.save(DATA / "media_emb.npy", emb)
    print(len(m), "media,", has.sum(), "with usable captions")


if __name__ == "__main__":
    main()
