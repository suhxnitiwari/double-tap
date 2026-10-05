"""Shared fixes applied on top of the parsed tables.

Timezone: Meta stamps the HTML export in a fixed UTC-8 with no daylight saving.
We proved it by decoding each reel's post time from its shortcode: the earliest
likes in every month land ~8.0h "before" the reel existed, summer and winter alike.
So export time + 8h = UTC, then convert to Austin.
"""
import pandas as pd

EXPORT_OFFSET = pd.Timedelta(hours=8)
LOCAL_TZ = "America/Chicago"
IG_EPOCH_MS = 1314220021721
B64 = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")}


def posted_at(shortcode):
    """UTC time a post was created: shortcode is base64 of the media id, whose top bits are ms since IG epoch."""
    if not isinstance(shortcode, str):
        return pd.NaT
    n = 0
    for c in shortcode[:11]:
        n = n * 64 + B64[c]
    return pd.Timestamp((n >> 23) + IG_EPOCH_MS, unit="ms")


def localize(df):
    df = df.copy()
    df["utc"] = df["time"] + EXPORT_OFFSET
    df["local"] = df["utc"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    if "shortcode" in df:
        df["posted_utc"] = df["shortcode"].map(posted_at)
        df["age_h"] = (df["utc"] - df["posted_utc"]).dt.total_seconds() / 3600
    return df


def load(name):
    from pathlib import Path
    return localize(pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / f"{name}.parquet"))
