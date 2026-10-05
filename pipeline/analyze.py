"""Compute every number the site shows and write site/data.js.

Nothing on the site is typed by hand: each story frame reads from this file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from enrich import load, localize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YEARS = list(range(2021, 2027))  # 2020 has 9 likes; too thin to stand as a year


def shares(df):
    known = df[df.topic != "Unknown"]
    return known.topic.value_counts(normalize=True)


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def main():
    media = pd.read_parquet(DATA / "media.parquet")[["shortcode", "topic", "topic_source"]]
    tag = lambda d: d.merge(media, on="shortcode", how="left").assign(topic=lambda x: x.topic.fillna("Unknown"))
    liked = tag(load("liked"))
    reels = liked[liked.kind == "reel"]
    ni = tag(load("not_interested"))
    saved = tag(load("saved"))
    dms = tag(localize(pd.read_parquet(DATA / "dm_shares.parquet")))
    sent, recv = dms[dms.from_me], dms[~dms.from_me]
    comments = load("comments")
    story_likes = load("story_likes")

    out = {}

    # ---------- totals
    out["totals"] = {
        "likes": len(liked), "liked_reels": len(reels), "liked_posts": int((liked.kind == "post").sum()),
        "not_interested": len(ni), "saved": len(saved), "dm_shares": len(dms), "sent": len(sent),
        "received": len(recv), "comments": len(comments), "story_likes": len(story_likes),
        "first": str(liked.local.min().date()), "last": str(liked.local.max().date()),
        "likes_by_account": liked.account.value_counts().to_dict(),
    }

    # ---------- taste
    s = pd.DataFrame({"liked": shares(reels), "saved": shares(saved), "sent": shares(sent),
                      "rejected": shares(ni)}).fillna(0)
    s = s.sort_values("liked", ascending=False)
    out["topics"] = [{"topic": t, **{k: r(v, 4) for k, v in row.items()},
                      "save_lift": r(row.saved / row.liked, 2), "send_lift": r(row.sent / row.liked, 2),
                      "reject_lift": r(row.rejected / row.liked, 2)} for t, row in s.iterrows()]
    out["topic_known_share"] = r((reels.topic != "Unknown").mean())
    acc = reels[reels.topic != "Unknown"].pipe(lambda d: pd.crosstab(d.topic, d.account, normalize="columns"))
    acc["gap"] = acc.spam - acc.main
    out["accounts"] = [{"topic": t, "main": r(v.main, 4), "spam": r(v.spam, 4)}
                       for t, v in acc.reindex(acc.gap.abs().sort_values(ascending=False).index).head(8).iterrows()]

    # ---------- not interested
    out["ni"] = {
        "surface": ni.surface.value_counts(normalize=True).round(3).to_dict(),
        "by_year": ni.groupby(ni.local.dt.year).size().reindex(YEARS, fill_value=0).to_dict(),
    }
    ev = pd.concat([liked.assign(ev="like")[["account", "local", "ev"]],
                    ni.assign(ev="ni")[["account", "local", "ev"]]]).sort_values(["account", "local"])
    ev["gap"] = ev.groupby("account").local.diff().dt.total_seconds() / 60
    ev["next"] = ev.groupby("account").ev.shift(-1)
    ev["next_gap"] = -ev.groupby("account").local.diff(-1).dt.total_seconds() / 60
    after = ev[(ev.ev == "ni") & (ev.next_gap <= 20)]
    out["ni"]["streak"] = r((after.next == "ni").mean())

    # ---------- decision speed
    in_session = ev[ev.gap <= 20]
    bins = [-0.1, 0, 1, 5, 20]
    labels = ["same minute", "1 min", "2–5 min", "6–20 min"]
    out["speed"] = {
        "gap_dist": {e: pd.cut(g.gap, bins, labels=labels).value_counts(normalize=True).reindex(labels).round(3).to_dict()
                     for e, g in in_session.groupby("ev")},
    }
    rr = reels[reels.age_h >= 0]
    age_bins = [0, 1, 24, 24 * 7, 24 * 30, 24 * 365, 1e9]
    age_labels = ["< 1 hour", "1–24 hours", "1–7 days", "1–4 weeks", "1–12 months", "> 1 year"]
    out["speed"]["age_dist"] = pd.cut(rr.age_h, age_bins, labels=age_labels).value_counts(normalize=True).reindex(age_labels).round(3).to_dict()
    out["speed"]["age_median_days"] = r(rr.age_h.median() / 24, 1)
    out["speed"]["fresh_by_year"] = rr.groupby(rr.local.dt.year).age_h.apply(lambda a: (a < 24).mean()).reindex(YEARS).round(3).to_dict()

    both = sent[["shortcode", "utc", "account"]].merge(liked[["shortcode", "utc", "account"]], on=["shortcode", "account"], suffixes=("_s", "_l"))
    lag = (both.utc_s - both.utc_l).dt.total_seconds() / 60
    out["speed"]["sent_also_liked"] = r(len(both) / len(sent))
    out["speed"]["liked_before_send"] = r((lag >= 0).mean())
    out["speed"]["same_minute_send"] = r((lag.abs() < 1).mean())

    got = recv[["shortcode", "utc", "account"]].merge(liked[["shortcode", "utc", "account"]], on=["shortcode", "account"], suffixes=("_r", "_l"))
    lag = (got.utc_l - got.utc_r).dt.total_seconds() / 60
    out["speed"]["recv_liked"] = r(len(got) / len(recv))
    out["speed"]["recv_liked_first"] = r((lag < 0).mean())
    out["speed"]["recv_lag_median_h"] = r(lag[lag >= 0].median() / 60, 1)
    out["speed"]["recv_within_hour"] = r((lag[lag >= 0] <= 60).mean())

    ev["sess"] = (ev.gap.isna() | (ev.gap > 20)).groupby(ev.account).cumsum()
    ss = ev.groupby(["account", "sess"]).agg(n=("ev", "size"), start=("local", "min"), end=("local", "max"))
    ss["mins"] = (ss.end - ss.start).dt.total_seconds() / 60
    binge = ss[ss.n >= 10]
    out["speed"]["binges"] = len(binge)
    out["speed"]["binge_sec_per_like"] = r(60 / (binge.n / binge.mins.clip(lower=1)).median(), 0)
    out["speed"]["binge_median_min"] = r(binge.mins.median(), 0)

    # ---------- when
    acts = pd.concat([liked[["local"]], ni[["local"]], story_likes[["local"]], comments[["local"]], sent[["local"]]])
    acts = acts[acts.local.dt.year.isin(YEARS)]
    acts["h"], acts["dow"], acts["y"] = acts.local.dt.hour, acts.local.dt.dayofweek, acts.local.dt.year
    recent = acts[acts.y >= 2025]
    hm = pd.crosstab(recent.dow, recent.h).reindex(index=range(7), columns=range(24), fill_value=0)
    out["when"] = {
        "events": len(acts),
        "heat": hm.values.tolist(),
        "heat_n": len(recent),
        "hour_share": (acts.h.value_counts(normalize=True).sort_index().reindex(range(24), fill_value=0)).round(4).tolist(),
        "peak_hour_by_year": acts.groupby("y").h.agg(lambda h: int(h.value_counts().idxmax())).to_dict(),
        "late_by_year": acts.groupby("y").h.apply(lambda h: ((h >= 22) | (h < 3)).mean()).round(3).to_dict(),
        "school_by_year": acts[acts.dow < 5].groupby("y").h.apply(lambda h: ((h >= 9) & (h < 15)).mean()).round(3).to_dict(),
    }
    daily = acts.groupby(acts.local.dt.normalize()).size()
    out["when"]["per_day_by_month"] = daily.groupby(daily.index.month).mean().round(1).tolist()
    out["when"]["top_days"] = [{"date": str(d.date()), "dow": d.day_name(), "n": int(n)} for d, n in daily.sort_values(ascending=False).head(5).items()]

    # ---------- growth
    out["growth"] = {"likes_by_year": liked[liked.local.dt.year.isin(YEARS)].groupby([liked.local.dt.year, "account"]).size().unstack(fill_value=0).to_dict("index")}
    known = reels[(reels.topic != "Unknown") & reels.local.dt.year.isin(YEARS)]
    yt = pd.crosstab(known.topic, known.local.dt.year, normalize="columns")
    swing = (yt.max(1) - yt.min(1)).sort_values(ascending=False)
    out["growth"]["topic_years"] = {t: yt.loc[t].round(4).tolist() for t in swing.index[:10]}
    out["growth"]["n_by_year"] = known.groupby(known.local.dt.year).size().reindex(YEARS).tolist()
    lift = yt.div(yt.mean(1), axis=0)
    out["growth"]["signature"] = {int(y): [t for t in lift[y].sort_values(ascending=False).index[:3]] for y in YEARS}

    c = comments.copy()
    c["comment"] = c.comment.fillna("")
    c["y"] = c.local.dt.year
    c = c[c.y.isin(YEARS)]
    c["tags"] = c.comment.str.contains(r"@[\w.]+")
    out["growth"]["comments_by_year"] = c.groupby("y").size().reindex(YEARS, fill_value=0).tolist()
    out["growth"]["comment_tag_share"] = c.groupby("y").tags.mean().reindex(YEARS).round(3).tolist()
    low = c.comment.str.lower()
    c["caps"] = c.comment.map(lambda s: sum(ch.isupper() for ch in s) / max(1, sum(ch.isalpha() for ch in s)) > .6)
    c["question"] = low.str.contains(r"\?")
    c["shop_q"] = low.str.contains(
        r"where(?:’|')?s? (?:is )?(?:your|the|this|that)? ?(?:dress|outfit|top|skirt|bag|set|fit)|which (?:size|color|shade|brand|concealer|sunscreen|foundation|lip|corrector|model)"
        r"|what size|outfit link|link (?:please|pls)|drop the (?:link|location)|which .* do you use|brands? do you recommend")
    for k in ("caps", "question", "shop_q"):
        out["growth"][f"comment_{k}"] = c.groupby("y")[k].mean().reindex(YEARS).round(3).tolist()

    # ---------- profile tabs: saved collections, reposts, how Meta tags me
    import re
    from bs4 import BeautifulSoup
    from classify import OFF_LIMITS, rule_label
    from parse import DATE, RAW, parse_file, parse_time
    # explicit allowlist of collections that are fine to share
    SAVED_OK = {"Food", "Shoes", "Pop", "Ariana grande", "Workplace", "To make with friend", "To try Delhi", "To try Rajasthan",
                "To try London", "To try Dallas", "To try Austin", "Poses", "Story ideas", "Tutorials", "To make",
                "Dallas stairs photo idea", "Self help videos", "Touched", "Relatable", "Quotes"}
    cols = {}
    for account in ("main", "spam"):
        for row in BeautifulSoup((RAW / account / "your_instagram_activity/saved/saved_collections.html").read_text(), "lxml").select("main > div"):
            t = row.get_text(" | ", strip=True)
            name, when = re.search(r"Name \| ([^|]+) \|", t), DATE.findall(t)
            if name and "Type" in t and name.group(1).strip() in SAVED_OK:
                cols[name.group(1).strip()] = {"name": name.group(1).strip(), "account": account, "date": str(parse_time(when[0]).date()) if when else None}
    out["saved_collections"] = sorted(cols.values(), key=lambda c: c["date"] or "", reverse=True)

    # what's inside my saves: the export doesn't record which collection a save is in,
    # so every saved post is sorted into the groups my collections are named after
    sv = saved.copy()
    sv["c"] = (sv.caption.fillna("") + " " + sv.hashtags.map(lambda h: " ".join("#" + x for x in h) if h is not None else "")).str.lower()
    sv = sv[~sv.c.str.contains(OFF_LIMITS)]
    SAVE_GROUPS = {
        "tutorials": {"photo poses & editing": r"pose|posing|photo ?(?:idea|tip|hack)|lightroom|preset|edit", "outfits & styling": r"outfit|styl|how to wear|get the look|lookbook|fit check",
                      "DIY & drawing": r"diy|craft|crochet|knit|paint|draw", "makeup & skincare": r"makeup|skin ?care|spf|serum|lip|blush|contour", "hair": r"hair|curl|braid|blowout"},
        "self-help": {"book lists": r"book|read(?:ing)?\b|novel", "building something & money": r"entrepreneur|business|hustle|ceo|money|billionaire|success",
                      "healing & love": r"heal|love does not|deserve|let go|breakup", "bullet journaling": r"journal|bujo|bullet ?journal",
                      "habits & discipline": r"atomic habits|habit|disciplin|routine|productiv", "stoicism & calm": r"stoic|calm|peace|storm in your mind|anxiety|overthink"},
        "food": {"cafés & coffee spots": r"cafe|café|coffee shop|coffeeshop", "Indian food": r"indian|chaat|paneer|dal\b|roti|biryani|kulfi|street food",
                 "restaurants & bars": r"restaurant|dinner|cocktail|bar\b|rooftop|speakeasy", "desserts": r"dessert|cake|cookie|brownie|bak(?:e|ing)|croissant|tiramisu",
                 "pasta & Italian": r"pasta|pizza|italian|burrata", "brunch": r"brunch|breakfast|pancake|french toast|waffle"},
        "places to try": {"Delhi": r"delhi|gurgaon|gurugram", "Dallas": r"dallas|dfw", "London": r"london", "Paris & Europe": r"paris|europe|italy|rome",
                          "Rajasthan": r"rajasthan|udaipur|jaipur", "NYC": r"\bnyc\b|new york", "Austin": r"austin|\batx\b"},
        "fashion": {"dresses": r"dress", "vintage & thrift": r"vintage|thrift|1999|y2k", "desi wear": r"lehenga|saree|sari|kurta|ethnic",
                    "bags": r"\bbag|handbag|purse|tote", "shoes": r"boot|heel|sneaker|loafer|ballet flat|mary jane|shoe"},
        "feelings": {"poetry & spoken word": r"poem|poetry|spoken word|poet", "love quotes": r"love|loved", "relatable": r"relat|literally me|same"},
    }
    out["saved_groups"] = {g: sorted([[k, int(sv.c.str.contains(p).sum())] for k, p in sub.items()], key=lambda x: -x[1])
                           for g, sub in SAVE_GROUPS.items()}
    out["saved_total"] = len(sv)

    rp = pd.concat([pd.DataFrame(parse_file(RAW / a / "your_instagram_activity/media/reposts.html")).assign(account=a) for a in ("main", "spam")])
    rp = localize(rp)
    cap = rp.caption.fillna("").str.lower()
    rp = rp[~cap.str.contains(OFF_LIMITS)].copy()
    rp = rp.merge(media[["shortcode", "topic"]].rename(columns={"topic": "t0"}), on="shortcode", how="left")
    rp["topic"] = rp.t0.where(rp.t0.notna() & (rp.t0 != "Unknown"), rp.caption.fillna("").str.lower().map(rule_label))
    rshare = rp.topic.value_counts(normalize=True)
    lshare = shares(reels)
    rlift = (rshare / lshare).dropna()
    rlift = rlift[rshare.reindex(rlift.index) >= .03].sort_values(ascending=False)
    by_year = rp.groupby([rp.local.dt.year, "account"]).size().unstack(fill_value=0)
    month = rp.groupby(rp.local.dt.to_period("M")).size()
    x = rp.sort_values("local")
    x["sess"] = (x.local.diff().dt.total_seconds().fillna(1e9) > 1800).cumsum()
    sess = x.groupby("sess").agg(n=("shortcode", "size"), start=("local", "min"))
    biggest = sess.sort_values("n").iloc[-1]
    q_saves = saved.groupby(saved.local.dt.to_period("Q")).size()
    q_reposts = rp.groupby(rp.local.dt.to_period("Q")).size()
    last_full = q_reposts.index.sort_values()[-2]
    out["reposts_pattern"] = {
        "by_account": rp.account.value_counts().to_dict(),
        "liked_first": r(rp.shortcode.isin(liked.shortcode).mean()), "saved_too": r(rp.shortcode.isin(saved.shortcode).mean()),
        "sent_too": r(rp.shortcode.isin(sent.shortcode).mean()),
        "q_label": str(last_full).replace("Q", " Q"), "q_reposts": int(q_reposts.get(last_full, 0)), "q_saves": int(q_saves.get(last_full, 0)),
        "biggest_session": int(biggest.n), "biggest_session_date": str(biggest.start.date()),
        "in_bursts": r(sess[sess.n >= 10].n.sum() / len(x)),
        "peak_hour": int(x.local.dt.hour.value_counts().idxmax()), "peak_hour_share": r(x.local.dt.hour.value_counts(normalize=True).max()),
        "like_peak_hour": int(liked.local.dt.hour.value_counts().idxmax()),
    }
    out["reposts"] = {"n": len(rp), "by_year": {int(y): v for y, v in by_year.to_dict("index").items()},
                      "top_month": str(month.idxmax()), "top_month_n": int(month.max()),
                      "lift": [{"topic": t, "lift": r(v, 1)} for t, v in rlift.head(5).items()]}

    # most-loved reels: hand-picked from the top of an engagement score
    # (like 1, save 2, repost 3, +1 per friend it was sent to, max 5), skipping anything off-limits
    # my picks: posts I liked and reposted
    BELOVED = ["DbzPAEitJXe", "DZ8CVJLSNu9", "DUt_X1Tj6rU", "DZM4aifxlSm", "DZ5Zhb7McL8", "DZFDHDgjImC",
               "DdY-C1IPbRe", "Db8chVqM5ic", "DdAnZggAKWc"]
    sent_to = sent.groupby("shortcode").thread.nunique()

    def first_line(text):
        """First sentence only, so long captions never drag off-limits asides onto the page."""
        t = re.sub(r"#\S+|@\S+|📍\S*|follow\s*_*", "", text or "").split("\n")[0]
        t = re.split(r"(?<=[.!?])\s", t.strip())[0]
        return re.sub(r"\s+", " ", t).strip()[:90]
    caps = pd.read_parquet(DATA / "media.parquet").set_index("shortcode")
    reposted = set(pd.concat([pd.DataFrame(parse_file(RAW / a / "your_instagram_activity/media/reposts.html")) for a in ("main", "spam")]).shortcode)
    out["beloved"] = [{"code": c, "owner": caps.owner.get(c), "caption": first_line(caps.caption.get(c)), "kind": caps.kind.get(c),
                       "friends": int(sent_to.get(c, 0)), "liked": c in set(liked.shortcode), "saved": c in set(saved.shortcode), "reposted": c in reposted}
                      for c in BELOVED]

    tags = {}
    for account in ("main", "spam"):
        for row in BeautifulSoup((RAW / account / "your_instagram_activity/ai/interest_categories.html").read_text(), "lxml").select("main > div"):
            m = re.search(r"interested in ([^|]+?) \|", row.get_text(" | ", strip=True))
            if m and not OFF_LIMITS.search(m.group(1)):
                tags[m.group(1).strip()] = account
    ad = BeautifulSoup((RAW / "main/ads_information/instagram_ads_and_businesses/other_categories_used_to_reach_you.html").read_text(), "lxml").select_one("main > div")
    keep = ["Away from family", "Away from hometown", "Birthday in March", "Engaged Shoppers", "Frequent international travelers",
            "Lived in India (Formerly Expats - India)", "Lives abroad", "Small business owners", "Relationship status: single", "Food and Restaurants"]
    ad_text = ad.get_text(" | ", strip=True) if ad else ""
    liked_all = load("liked")[["shortcode"]].merge(media[["shortcode"]].merge(pd.read_parquet(DATA / "media.parquet")[["shortcode", "caption"]], on="shortcode"), on="shortcode", how="left").caption.fillna("").str.lower()
    out["tagged"] = {"interests": [{"tag": k, "account": v} for k, v in tags.items()],
                     "ad_categories": [k for k in keep if k in ad_text],
                     "lacrosse_likes": int(liked_all.str.contains(r"lacrosse").sum())}

    # ---------- the spam account's 2025 self-upgrade season
    lk = load("liked")[["shortcode", "account", "local"]].merge(pd.read_parquet(DATA / "media.parquet")[["shortcode", "caption", "off_limits"]], on="shortcode", how="left")
    lk = lk[~lk.off_limits.fillna(False).astype(bool)]
    lk["c"] = lk.caption.fillna("").str.lower()
    q = lk[lk.local >= "2021-01-01"].groupby([lk.local.dt.to_period("Q"), "account"]).size().unstack(fill_value=0)
    out["spam_season"] = {"quarters": [str(p).replace("Q", " Q") for p in q.index[:-1]], "share": (q.spam / q.sum(1)).round(3).tolist()[:-1]}
    themes = {"upgrade & glow-up": r"upgrade|level ?up|glow ?up|transform", "money mindset": r"money ?mindset|moneymindset|financial freedom",
              "divine feminine": r"divine ?feminine|dark ?feminine|feminine energy", "high value": r"high ?value", "discipline": r"disciplin",
              "meditation": r"meditat|mindful", "regret": r"regret", "Pisces": r"pisces|horoscope|zodiac|astrolog"}
    sp, mn = lk[lk.account == "spam"], lk[lk.account == "main"]
    out["spam_season"]["themes"] = sorted([{"theme": k, "lift": r(sp.c.str.contains(p).mean() / mn.c.str.contains(p).mean(), 1)} for k, p in themes.items()], key=lambda x: -x["lift"])
    up = sp[sp.c.str.contains(themes["upgrade & glow-up"])]
    out["spam_season"]["upgrade_peak"] = str(up.groupby(up.local.dt.to_period("Q")).size().idxmax()).replace("Q", " Q")

    # ---------- kindness: what I say to people, and what I like
    say = comments.comment.fillna("")
    KIND = {"thanks": r"(?i)thank|grateful|appreciate", "compliments": r"(?i)beautiful|gorgeous|stunning|pretty|cute|slay|queen|glowing|elegant",
            "birthdays": r"(?i)happy (?:birthday|bday|b-day)|hbd", "congrats": r"(?i)congrat|congrad|proud of|so happy for",
            "love": r"(?i)love (?:you|u)\b|miss (?:you|u)\b|\bily\b|\bimy\b"}
    caps = pd.read_parquet(DATA / "media.parquet")[["shortcode", "caption"]]
    liked_caps = load("liked")[["shortcode"]].merge(caps, on="shortcode", how="left").caption.fillna("").str.lower()
    out["kind"] = {k: int(say.str.contains(p).sum()) for k, p in KIND.items()}
    out["kind"]["liked_gratitude"] = int(liked_caps.str.contains(r"grateful|gratitude|thankful|blessed").sum())
    out["kind"]["liked_parents"] = int(liked_caps.str.contains(r"my (?:mom|dad|parents)|mother.?s day|father.?s day|thank you (?:mom|dad|mumma|papa)").sum())
    out["kind"]["story_likes_by_year"] = story_likes.groupby(story_likes.local.dt.year).size().to_dict()
    out["kind"]["digs"] = int(say.str.contains(r"(?i)\bugly\b|\bstupid\b|\bdumb\b|\bidiot|\bloser|\bshut up\b|\bcringe\b|\bgross\b|\bpathetic\b|\bfake\b").sum())

    # ---------- this week: the only real watch log Instagram keeps
    week = pd.concat([load("watched").assign(src="watched"), load("viewed").assign(src="viewed")])
    week = week[week.local >= "2026-09-26"]
    groups = {
        "planning Austin weekends": ["whenwherewhataustin", "365thingsaustin", "austin.bucketlist", "theflock_atx", "thelobby.atx",
                                     "balletaustinctr", "inviteonlyatx", "lunarooftop.atx", "mozartscoffee"],
        "window-shopping dresses": ["loveshackfancy", "peppermayo", "forloveandlemons", "eberjey", "buckmasonwomens", "coach",
                                    "elizabethscarlett", "shantisarna", "enidsullins"],
        "in a book club": ["primebookclub", "poetsandquotes_", "poemsporn_", "thelovehypothesismovie"],
        "deciding what to stream": ["netflix_in", "primevideo"],
        "buying stationery I don't need": ["riflepaperco", "erincondren", "papier"],
        "reading campus news & finance gossip": ["thedailytexan", "wallstreetoasis"],
        "planning workouts": ["dndactive.co", "workit.app", "setactive"],
    }
    vc = week.owner.value_counts()
    out["week"] = {
        "n": len(week), "start": str(week.local.min().date()), "end": str(week.local.max().date()),
        "late": r(((week.local.dt.hour >= 1) & (week.local.dt.hour < 5)).mean()),
        "busiest_day": week.groupby(week.local.dt.day_name()).size().idxmax(),
        "groups": sorted([{"name": g, "n": int(vc.reindex(acc).fillna(0).sum()),
                           "accounts": [a for a in acc if vc.get(a, 0) > 0][:4]} for g, acc in groups.items()], key=lambda x: -x["n"]),
        "hours": week.groupby(week.local.dt.hour).size().reindex(range(24), fill_value=0).tolist(),
    }
    watched = load("watched")
    repeats = watched[watched.local >= "2026-09-26"].groupby("owner").shortcode.agg(lambda s: s.value_counts().iloc[0])
    out["week"]["coach_repeats"] = int(watched[watched.caption.fillna("").str.contains("Tabby Bag")].shape[0])
    out["week"]["top_repeats"] = repeats.sort_values(ascending=False).head(6).to_dict()

    # ---------- pipeline / method
    L = load("liked").dropna(subset=["shortcode"])
    raw_gap = (L.time - L.posted_utc).dt.total_seconds() / 3600
    floor = raw_gap.groupby(L.time.dt.to_period("M")).quantile(0.005)
    floor = floor[(floor.index >= pd.Period("2021-01", "M")) & (floor.index <= pd.Period("2026-09", "M"))]
    m = pd.read_parquet(DATA / "media.parquet")
    out["method"] = {
        "tz_floor": [{"m": str(p), "h": r(v, 2)} for p, v in floor.items()],
        "media": len(m),
        "label_steps": {"rules": r((m.topic_source == "rules").mean()),
                        "model": r((m.topic_source == "model").mean()),
                        "creator": r((m.topic_source == "creator").mean())},
        "html_files": sum(1 for _ in (Path.home() / "Desktop/Instagram Data/raw").rglob("*.html")),
    }

    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site/data.js").write_text("window.DATA = " + json.dumps(out, default=str, ensure_ascii=False) + ";\n")
    print(json.dumps({k: v for k, v in out.items() if k != "when"}, default=str, indent=1)[:6000])


if __name__ == "__main__":
    main()
