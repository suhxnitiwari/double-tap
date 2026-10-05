"""Label every reel/post with one topic, in three passes.

1. Rules: a keyword/hashtag dictionary labels captions that say what they are.
2. Model: logistic regression on caption embeddings, trained on the rule labels,
   labels the captions rules missed (kept only when confident).
3. Creator: anything still unlabeled (emoji-only, no caption) inherits its
   creator's majority topic when that creator has enough labeled posts.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"

TOPICS = {
    "Dating & relationships": r"dating|situationship|boyfriend|girlfriend|\bbf\b|\bgf\b|couple\w*|relationship\w*|crush|"
                              r"red ?flag|green ?flag|ex ?boyfriend|my man|your man|breakup|heartbreak|love ?language|soulmate",
    "Marriage & family life": r"married ?life|marriage\w*|husband|wife ?life|\bwife\b|in-?laws|saas|shaadi ke baad",
    "Motherhood & babies": r"\bmom ?life|mum ?life|motherhood|\bbaby\b|babies|toddler|parenting|boymom|girlmom|newborn|pregnan\w*|\bkids\b",
    "Friendship": r"bestie\w*|best ?friends?|\bbff\b|girls ?night|girls ?trip|tag (a|your) friend|friendship|\bfriends\b",
    "Weddings & bridal": r"bride\w*|bridal|wedding\w*|mehendi|mehndi|sangeet|haldi|lehenga|baraat|engagement",
    "Fashion & outfits": r"outfit\w*|\bootd\w*|fashion\w*|\bstyle\w*|\bdress(es)?\b|grwm|\bwear\b|closet|\bfit check|lookbook|saree|kurta",
    "Makeup & beauty": r"makeup\w*|make-up|skincare|skin ?care|lipstick|\blip\w*|mascara|blush|hair ?care|hairstyle\w*|\bnails?\b|beauty\w*|glam|sephora|serum",
    "Luxury, money & shopping": r"luxury|old ?money|chanel|hermes|birkin|cartier|dior|louis ?vuitton|\bmoney\b|finance|invest\w*|\brich\b|wealth\w*|shopping|shop ?with ?me|haul",
    "Bollywood & desi": r"bollywood|\bdesi\w*|indian|\bindia\b|hindi|punjabi|gujarati|diwali|holi|navratri|garba|bhangra|srk|shah ?rukh|"
                        r"alia ?bhatt|ranbir|kareena|deepika|ranveer|dhurandhar|bhai|yaar|\bbeta\b|mummy|papa ji|chai|aunty|bachpan",
    "TV, film & celebs": r"taylor ?swift|swiftie\w*|gossip ?girl|blair|summer ?i ?turned|tsitp|stranger ?things|bridgerton|vampire ?diaries|"
                         r"\bmovie\w*|netflix|\bseries\b|\bshow\b|\bactress\b|celebrit\w*|ariana|sabrina ?carpenter|oscars|met ?gala|kardashian|love ?island",
    "Books & reading": r"\bbooks?\b|bookstagram|booktok|reading|\bnovel\w*|\bauthor\b|romance ?read\w*|bookworm|colleen ?hoover",
    "Self-growth & mindset": r"mindset|self ?love|self ?growth|healing|manifest\w*|affirmation\w*|motivation\w*|discipline|confidence|"
                             r"personal ?growth|self ?respect|self ?care|glow ?up|\bhabits?\b|therapy|anxiety|mental ?health|journaling|productiv\w*",
    "Women & culture": r"feminis\w*|women ?empower\w*|womensupportingwomen|\bpatriarchy|girl ?boss|women in (?:stem|business)|sisterhood|\bwomen\b",
    "Fitness & wellness": r"\bgym\w*|workout\w*|pilates|fitness|\byoga\b|protein|\babs\b|\bglutes?\b|running|\brun club|\bhealth\w*|wellness|"
                          r"nontoxic|non-toxic|cortisol|hormone\w*|\bpcos\b|weight ?loss|\bdiet\b",
    "Food & coffee": r"\bfood\w*|recipe\w*|\bcoffee\b|matcha|latte|\bcafe\b|brunch|restaurant\w*|\beats\b|baking|\bdessert\w*|\bcook\w*|\bpasta\b|\bpizza\b|snack",
    "Travel & cities": r"\btravel\w*|wanderlust|vacation|\bnyc\b|new ?york|\blondon\b|\bparis\b|\bitaly\b|\beurope\b|\bairport\b|flight|passport|"
                       r"\baustin\b|\batx\b|\bdallas\b|\bhouston\b|\btexas\b|city ?life|beach",
    "College & career": r"college\w*|\buni\b|university|\bstudy\w*|student\w*|exam\w*|finals|midterm\w*|\bsat\b|internship\w*|corporate|"
                        r"\b9 ?to ?5\b|coworker\w*|\boffice\b|\bjob\b|career|recruit\w*|hookem|\butaustin\b|mccombs|\bboss\b|linkedin|resume",
    "Art, music & dance": r"\bart\b|artist\w*|painting|drawing|illustration|\bdance\w*|dancing|choreo\w*|bharatanatyam|kathak|\bsinging|"
                          r"\bsong\b|\bmusic\b|cover|concert|guitar|piano|\bcrochet|\bdiy\b|pottery",
    "Home & aesthetic living": r"home ?decor|\bdecor\b|apartment|interior\w*|room ?tour|\bcleaning\b|organiz\w*|morning ?routine|\bcozy\b|romanticiz\w*|pinterest",
    "Pets & animals": r"\bdog\w*|\bpupp\w*|\bcat\b|\bcats\b|kitten\w*|\bpets?\b|golden ?retriever|bunny|panda\w*|\banimals?\b|kirby",
}
PATTERNS = {t: re.compile(p) for t, p in TOPICS.items()}
# Off-limits content never gets a topic, so it can't reach any chart, quote or tile.
# The term list lives in off_limits.txt next to this file and is kept out of git.
OFF_LIMITS = re.compile((Path(__file__).parent / "off_limits.txt").read_text().strip())


def rule_label(text):
    hits = {t: len(p.findall(text)) for t, p in PATTERNS.items()}
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        return None
    ranked = sorted(hits.values(), reverse=True)
    return best if ranked[0] > ranked[1] else None  # ties stay unlabeled for the model


def main():
    m = pd.read_parquet(DATA / "media.parquet")
    emb = np.load(DATA / "media_emb.npy")
    m["off_limits"] = m.text.str.contains(OFF_LIMITS)
    m["topic"] = m.text.map(rule_label).where(~m.off_limits)
    m["topic_source"] = np.where(m.topic.notna(), "rules", None)
    print("rules:", m.topic.notna().mean().round(3))

    # 2. model on embeddings, trained on rule labels
    has = m.has_text.values
    train = has & m.topic.notna().values
    X, y = emb[train], m.topic[train].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced").fit(Xtr, ytr)
    print("held-out agreement with rules:", round(clf.score(Xte, yte), 3))
    clf.fit(X, y)
    todo = has & m.topic.isna().values & ~m.off_limits.values
    proba = clf.predict_proba(emb[todo])
    conf = proba.max(1) >= 0.6
    idx = np.where(todo)[0][conf]
    m.loc[m.index[idx], "topic"] = clf.classes_[proba.argmax(1)[conf]]
    m.loc[m.index[idx], "topic_source"] = "model"
    print("after model:", m.topic.notna().mean().round(3))

    # 3. creator majority
    lab = m.dropna(subset=["topic"])
    share = lab.groupby("owner").topic.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    major = lab.groupby("owner").topic.agg(lambda s: s.value_counts().index[0])
    n = lab.groupby("owner").size()
    good = major[(n >= 3) & (share >= 0.5)]
    fill = m.topic.isna() & m.owner.isin(good.index) & ~m.off_limits
    m.loc[fill, "topic"] = m.loc[fill, "owner"].map(good)
    m.loc[fill, "topic_source"] = "creator"
    m["topic"] = m.topic.fillna("Unknown")
    print("after creator:", (m.topic != "Unknown").mean().round(3))
    print(m.topic_source.value_counts(dropna=False).to_string())
    m.drop(columns=[]).to_parquet(DATA / "media.parquet")


if __name__ == "__main__":
    main()
