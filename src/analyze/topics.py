"""Sentiment filtering, topic clustering and ranking."""

import re
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Words that dominate topic labels without adding meaning
GENERIC_WORDS = {
    "game", "games", "like", "just", "really", "play", "playing", "played", "time",
    "good", "bad", "don", "doesn", "didn", "isn", "ve", "ll", "im", "thing", "things",
    "lot", "way", "make", "feel", "feels", "want",
}


def load_reviews(db_path: Path, appid: int) -> pd.DataFrame:
    """All stored reviews for a game, positive and negative."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM reviews WHERE appid = ?", conn, params=(appid,))


def add_sentiment(sentences: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Tag each sentence POSITIVE/NEGATIVE. Only reviews not seen before are processed."""
    cached = pd.read_pickle(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty:
        cached = cached[cached["review_id"].isin(sentences["review_id"])]
        todo = sentences[~sentences["review_id"].isin(cached["review_id"])]
    else:
        todo = sentences

    if len(todo):
        from tqdm import tqdm
        from transformers import pipeline

        print(f"  running sentiment on {len(todo)} new sentences...")
        clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True)
        texts = todo["sentence"].tolist()
        preds = []
        for i in tqdm(range(0, len(texts), 1000), desc="  sentiment"):
            preds += clf(texts[i:i + 1000], batch_size=64)
        todo = todo.assign(
            sentiment=[p["label"] for p in preds],
            confidence=[p["score"] for p in preds],
        )

    result = pd.concat([cached, todo], ignore_index=True)[
        ["review_id", "sentence", "sentiment", "confidence"]
    ]
    # Review metadata (votes, playtime...) always comes from the fresh data, so it stays current
    # and caches created by older versions of this code keep working.
    meta_cols = [c for c in sentences.columns if c != "sentence"]
    result = result.merge(sentences[meta_cols].drop_duplicates("review_id"), on="review_id", how="left")
    result.to_pickle(cache_path)
    return result


def build_stop_words(game_name: str) -> list[str]:
    """Generic words + the game's own name (e.g. "Baldur's Gate 3" -> baldurs, gate, bg3)."""
    tokens = re.findall(r"[a-z0-9]+", game_name.lower().replace("'", ""))
    words = {t for t in tokens if len(t) > 1}
    initials = "".join(t[0] for t in tokens if t.isalpha())
    digits = "".join(t for t in tokens if t.isdigit())
    if len(initials) > 1:
        words |= {initials, initials + digits}
    words |= {w.rstrip("s") for w in words}  # "baldurs" -> "baldur"
    return list(ENGLISH_STOP_WORDS | GENERIC_WORDS | words)


def auto_min_topic_size(n_sentences: int) -> int:
    """Scale cluster size with the data: ~8 for small games, up to 40 for huge ones."""
    return max(8, min(40, n_sentences // 150))


def cluster_complaints(docs: list[str], stop_words: list[str],
                       min_topic_size: int = 40, outlier_threshold: float = 0.4, seed: int = 42):
    """Cluster sentences into topics, auto-merge similar topics, reassign outliers."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True)

    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2),
                                 min_df=3 if len(docs) >= 1000 else 1)
    model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=seed),
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        nr_topics="auto",  # merges near-duplicate topics automatically
    )
    topics, _ = model.fit_transform(docs, embeddings)

    if -1 in topics and len(set(topics)) > 1:
        topics = model.reduce_outliers(docs, topics, strategy="embeddings",
                                       embeddings=embeddings, threshold=outlier_threshold)
        model.update_topics(docs, topics=topics, vectorizer_model=vectorizer)

    return model, list(topics), embeddings, embedder


def rank_topics(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pain point, counted per review (one long rant can't inflate a topic).
    share_of_complaining_reviews: of all reviews containing any complaint, how many mention this one.
    """
    n_complaining = complaints["review_id"].nunique()
    per_review = complaints[complaints["topic"] != -1].drop_duplicates(["topic", "review_id"])
    ranking = per_review.groupby("topic").agg(
        reviews=("review_id", "size"),
        in_positive_reviews=("voted_up", "sum"),
        total_upvotes=("votes_up", "sum"),
        median_hours=("hours_at_review", "median"),
    )
    ranking["share_of_complaining_reviews"] = (ranking["reviews"] / n_complaining).round(3)
    ranking["median_hours"] = ranking["median_hours"].round(1)

    # Most upvoted sentence per topic as a representative quote
    top = per_review.sort_values("votes_up", ascending=False).drop_duplicates("topic")
    ranking["example"] = top.set_index("topic")["sentence"]
    return ranking.sort_values("reviews", ascending=False)


def monthly_timeline(complaints: pd.DataFrame) -> pd.DataFrame:
    """Number of reviews mentioning each topic per month (long format)."""
    df = complaints[complaints["topic"] != -1].copy()
    df["month"] = pd.to_datetime(df["created_at"], unit="s").dt.to_period("M").dt.to_timestamp()
    return (df.groupby(["month", "topic"])["review_id"].nunique()
              .rename("reviews").reset_index())
