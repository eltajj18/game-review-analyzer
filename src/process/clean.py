"""Text cleaning, language filtering and sentence splitting."""

import html
import re

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0  # langdetect is random otherwise

BBCODE = re.compile(r"\[/?[a-z0-9*]+(?:=[^\]]*)?\]", re.I)  # [b], [h1], [url=...]
URL = re.compile(r"https?://\S+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean(text: str) -> str:
    text = html.unescape(text or "").replace("\xa0", " ")
    text = BBCODE.sub(" ", text)
    text = URL.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)  # keep newlines: they often separate points
    return text.strip()


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def split_sentences(text: str, min_words: int = 5, max_words: int = 60) -> list[str]:
    out = []
    for s in SENTENCE_SPLIT.split(text):
        s = s.strip(" -*•\t")
        if min_words <= len(s.split()) <= max_words:
            out.append(s)
    return out


def prepare_sentences(reviews: pd.DataFrame, min_chars: int = 50) -> pd.DataFrame:
    """Clean reviews, keep English ones, and return one row per sentence."""
    df = reviews.copy()
    df["clean"] = df["text"].map(clean)
    df = df[df["clean"].str.len() >= min_chars]
    df = df[df["clean"].map(is_english)]
    df["hours_at_review"] = df["playtime_at_review"].fillna(0) / 60

    return (
        df[["review_id", "voted_up", "votes_up", "hours_at_review", "created_at"]]
        .assign(sentence=df["clean"].map(split_sentences))
        .explode("sentence")
        .dropna(subset=["sentence"])
        .reset_index(drop=True)
    )
