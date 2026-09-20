"""
Automatic topic naming and categorization.

Two strategies:
- LLM (if ANTHROPIC_API_KEY is set): Claude reads each topic's keywords and example
  sentences and returns a short name + category. Best quality, costs a few cents.
- Embeddings (free fallback): the category whose description is most similar to the
  topic's sentences wins; the name is built from the topic's top keywords.
"""

import json
import os
import re

import numpy as np
import pandas as pd

LLM_MODEL = "claude-haiku-4-5-20251001"

# Several phrasings per category; a topic gets the category of its best-matching phrase.
CATEGORIES = {
    "Technical": [
        "the game has bugs and glitches",
        "the game crashes and freezes",
        "poor performance, low frame rate and stuttering",
        "save files get corrupted or lost",
    ],
    "Gameplay": [
        "the combat is slow and tedious",
        "the dice rolls and random chance feel unfair",
        "game mechanics, rules and balance are frustrating",
        "the difficulty is badly tuned",
    ],
    "Content": [
        "the story and writing are weak",
        "the ending and final act are disappointing",
        "the characters and companions are unlikeable",
        "the quests and dialogue are poorly written",
    ],
    "UX": [
        "the controls are awkward and unintuitive",
        "the camera is hard to control",
        "the user interface, menus and inventory are clunky",
        "the tutorial does not explain the mechanics",
    ],
    "Business": [
        "the price is too high and not worth the money",
        "I want a refund",
        "problems with the store, platform or DLC",
    ],
    "Not actionable": [
        "I just did not enjoy it and it was not fun",
        "this genre is not for me, personal preference",
        "I played for many hours",
        "complaints about the developer company, not the game itself",
    ],
}


def _topic_summaries(model, complaints: pd.DataFrame, n_examples: int = 8) -> list[dict]:
    summaries = []
    for topic in sorted(t for t in complaints["topic"].unique() if t != -1):
        sents = complaints.loc[complaints["topic"] == topic, "sentence"]
        summaries.append({
            "topic": int(topic),
            "keywords": [w for w, _ in model.get_topic(topic)[:10]],
            "examples": sents.sample(min(n_examples, len(sents)), random_state=0).tolist(),
        })
    return summaries


def _similar(a: str, b: str) -> bool:
    """Rough check for word variants: act/acts, bug/buggy, crash/crashes."""
    a, b = a.rstrip("s"), b.rstrip("s")
    if len(a) >= 5 and len(b) >= 5:
        return a[:5] == b[:5]
    return a.startswith(b) or b.startswith(a)


def keyword_name(keywords: list[str], n_words: int = 3) -> str:
    chosen: list[str] = []
    for kw in keywords:
        parts = kw.split()
        if any(_similar(p, c) for p in parts for cw in chosen for c in cw.split()):
            continue
        chosen.append(kw)
        if len(chosen) == n_words:
            break
    return " / ".join(chosen).capitalize() if chosen else "Unnamed"


def embedding_labels(summaries, complaints, embeddings, embedder) -> dict:
    phrases, owners = [], []
    for category, descs in CATEGORIES.items():
        phrases += descs
        owners += [category] * len(descs)
    phrase_vecs = embedder.encode(phrases, normalize_embeddings=True)

    topics = complaints["topic"].to_numpy()
    labels = {}
    for s in summaries:
        centroid = embeddings[topics == s["topic"]].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        best = int(np.argmax(phrase_vecs @ centroid))
        labels[s["topic"]] = (keyword_name(s["keywords"]), owners[best], "embeddings")
    return labels


def llm_labels(summaries: list[dict], game_name: str) -> dict:
    import anthropic

    prompt = f"""You are analyzing clusters of complaint sentences from negative Steam reviews of "{game_name}".

For each topic below, give:
- "name": a short, specific, human-readable name for the complaint (2-5 words, e.g. "Act 3 feels rushed")
- "category": exactly one of {list(CATEGORIES)}

Use "Not actionable" when the topic is not a specific, fixable problem with the game
(general dislike, genre preference, playtime statements, off-topic events or controversies).

Topics:
{json.dumps(summaries, indent=1)}

Respond with ONLY a JSON list, no other text:
[{{"topic": 0, "name": "...", "category": "..."}}, ...]"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = re.sub(r"```(?:json)?", "", text).strip()

    return {
        int(item["topic"]): (item["name"], item["category"], "llm")
        for item in json.loads(text)
        if item.get("category") in CATEGORIES
    }


def label_topics(model, complaints, embeddings, embedder, game_name: str,
                 use_llm: bool = True) -> pd.DataFrame:
    summaries = _topic_summaries(model, complaints)
    labels = embedding_labels(summaries, complaints, embeddings, embedder)

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            labels.update(llm_labels(summaries, game_name))  # LLM wins where it answered
            print("  topics labeled with Claude")
        except Exception as e:
            print(f"  LLM labeling failed ({e}); using embedding-based labels")
    else:
        print("  topics labeled with embeddings (set ANTHROPIC_API_KEY for better names)")

    return pd.DataFrame.from_dict(
        labels, orient="index", columns=["name", "category", "label_method"]
    ).rename_axis("topic")
