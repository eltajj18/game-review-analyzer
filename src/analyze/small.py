"""
Pain-point detection for games with few complaints, where clustering has too little data.

- llm_group: Claude reads every complaint sentence and groups them into pain points.
- classifier_group: free and offline; a classifier trained on labeled review sentences
  (see src/train/) predicts one of a fixed set of aspects per sentence.
- template_group: last-resort fallback; each sentence is matched to the most similar
  common pain-point description (e.g. "the camera is hard to control").

Both return the same thing clustering does: a topic per sentence (-1 = none)
plus a {topic: (name, category, method)} label dict.
"""

import json
import re

import numpy as np
import pandas as pd

from src.analyze.aspects import ASPECTS, MODEL_PATH
from src.analyze.label import CATEGORIES, LLM_MODEL
from src.analyze.topics import EMBEDDING_MODEL


def llm_group(sentences: list[str], game_name: str) -> tuple[list[int], dict]:
    import anthropic

    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    prompt = f"""Below are complaint sentences from Steam reviews of the game "{game_name}".
Group them into distinct pain points a developer could act on.

Rules:
- Each pain point gets a short, specific "name" (2-5 words, e.g. "Controller aiming feels floaty")
  and a "category": exactly one of {list(CATEGORIES)}.
- Use "Not actionable" for general dislike, genre preference or off-topic remarks.
- "sentence_ids" lists the sentences that express that pain point. Each sentence belongs to at
  most one pain point. Leave out sentences that aren't really complaints.
- Merge sentences about the same underlying problem; don't create near-duplicate pain points.

Sentences:
{numbered}

Respond with ONLY a JSON list, no other text:
[{{"name": "...", "category": "...", "sentence_ids": [0, 4, 7]}}, ...]"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    groups = json.loads(re.sub(r"```(?:json)?", "", text).strip())

    topics = [-1] * len(sentences)
    labels = {}
    for group in groups:
        if group.get("category") not in CATEGORIES:
            continue
        topic_id = len(labels)
        ids = [i for i in group.get("sentence_ids", [])
               if isinstance(i, int) and 0 <= i < len(sentences) and topics[i] == -1]
        if not ids:
            continue
        for i in ids:
            topics[i] = topic_id
        labels[topic_id] = (group["name"], group["category"], "llm")
    return topics, labels


def template_group(sentences: list[str], threshold: float = 0.35) -> tuple[list[int], dict]:
    from sentence_transformers import SentenceTransformer

    phrases, owners = [], []
    for category, descs in CATEGORIES.items():
        phrases += descs
        owners += [category] * len(descs)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    sims = (embedder.encode(sentences, normalize_embeddings=True)
            @ embedder.encode(phrases, normalize_embeddings=True).T)
    best = sims.argmax(axis=1)
    topics = np.where(sims.max(axis=1) >= threshold, best, -1).tolist()

    labels = {int(t): (phrases[t][0].upper() + phrases[t][1:], owners[t], "templates")
              for t in set(topics) if t != -1}
    return topics, labels


def classifier_group(sentences: list[str], min_confidence: float = 0.4) -> tuple[list[int], dict]:
    import joblib
    from sentence_transformers import SentenceTransformer

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    embeddings = SentenceTransformer(bundle["embedding_model"]).encode(
        sentences, normalize_embeddings=True)
    probs = model.predict_proba(embeddings)

    topic_of = {aspect: i for i, aspect in enumerate(ASPECTS)}  # stable ids across runs
    topics, labels = [], {}
    for p in probs:
        aspect = model.classes_[p.argmax()]
        if p.max() < min_confidence or aspect not in topic_of:
            topics.append(-1)
            continue
        topic = topic_of[aspect]
        topics.append(topic)
        name, category, _ = ASPECTS[aspect]
        labels[topic] = (name, category, "classifier")
    return topics, labels


def labels_to_frame(labels: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        labels, orient="index", columns=["name", "category", "label_method"]
    ).rename_axis("topic")
