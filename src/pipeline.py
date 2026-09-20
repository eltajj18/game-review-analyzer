"""
Game Review Pain Point Analyzer: full pipeline in one command.

    python -m src.pipeline --game "Baldur's Gate 3"
    python -m src.pipeline --appid 1086940

Steps: collect reviews -> clean & split into sentences -> find complaint sentences
-> group them into pain points -> rank -> save.

The grouping method adapts to how much feedback a game has:
- many complaints (big games): topic clustering with BERTopic
- few complaints + ANTHROPIC_API_KEY: Claude groups them directly
- few complaints, no key: trained aspect classifier (if models/ has one),
  otherwise match against common pain-point descriptions
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # optional: read ANTHROPIC_API_KEY from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

from src.analyze.label import label_topics
from src.analyze.aspects import MODEL_PATH
from src.analyze.small import (classifier_group, drop_non_complaints, labels_to_frame,
                               llm_group, template_group)
from src.analyze.topics import (
    add_sentiment, auto_min_topic_size, build_stop_words, cluster_complaints,
    load_reviews, monthly_timeline, rank_topics,
)
from src.collect.steam import DB_PATH, collect, get_game_name, resolve_game
from src.process.clean import prepare_sentences

DATA_DIR = Path("data")
CLUSTERING_THRESHOLD = 300  # complaint sentences needed for clustering to work well
MIN_COMPLAINTS = 3


def find_pain_points(complaints: pd.DataFrame, game: str, min_topic_size: int | None,
                     use_llm: bool) -> tuple[list[int], pd.DataFrame, str]:
    """Returns (topic per sentence, labels frame, method used)."""
    docs = complaints["sentence"].tolist()
    has_llm = use_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))

    if len(docs) >= CLUSTERING_THRESHOLD:
        size = min_topic_size or auto_min_topic_size(len(docs))
        print(f"  {len(docs)} complaints: clustering (min_topic_size={size})")
        model, topics, embeddings, embedder = cluster_complaints(docs, build_stop_words(game), size)
        labeled = complaints.assign(topic=topics)
        labels = label_topics(model, labeled, embeddings, embedder, game, use_llm)
        return topics, labels, "clustering"

    if has_llm:
        print(f"  only {len(docs)} complaints: letting Claude group them")
        try:
            topics, labels = llm_group(docs, game)
            return topics, labels_to_frame(labels), "llm"
        except Exception as e:
            print(f"  Claude grouping failed ({e}); falling back to templates")

    if MODEL_PATH.exists():
        print(f"  only {len(docs)} complaints: classifying with the trained aspect model")
        topics, labels = classifier_group(docs)
        return topics, labels_to_frame(labels), "classifier"

    print(f"  only {len(docs)} complaints: matching against common pain points"
          + ("" if has_llm else " (set ANTHROPIC_API_KEY for better results)"))
    topics, labels = template_group(docs)
    return topics, labels_to_frame(labels), "templates"


def run(appid: int, skip_collect: bool = False, max_negative: int | None = None,
        max_positive: int | None = 5000, language: str = "english",
        min_topic_size: int | None = None, min_confidence: float = 0.98,
        use_llm: bool = True, game: str | None = None,
        classifier_filter: bool = True) -> pd.DataFrame:
    out_dir = DATA_DIR / str(appid)
    out_dir.mkdir(parents=True, exist_ok=True)
    game = game or get_game_name(appid)
    print(f"\n=== {game} (app {appid}) ===")

    if not skip_collect:
        print("[1/5] Collecting reviews")
        collect(appid, language, "negative", max_negative)
        collect(appid, language, "positive", max_positive)

    print("[2/5] Cleaning and splitting into sentences")
    reviews = load_reviews(DB_PATH, appid)
    if reviews.empty:
        raise SystemExit("No reviews found. Check the app ID or run without --skip-collect.")
    sentences = prepare_sentences(reviews)
    if sentences.empty:
        raise SystemExit("No usable English review text found for this game.")
    n_reviews = sentences["review_id"].nunique()
    print(f"  {len(sentences)} sentences from {n_reviews} English reviews")

    print("[3/5] Finding complaint sentences")
    sentences = add_sentiment(sentences, out_dir / "sentences.pkl")
    complaints = sentences[
        (sentences["sentiment"] == "NEGATIVE") & (sentences["confidence"] >= min_confidence)
    ].reset_index(drop=True)
    if classifier_filter:
        complaints = drop_non_complaints(complaints)
    print(f"  {len(complaints)} complaint sentences in {complaints['review_id'].nunique()} reviews")
    if len(complaints) < MIN_COMPLAINTS:
        raise SystemExit("Too few complaints to analyze. Good news for the developer!")

    print("[4/5] Grouping complaints into pain points")
    topics, labels, mode = find_pain_points(complaints, game, min_topic_size, use_llm)
    complaints["topic"] = topics
    print(f"  {len(labels)} pain points, {(complaints['topic'] == -1).mean():.0%} of complaints unassigned")

    print("[5/5] Ranking and saving")
    ranking = labels.join(rank_topics(complaints), how="inner").sort_values("reviews", ascending=False)
    ranking.to_csv(out_dir / "pain_points.csv")
    complaints.to_csv(out_dir / "complaints.csv", index=False)
    monthly_timeline(complaints).to_csv(out_dir / "timeline.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "appid": appid,
        "game": game,
        "reviews_analyzed": int(n_reviews),
        "negative_reviews": int(sentences.loc[sentences["voted_up"] == 0, "review_id"].nunique()),
        "complaining_reviews": int(complaints["review_id"].nunique()),
        "complaint_sentences": int(len(complaints)),
        "method": mode,
        "label_method": labels["label_method"].mode().iat[0] if len(labels) else None,
        "analyzed_on": date.today().isoformat(),
    }, indent=2))

    cols = ["name", "category", "reviews", "share_of_complaining_reviews", "in_positive_reviews", "median_hours"]
    actionable = ranking[ranking["category"] != "Not actionable"]
    print(f"\nTop pain points for {game} (method: {mode}):\n")
    print(actionable[cols].head(15).to_string() if len(actionable) else "  none found")
    not_actionable = ranking.loc[ranking["category"] == "Not actionable", "name"]
    if len(not_actionable):
        print(f"\nNot actionable: {', '.join(not_actionable)}")
    print(f"\nResults saved to {out_dir}/")
    return ranking


def main():
    parser = argparse.ArgumentParser(description="Find the biggest pain points in a game's Steam reviews.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--game", help='Game name as on Steam, e.g. "Hollow Knight"')
    target.add_argument("--appid", type=int, help="Steam app ID (number in the store URL)")
    parser.add_argument("--skip-collect", action="store_true", help="Reuse reviews already in the database")
    parser.add_argument("--max-negative", type=int, default=None, help="Max negative reviews to fetch (default: all)")
    parser.add_argument("--max-positive", type=int, default=5000, help="Max positive reviews to fetch (default: 5000)")
    parser.add_argument("--language", default="english")
    parser.add_argument("--min-topic-size", type=int, default=None,
                        help="Override clustering granularity (default: scales with data)")
    parser.add_argument("--sentiment-threshold", type=float, default=0.98,
                        help="How sure the model must be that a sentence is a complaint (0-1)")
    parser.add_argument("--no-classifier-filter", action="store_true",
                        help="Skip the trained complaint filter (sentiment only)")
    parser.add_argument("--no-llm", action="store_true", help="Never call the Claude API")
    args = parser.parse_args()

    game = None
    if args.game:
        args.appid, game = resolve_game(args.game)
        print(f"Found: {game} (app {args.appid})")

    run(args.appid, args.skip_collect, args.max_negative, args.max_positive, args.language,
        args.min_topic_size, args.sentiment_threshold, use_llm=not args.no_llm, game=game,
        classifier_filter=not args.no_classifier_filter)


if __name__ == "__main__":
    main()
