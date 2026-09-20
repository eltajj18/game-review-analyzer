"""
Step 1 of training: build a dataset of complaint sentences from many games.

    python -m src.train.build_dataset

Test games are held out completely, so evaluation measures how well the model handles
games it has never seen. Edit TRAIN_GAMES / TEST_GAMES to change the mix.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.analyze.topics import add_sentiment, load_reviews
from src.collect.steam import DB_PATH, collect, resolve_game
from src.process.clean import prepare_sentences

TRAIN_DIR = Path("data/training")

# A mix of genres and sizes, AAA to small indie
TRAIN_GAMES = [
    "Baldur's Gate 3", "Cyberpunk 2077", "Elden Ring", "Cities: Skylines II", "No Man's Sky",
    "Palworld", "Dead by Daylight", "Among Us", "Hollow Knight", "Stardew Valley", "Terraria",
    "Hades", "Slay the Spire", "Lethal Company", "Phasmophobia", "Dave the Diver",
]
TEST_GAMES = ["Starfield", "Valheim", "Celeste", "Vampire Survivors"]


def sentence_id(review_id: str, sentence: str) -> str:
    return hashlib.md5(f"{review_id}|{sentence}".encode()).hexdigest()[:12]


def game_complaints(appid: int, reviews_per_side: int, per_game: int,
                    min_confidence: float = 0.9) -> pd.DataFrame:
    """Sample reviews from the database, keep complaint sentences, sample per_game of them."""
    reviews = load_reviews(DB_PATH, appid)
    sample = pd.concat([
        side.sample(min(reviews_per_side, len(side)), random_state=42)
        for _, side in reviews.groupby("voted_up")
    ]) if len(reviews) else reviews
    sentences = prepare_sentences(sample)
    if sentences.empty:
        return sentences

    # Separate cache from the pipeline's, so a subset never overwrites the full-game cache
    sentences = add_sentiment(sentences, TRAIN_DIR / f"sentiment_{appid}.pkl")
    complaints = sentences[(sentences["sentiment"] == "NEGATIVE")
                           & (sentences["confidence"] >= min_confidence)]
    return complaints.sample(min(per_game, len(complaints)), random_state=42)


def build(games: dict[str, list[tuple[int, str]]], reviews_per_side: int = 400,
          per_game: int = 250, skip_collect: bool = False) -> pd.DataFrame:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for split, entries in games.items():
        for appid, name in entries:
            print(f"\n[{split}] {name} (app {appid})")
            if not skip_collect:
                collect(appid, "english", "negative", reviews_per_side)
                collect(appid, "english", "positive", reviews_per_side)
            complaints = game_complaints(appid, reviews_per_side, per_game)
            print(f"  {len(complaints)} complaint sentences sampled")
            if len(complaints):
                parts.append(complaints.assign(appid=appid, game=name, split=split))

    dataset = pd.concat(parts, ignore_index=True)
    dataset["sent_id"] = [sentence_id(r, s) for r, s in zip(dataset["review_id"], dataset["sentence"])]
    dataset = dataset.drop_duplicates("sent_id")
    dataset = dataset[["sent_id", "appid", "game", "split", "voted_up", "sentence"]]
    dataset.to_csv(TRAIN_DIR / "dataset.csv", index=False)

    print(f"\nSaved {len(dataset)} sentences to {TRAIN_DIR / 'dataset.csv'}")
    print(dataset.groupby(["split", "game"]).size().to_string())
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build the complaint-sentence training dataset.")
    parser.add_argument("--reviews-per-side", type=int, default=400,
                        help="Positive and negative reviews sampled per game")
    parser.add_argument("--per-game", type=int, default=250, help="Complaint sentences kept per game")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    games = {}
    for split, names in (("train", TRAIN_GAMES), ("test", TEST_GAMES)):
        games[split] = [resolve_game(n, interactive=False) for n in names]
    print("Games (check these are the right ones):")
    for split, entries in games.items():
        print(f"  {split}: " + ", ".join(name for _, name in entries))

    build(games, args.reviews_per_side, args.per_game, args.skip_collect)


if __name__ == "__main__":
    main()
