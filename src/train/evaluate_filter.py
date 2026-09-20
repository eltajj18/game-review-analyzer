"""
How good is the complaint filter? Measures it against your hand labels.

    python -m src.train.evaluate_filter
    python -m src.train.evaluate_filter --compare cardiffnlp/twitter-roberta-base-sentiment-latest

Every sentence in the dataset already passed the filter, so this measures precision
(what share of kept sentences are real complaints) and how much data each setting keeps.
Sentences you marked "not a complaint" are the filter's mistakes.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.train.build_dataset import TRAIN_DIR, sentence_id

THRESHOLDS = [0.9, 0.95, 0.98, 0.99, 0.995]


def load_scored_sentences() -> pd.DataFrame:
    """Sentences with the sentiment score the pipeline gave them."""
    parts = []
    for path in sorted(TRAIN_DIR.glob("sentiment_*.pkl")):
        df = pd.read_pickle(path)
        df["sent_id"] = [sentence_id(r, s) for r, s in zip(df["review_id"], df["sentence"])]
        parts.append(df[["sent_id", "sentence", "sentiment", "confidence"]])
    if not parts:
        raise SystemExit("No sentiment caches found. Run src.train.build_dataset first.")
    return pd.concat(parts).drop_duplicates("sent_id")


def report(name: str, data: pd.DataFrame, score_col: str, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        kept = data[data[score_col] >= t]
        if kept.empty:
            continue
        rows.append({
            "filter": name,
            "threshold": t,
            "kept": len(kept),
            "kept_share": round(len(kept) / len(data), 2),
            "real_complaints": round((kept["is_complaint"]).mean(), 3),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the complaint filter against hand labels.")
    parser.add_argument("--compare", help="Also score the sentences with another sentiment model")
    args = parser.parse_args()

    human = pd.read_csv(TRAIN_DIR / "human_labels.csv")
    human = human[human["aspect"] != "unclear"]          # genuinely ambiguous: excluded
    human["is_complaint"] = human["aspect"] != "not_complaint"

    data = human.merge(load_scored_sentences(), on="sent_id")
    if data.empty:
        raise SystemExit("Could not match your labels to scored sentences.")
    print(f"\n{len(data)} hand-labeled sentences, of which "
          f"{(~data['is_complaint']).sum()} are not actually complaints "
          f"({(~data['is_complaint']).mean():.0%} of what the filter let through)\n")

    tables = [report("distilbert sst-2 (current)", data, "confidence")]

    if args.compare:
        from transformers import pipeline
        print(f"Scoring with {args.compare}...")
        clf = pipeline("sentiment-analysis", model=args.compare, truncation=True)
        preds = clf(data["sentence"].tolist(), batch_size=32)
        # Models differ in label names; treat anything starting with "neg" as negative
        data["other"] = [p["score"] if p["label"].lower().startswith("neg") else 0.0 for p in preds]
        tables.append(report(args.compare.split("/")[-1], data, "other"))

    print(pd.concat(tables).to_string(index=False))
    print("\n'real_complaints' is precision: the share of kept sentences that are genuine complaints.")
    print("Pick the threshold that raises precision without throwing away too much data,")
    print("then pass it to the pipeline with --sentiment-threshold.")

    mistakes = data[~data["is_complaint"]].sort_values("confidence", ascending=False)
    print(f"\nExamples the filter got wrong (highest confidence first):")
    for _, r in mistakes.head(8).iterrows():
        print(f"  {r['confidence']:.3f}  {r['sentence'][:90]}")


if __name__ == "__main__":
    main()
