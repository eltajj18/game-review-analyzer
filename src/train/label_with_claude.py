"""
Step 2 of training: have Claude label each sentence with an aspect (the "teacher").

    python -m src.train.label_with_claude --limit 200   # try a small batch first
    python -m src.train.label_with_claude               # label everything

Progress is appended to data/training/labels.csv after every batch, so the script can be
stopped and resumed; already-labeled sentences are skipped.
"""

import argparse
import json
import re
import time

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.analyze.aspects import ASPECTS
from src.analyze.label import LLM_MODEL
from src.train.build_dataset import TRAIN_DIR

BATCH_SIZE = 40
ASPECT_LIST = "\n".join(f'- "{key}": {desc}' for key, (_, _, desc) in ASPECTS.items())


def label_batch(client, sentences: list[str]) -> dict[int, str]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    prompt = f"""Each sentence below is a complaint from a Steam game review.
Classify each one into exactly one aspect: the main thing the player is complaining about.

Aspects:
{ASPECT_LIST}

Sentences:
{numbered}

Respond with ONLY a JSON object mapping each sentence number to an aspect key, no other text:
{{"0": "bugs", "1": "price", ...}}"""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    raw = json.loads(re.sub(r"```(?:json)?", "", text).strip())
    return {int(k): v for k, v in raw.items() if v in ASPECTS and str(k).isdigit()}


def main():
    parser = argparse.ArgumentParser(description="Label training sentences with Claude.")
    parser.add_argument("--limit", type=int, default=None, help="Only label this many new sentences")
    args = parser.parse_args()

    import anthropic
    from tqdm import tqdm

    dataset = pd.read_csv(TRAIN_DIR / "dataset.csv")
    labels_path = TRAIN_DIR / "labels.csv"
    done = set(pd.read_csv(labels_path)["sent_id"]) if labels_path.exists() else set()
    todo = dataset[~dataset["sent_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already labeled, {len(todo)} to go")
    if todo.empty:
        return

    client = anthropic.Anthropic()
    skipped = 0
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="labeling"):
        batch = todo.iloc[start:start + BATCH_SIZE]
        for attempt in range(3):
            try:
                result = label_batch(client, batch["sentence"].tolist())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n  batch failed 3 times ({e}), skipping it")
                    result = {}
                time.sleep(2 ** attempt)

        rows = [{"sent_id": sid, "aspect": result[i]}
                for i, sid in enumerate(batch["sent_id"]) if i in result]
        skipped += len(batch) - len(rows)
        if rows:
            pd.DataFrame(rows).to_csv(labels_path, mode="a", header=not labels_path.exists(), index=False)

    total = pd.read_csv(labels_path)
    print(f"\nDone. {len(total)} labeled sentences in {labels_path}"
          + (f" ({skipped} skipped; rerun to retry them)" if skipped else ""))
    print(total["aspect"].value_counts().to_string())


if __name__ == "__main__":
    main()
