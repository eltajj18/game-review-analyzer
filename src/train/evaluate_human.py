"""
Compare your hand labels against Claude's labels and the trained classifier.

    python -m src.train.evaluate_human

Three numbers matter:
- Claude vs human: how good the teacher's labels are (the student's realistic ceiling)
- classifier vs human: the honest quality of the model
- classifier vs Claude: how well the student imitates its teacher
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.analyze.aspects import ASPECTS, MODEL_PATH
from src.train.build_dataset import TRAIN_DIR
from src.train.train_classifier import to_category

HUMAN_PATH = TRAIN_DIR / "human_labels.csv"


def compare(name: str, y_true, y_pred) -> dict:
    return {
        "comparison": name,
        "aspect_accuracy": round(accuracy_score(y_true, y_pred), 3),
        "aspect_macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "aspect_kappa": round(cohen_kappa_score(y_true, y_pred), 3),
        "category_accuracy": round(accuracy_score(to_category(y_true), to_category(y_pred)), 3),
    }


def main():
    from sentence_transformers import SentenceTransformer

    human = pd.read_csv(HUMAN_PATH)
    human = human[human["aspect"].isin(ASPECTS)]  # drop "unclear"
    data = (pd.read_csv(TRAIN_DIR / "dataset.csv")
            .merge(human.rename(columns={"aspect": "human"}), on="sent_id")
            .merge(pd.read_csv(TRAIN_DIR / "labels.csv").drop_duplicates("sent_id")
                   .rename(columns={"aspect": "claude"}), on="sent_id"))
    if data.empty:
        raise SystemExit("No overlap between your labels and Claude's. Label some sentences first.")

    bundle = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(bundle["embedding_model"])
    data["model"] = bundle["model"].predict(
        embedder.encode(data["sentence"].tolist(), normalize_embeddings=True))

    rows = [
        compare("Claude vs human (teacher quality)", data["human"], data["claude"]),
        compare("classifier vs human (honest score)", data["human"], data["model"]),
        compare("classifier vs Claude (imitation)", data["claude"], data["model"]),
    ]
    table = pd.DataFrame(rows).set_index("comparison")
    print(f"\nOn {len(data)} hand-labeled sentences from unseen games:\n")
    print(table.to_string())
    print("\nKappa corrects for agreement by chance: below 0.2 is poor, above 0.6 is strong.")

    disagreements = data[data["human"] != data["claude"]]
    print(f"\nYou and Claude disagreed on {len(disagreements)} of {len(data)} sentences. Examples:")
    for _, r in disagreements.head(8).iterrows():
        print(f"  you: {r['human']:<14} claude: {r['claude']:<14} {r['sentence'][:80]}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (MODEL_PATH.parent / "human_eval.json").write_text(json.dumps(
        {"n_sentences": len(data), "results": rows}, indent=2))
    print(f"\nSaved to {MODEL_PATH.parent / 'human_eval.json'}")


if __name__ == "__main__":
    main()
