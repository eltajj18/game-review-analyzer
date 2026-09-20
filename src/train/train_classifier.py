"""
Step 3 of training: train the aspect classifier (the "student") and evaluate it.

    python -m src.train.train_classifier

- Features: MiniLM sentence embeddings (the same model the pipeline already uses)
- Model: logistic regression; regularization strength picked by cross-validation
  grouped by game, so tuning also rewards generalizing to unseen games
- Evaluation: on held-out test games, against Claude's labels, compared with
  a majority-class baseline and the old template-matching method
- The final model is refit on all labeled data and saved to models/
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.analyze.aspects import ASPECTS, MODEL_PATH
from src.analyze.topics import EMBEDDING_MODEL
from src.train.build_dataset import TRAIN_DIR

C_VALUES = [0.25, 0.5, 1, 2, 4, 8]


def to_category(aspects) -> list[str]:
    return [ASPECTS[a][1] if a in ASPECTS else "Unassigned" for a in aspects]


def template_categories(sentences: list[str]) -> list[str]:
    """The previous offline method, for comparison."""
    from src.analyze.small import template_group

    topics, labels = template_group(sentences)
    return [labels[t][1] if t != -1 else "Unassigned" for t in topics]


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)}


def main():
    from sentence_transformers import SentenceTransformer

    data = pd.read_csv(TRAIN_DIR / "dataset.csv").merge(
        pd.read_csv(TRAIN_DIR / "labels.csv").drop_duplicates("sent_id"), on="sent_id")
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    print(f"{len(train)} training sentences ({train['game'].nunique()} games), "
          f"{len(test)} test sentences ({test['game'].nunique()} unseen games)\n")

    print("Embedding sentences...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(data["sentence"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    X_train, X_test = X[data["split"].to_numpy() == "train"], X[data["split"].to_numpy() == "test"]

    # --- Tune C with cross-validation grouped by game
    folds = GroupKFold(n_splits=min(5, train["game"].nunique()))
    cv = {}
    for C in C_VALUES:
        model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        pred = cross_val_predict(model, X_train, train["aspect"], groups=train["game"], cv=folds)
        cv[C] = f1_score(train["aspect"], pred, average="macro", zero_division=0)
        print(f"  C={C:<5} cross-validated macro F1 = {cv[C]:.3f}")
    best_C = max(cv, key=cv.get)
    print(f"Best C = {best_C}\n")

    # --- Evaluate on held-out games
    model = LogisticRegression(C=best_C, max_iter=3000, class_weight="balanced").fit(X_train, train["aspect"])
    pred = model.predict(X_test)
    majority = train["aspect"].mode().iat[0]

    true_cat = to_category(test["aspect"])
    results = {
        "aspect level (21 classes)": {
            "majority baseline": scores(test["aspect"], [majority] * len(test)),
            "classifier": scores(test["aspect"], pred),
        },
        "category level (6 classes)": {
            "majority baseline": scores(true_cat, to_category([majority] * len(test))),
            "templates (old method)": scores(true_cat, template_categories(test["sentence"].tolist())),
            "classifier": scores(true_cat, to_category(pred)),
        },
    }

    print("Results on unseen test games (agreement with Claude's labels):")
    for level, rows in results.items():
        print(f"\n  {level}")
        for method, s in rows.items():
            print(f"    {method:<24} accuracy {s['accuracy']:.3f}   macro F1 {s['macro_f1']:.3f}")
    report = classification_report(test["aspect"], pred, zero_division=0)
    print("\nPer-aspect results:\n" + report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (MODEL_PATH.parent / "classification_report.txt").write_text(report)
    _save_confusion_matrix(test["aspect"], pred)

    # --- Final model: refit on everything we have
    final = LogisticRegression(C=best_C, max_iter=3000, class_weight="balanced").fit(X, data["aspect"])
    joblib.dump({"model": final, "embedding_model": EMBEDDING_MODEL}, MODEL_PATH)
    meta = {"best_C": best_C, "train_sentences": len(train), "test_sentences": len(test),
            "test_games": sorted(test["game"].unique()), "results": results,
            "final_model_trained_on": len(data)}
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {MODEL_PATH} (refit on all {len(data)} labeled sentences)")


def _save_confusion_matrix(y_true, y_pred):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("(install matplotlib to also save a confusion matrix image)")
        return
    labels = [a for a in ASPECTS if a in set(y_true) | set(y_pred)]
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true",
                                            values_format=".1f", ax=ax, colorbar=False,
                                            xticks_rotation=90)
    ax.set_title("Aspect classifier on unseen games (rows sum to 1)")
    fig.tight_layout()
    fig.savefig(MODEL_PATH.parent / "confusion_matrix.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
