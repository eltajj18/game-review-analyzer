"""
Step 3 of training: train the aspect classifier (the "student") and evaluate it.

    python -m src.train.train_classifier

Two models, because one model doing both jobs confuses "not a complaint" with
"a complaint that isn't actionable":
1. complaint detector: binary, is this sentence a complaint at all?
2. aspect model: trained only on real complaints, which of the 16 aspects is it?

- Features: MiniLM sentence embeddings (the same model the pipeline already uses)
- Model: logistic regression; regularization strength picked by cross-validation
  grouped by game, so tuning also rewards generalizing to unseen games
- Evaluation: on held-out test games, against Claude's labels, compared with
  a majority-class baseline and the old template-matching method
- The final model is refit on all labeled data and saved to models/
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.analyze.aspects import ASPECTS, LABEL_CHOICES, MODEL_PATH, NON_COMPLAINT, canonical
from src.analyze.topics import EMBEDDING_MODEL
from src.train.build_dataset import TRAIN_DIR

C_VALUES = [0.25, 0.5, 1, 2, 4, 8]


def to_category(aspects) -> list[str]:
    return [ASPECTS[a][1] if a in ASPECTS else "Not a complaint" for a in aspects]


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

    parser = argparse.ArgumentParser(description="Train the aspect classifier.")
    parser.add_argument("--min-sentiment", type=float, default=0.0,
                        help="Drop training sentences the complaint filter was less sure about "
                             "(see src.train.evaluate_filter)")
    args = parser.parse_args()

    data = pd.read_csv(TRAIN_DIR / "dataset.csv").merge(
        pd.read_csv(TRAIN_DIR / "labels.csv").drop_duplicates("sent_id"), on="sent_id")
    data["aspect"] = data["aspect"].map(canonical)  # merged aspects, no relabeling needed

    if args.min_sentiment:
        from src.train.evaluate_filter import load_scored_sentences
        before = len(data)
        sentiment_scores = load_scored_sentences()[["sent_id", "confidence"]]
        data = data.merge(sentiment_scores, on="sent_id", how="left")
        data = data[data["confidence"].fillna(1.0) >= args.min_sentiment].reset_index(drop=True)
        print(f"Dropped {before - len(data)} of {before} sentences below "
              f"sentiment confidence {args.min_sentiment}")
        if len(data) < 50:
            raise SystemExit("Almost nothing left; lower --min-sentiment.")
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    data = data[data["aspect"].isin(LABEL_CHOICES)]
    print(f"{len(ASPECTS)} aspects (+ not_complaint) after merging")
    print(f"{len(train)} training sentences ({train['game'].nunique()} games), "
          f"{len(test)} test sentences ({test['game'].nunique()} unseen games)\n")

    print("Embedding sentences...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(data["sentence"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    X_train, X_test = X[data["split"].to_numpy() == "train"], X[data["split"].to_numpy() == "test"]

    def tune(features, targets, groups, what: str) -> float:
        folds = GroupKFold(n_splits=min(5, groups.nunique()))
        best, scores_by_c = None, {}
        for C in C_VALUES:
            model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            pred = cross_val_predict(model, features, targets, groups=groups, cv=folds)
            scores_by_c[C] = f1_score(targets, pred, average="macro", zero_division=0)
        best = max(scores_by_c, key=scores_by_c.get)
        print(f"  {what}: best C = {best} (cross-validated macro F1 "
              f"{scores_by_c[best]:.3f}, grouped by game)")
        return best

    def fit(features, targets, C):
        return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(features, targets)

    # --- Stage 1: is this a complaint at all?
    is_complaint = (data["aspect"] != NON_COMPLAINT).to_numpy()
    train_mask = (data["split"] == "train").to_numpy()
    C_detect = tune(X_train, is_complaint[train_mask], train["game"], "complaint detector")
    detector = fit(X_train, is_complaint[train_mask], C_detect)
    detect_pred = detector.predict(X_test)
    detect_true = is_complaint[~train_mask]

    # --- Stage 2: which aspect, trained on real complaints only
    aspect_train = train[train["aspect"] != NON_COMPLAINT]
    aspect_test = test[test["aspect"] != NON_COMPLAINT]
    Xa_train = X_train[(train["aspect"] != NON_COMPLAINT).to_numpy()]
    Xa_test = X_test[(test["aspect"] != NON_COMPLAINT).to_numpy()]
    C_aspect = tune(Xa_train, aspect_train["aspect"], aspect_train["game"], "aspect model")
    aspect_model = fit(Xa_train, aspect_train["aspect"], C_aspect)
    pred = aspect_model.predict(Xa_test)
    majority = aspect_train["aspect"].mode().iat[0]

    true_cat = to_category(aspect_test["aspect"])
    results = {
        "complaint detection (binary)": {
            "always 'complaint'": scores(detect_true, [True] * len(detect_true)),
            "detector": scores(detect_true, detect_pred),
        },
        f"aspect level ({len(ASPECTS)} classes, real complaints only)": {
            "majority baseline": scores(aspect_test["aspect"], [majority] * len(aspect_test)),
            "classifier": scores(aspect_test["aspect"], pred),
        },
        "category level (6 classes)": {
            "majority baseline": scores(true_cat, to_category([majority] * len(aspect_test))),
            "templates (old method)": scores(true_cat, template_categories(aspect_test["sentence"].tolist())),
            "classifier": scores(true_cat, to_category(pred)),
        },
    }

    print("\nResults on unseen test games (agreement with Claude's labels):")
    for level, rows in results.items():
        print(f"\n  {level}")
        for method, s in rows.items():
            print(f"    {method:<24} accuracy {s['accuracy']:.3f}   macro F1 {s['macro_f1']:.3f}")
    report = classification_report(aspect_test["aspect"], pred, zero_division=0)
    print("\nPer-aspect results:\n" + report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (MODEL_PATH.parent / "classification_report.txt").write_text(report)
    _save_confusion_matrix(aspect_test["aspect"], pred)

    # --- Final models: refit on everything we have
    complaints_only = (data["aspect"] != NON_COMPLAINT).to_numpy()
    joblib.dump({
        "complaint_model": fit(X, is_complaint, C_detect),
        "aspect_model": fit(X[complaints_only], data.loc[complaints_only, "aspect"], C_aspect),
        "embedding_model": EMBEDDING_MODEL,
    }, MODEL_PATH)
    meta = {"C_detector": C_detect, "C_aspect": C_aspect, "min_sentiment": args.min_sentiment,
            "train_sentences": len(train), "test_sentences": len(test),
            "test_games": sorted(test["game"].unique()), "results": results,
            "final_model_trained_on": len(data)}
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved both models to {MODEL_PATH} (refit on all {len(data)} labeled sentences)")


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
