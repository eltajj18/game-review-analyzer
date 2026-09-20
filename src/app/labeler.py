"""
Hand-label test sentences to measure the classifier against human judgment.

    streamlit run src/app/labeler.py

Claude's label is deliberately hidden: seeing it first would bias your choice and
make the agreement number meaningless. Labels are saved after every click, so you
can stop and come back.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analyze.aspects import ASPECTS

TRAIN_DIR = Path("data/training")
LABELS_PATH = TRAIN_DIR / "human_labels.csv"
SAMPLE_SIZE = 100

st.set_page_config(page_title="Label sentences", page_icon="🏷️", layout="centered")


@st.cache_data
def sample_sentences(n: int) -> pd.DataFrame:
    dataset = pd.read_csv(TRAIN_DIR / "dataset.csv")
    test = dataset[dataset["split"] == "test"]
    return test.sample(min(n, len(test)), random_state=0).reset_index(drop=True)


def load_done() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["sent_id", "aspect"])


def save(sent_id: str, aspect: str):
    pd.DataFrame([{"sent_id": sent_id, "aspect": aspect}]).to_csv(
        LABELS_PATH, mode="a", header=not LABELS_PATH.exists(), index=False)


if not (TRAIN_DIR / "dataset.csv").exists():
    st.error("No dataset found. Run `python -m src.train.build_dataset` first.")
    st.stop()

sentences = sample_sentences(SAMPLE_SIZE)
done = load_done()
remaining = sentences[~sentences["sent_id"].isin(done["sent_id"])]

st.title("🏷️ Label complaint sentences")
st.progress(len(done) / len(sentences), text=f"{len(done)} / {len(sentences)} labeled")

if remaining.empty:
    st.success("All done! Run `python -m src.train.evaluate_human` to see the results.")
    st.bar_chart(done["aspect"].value_counts())
    if st.button("Start over (deletes your labels)"):
        LABELS_PATH.unlink(missing_ok=True)
        st.rerun()
    st.stop()

current = remaining.iloc[0]
st.markdown(f"### “{current['sentence']}”")
st.caption(f"from a review of {current['game']}")
st.write("**What is the player complaining about?**")

keys = list(ASPECTS)
columns = st.columns(3)
for i, key in enumerate(keys):
    name, category, _ = ASPECTS[key]
    if columns[i % 3].button(name, key=f"btn_{key}", width="stretch"):
        save(current["sent_id"], key)
        st.rerun()

st.divider()
left, right = st.columns(2)
if left.button("Skip (unclear or not a complaint)", width="stretch"):
    save(current["sent_id"], "unclear")
    st.rerun()
if right.button("Undo last", width="stretch", disabled=done.empty):
    done.iloc[:-1].to_csv(LABELS_PATH, index=False)
    st.rerun()
