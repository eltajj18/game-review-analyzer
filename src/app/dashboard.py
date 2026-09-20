"""
Dashboard for pipeline results.

    streamlit run src/app/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path("data")
CATEGORY_COLORS = {
    "Technical": "#e45756", "Gameplay": "#4c78a8", "Content": "#b279a2",
    "UX": "#f58518", "Business": "#54a24b", "Not actionable": "#9d9d9d",
}

st.set_page_config(page_title="Game Review Pain Point Analyzer", page_icon="🎮", layout="wide")


def read_meta(game_dir: Path) -> dict:
    path = game_dir / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {"game": f"App {game_dir.name}"}


@st.cache_data
def load(game_dir: str, modified: float):
    """`modified` is part of the cache key, so rerunning the pipeline refreshes the dashboard."""
    d = Path(game_dir)
    pain = pd.read_csv(d / "pain_points.csv")
    complaints = pd.read_csv(d / "complaints.csv")
    # Results from older pipeline versions
    pain = pain.rename(columns={"share_of_negative_reviews": "share_of_complaining_reviews"})
    if "in_positive_reviews" not in pain:
        pain["in_positive_reviews"] = 0
    if "voted_up" not in complaints:
        complaints["voted_up"] = 0
    timeline = pd.read_csv(d / "timeline.csv", parse_dates=["month"])
    return pain, complaints, timeline


# ---------- Game picker ----------
# Most recently analyzed game first
results = [p for p in DATA_DIR.glob("*") if (p / "pain_points.csv").exists()]
results.sort(key=lambda p: (p / "pain_points.csv").stat().st_mtime, reverse=True)
games = [p.name for p in results]
if not games:
    st.title("🎮 Game Review Pain Point Analyzer")
    st.info("No results yet. Run the pipeline first:\n\n`python -m src.pipeline --appid <steam app id>`")
    st.stop()

game_names = {g: read_meta(DATA_DIR / g)["game"] for g in games}
game_dir = DATA_DIR / st.sidebar.selectbox("Game", games, format_func=lambda g: game_names.get(str(g), str(g)))
meta = read_meta(game_dir)
pain, complaints, timeline = load(str(game_dir), (game_dir / "pain_points.csv").stat().st_mtime)

# ---------- Filters ----------
st.sidebar.header("Filters")
all_categories = [c for c in CATEGORY_COLORS if c in set(pain["category"])]
default = [c for c in all_categories if c != "Not actionable"]
categories = st.sidebar.multiselect("Categories", all_categories, default=default, key="categories")
if len(pain) > 5:
    top_n = st.sidebar.slider("Pain points shown", 5, len(pain), min(15, len(pain)))
else:
    top_n = len(pain)  # small games: just show everything

shown = pain[pain["category"].isin(categories)].head(top_n)
names = pain.set_index("topic")["name"]

# ---------- Header ----------
st.title(f"🎮 {meta['game']}")
METHODS = {"clustering": "topic clustering", "llm": "grouped by Claude",
           "classifier": "trained aspect classifier", "templates": "template matching"}
n_analyzed = meta.get("reviews_analyzed", meta.get("negative_reviews", 0))
st.caption(
    f"Based on {n_analyzed:,} English Steam reviews"
    + (f" · method: {METHODS.get(meta['method'], meta['method'])}" if meta.get("method") else "")
    + f" · analyzed {meta.get('analyzed_on', 'unknown date')}"
)

actionable = pain[pain["category"] != "Not actionable"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analyzed", f"{n_analyzed:,}")
c2.metric("Reviews with complaints", f"{complaints['review_id'].nunique():,}")
c3.metric("Actionable pain points", len(actionable))
if not actionable.empty:
    top = actionable.iloc[0]
    c4.metric("Biggest pain point", top["name"], f"{top['share_of_complaining_reviews']:.0%} of complaints",
              delta_color="off")

# ---------- Ranking ----------
st.subheader("Biggest pain points")
st.caption("Of all reviews containing a complaint, the share that mention each issue. "
           "Includes complaints inside positive reviews.")
if shown.empty:
    st.warning("No pain points match the selected categories.")
else:
    fig = px.bar(
        shown.iloc[::-1], x="share_of_complaining_reviews", y="name", color="category",
        orientation="h", color_discrete_map=CATEGORY_COLORS,
        hover_data={"reviews": True, "in_positive_reviews": True, "total_upvotes": True,
                    "median_hours": True, "name": False},
        labels={"share_of_complaining_reviews": "Share of reviews with complaints", "name": "",
                "in_positive_reviews": "In positive reviews"},
    )
    fig.update_layout(xaxis_tickformat=".0%", height=max(300, 32 * len(shown)),
                      legend_title_text="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# ---------- Categories ----------
left, right = st.columns(2)
with left:
    st.subheader("By category")
    by_cat = (complaints.merge(pain[["topic", "category"]], on="topic")
              .groupby("category")["review_id"].nunique().sort_values())
    fig = px.bar(by_cat, orientation="h", color=by_cat.index, color_discrete_map=CATEGORY_COLORS,
                 labels={"value": "Reviews mentioning it", "category": ""})
    fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# ---------- Timeline ----------
with right:
    st.subheader("Over time")
    options = pain["topic"].tolist()
    default_topics = shown["topic"].head(3).tolist()
    picked = st.multiselect("Pain points", options, default=default_topics,
                            format_func=lambda t: names.get(t, str(t)), key="timeline_topics")
    series = timeline[timeline["topic"].isin(picked)].assign(name=lambda d: d["topic"].map(names))
    if series.empty:
        st.info("Pick one or more pain points.")
    else:
        fig = px.line(series, x="month", y="reviews", color="name", markers=True,
                      labels={"reviews": "Reviews per month", "month": "", "name": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    st.caption("Spikes often line up with patches, releases or controversies.")

# ---------- Drill-down ----------
st.subheader("What players actually say")
topic = st.selectbox("Pain point", pain["topic"].tolist(),
                     format_func=lambda t: names.get(t, str(t)), key="drilldown")
row = pain.set_index("topic").loc[topic]
st.markdown(
    f"**{row['name']}** · {row['category']} · mentioned in **{row['reviews']:,}** reviews "
    f"({row['share_of_complaining_reviews']:.0%}), {int(row['in_positive_reviews']):,} of them positive "
    f"· median **{row['median_hours']:.0f} h** played"
)
quotes = (complaints[complaints["topic"] == topic]
          .sort_values("votes_up", ascending=False)
          .drop_duplicates("review_id").head(10))
for _, q in quotes.iterrows():
    verdict = "👍 Recommended" if q["voted_up"] == 1 else "👎 Not recommended"
    st.markdown(f"> {q['sentence']}\n\n<small>{verdict} · {q['votes_up']:,} found helpful · "
                f"{q['hours_at_review']:.0f} h played</small>",
                unsafe_allow_html=True)

# ---------- Raw table ----------
with st.expander("All pain points (table)"):
    cols = ["name", "category", "reviews", "share_of_complaining_reviews", "in_positive_reviews",
            "total_upvotes", "median_hours", "example"]
    st.dataframe(pain[cols], width="stretch", hide_index=True)
    st.download_button("Download CSV", pain.to_csv(index=False), f"{game_dir.name}_pain_points.csv")
