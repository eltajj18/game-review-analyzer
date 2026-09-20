# Game Review Pain Point Analyzer

Finds what players actually complain about in a game's Steam reviews, groups the complaints
into ranked pain points, and shows them in a dashboard — so a developer can see what to fix
first instead of reading thousands of reviews.

![Dashboard](docs/images/dashboard.png)
<!-- TODO: screenshot of the dashboard. Win+Shift+S, save to docs/images/dashboard.png -->

```bash
python -m src.pipeline --game "Hollow Knight"
streamlit run src/app/dashboard.py
```

## What it does

Reviews are messy: a single review mixes praise, complaints and jokes, and most of a negative
review isn't a complaint at all. The pipeline works at the **sentence** level, decides which
sentences are genuine complaints, and groups those into pain points ranked by how many reviews
mention them.

It adapts to the size of the game, which matters because a hit has 50,000 reviews and an indie
game might have 50:

| Complaints found | Method | Why |
|---|---|---|
| 300+ | BERTopic clustering | Discovers game-specific issues no fixed list could predict |
| < 300, with an API key | Claude groups them directly | Statistics need data; an LLM can read 80 sentences |
| < 300, offline | Trained aspect classifier | Free, offline, no API key needed |

## Quick start

```bash
git clone https://github.com/eltajj18/game-review-analyzer
cd game-review-analyzer
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt

python -m src.pipeline --game "Baldur's Gate 3"       # or --appid 1086940
streamlit run src/app/dashboard.py
```

Reviews aren't included in the repo; the first command downloads them into `data/`.
An `ANTHROPIC_API_KEY` in a `.env` file is optional and only improves topic naming.

Useful flags: `--skip-collect` (reuse downloaded reviews), `--max-negative`, `--max-positive`,
`--sentiment-threshold`, `--min-topic-size`, `--no-llm`.

## How it works

```
Steam reviews  →  clean & split     →  complaint filter    →  grouping        →  ranking
(positive +       BBCode, English      sentiment model +      clustering /       per review,
 negative)        detection,           trained complaint      LLM /              weighted by
                  sentences            detector               classifier         upvotes
```

1. **Collect** — Steam's public review API, stored in SQLite. Positive reviews are included:
   "great game, but the camera is awful" is a complaint too.
2. **Clean & split** — strip Steam markup, detect language per review (Steam's language field is
   what the reviewer *selected*, not what they wrote), split into sentences.
3. **Filter** — a sentiment model keeps negative sentences, then a trained complaint detector
   removes the neutral narration sentiment alone lets through.
4. **Group** — into pain points, by whichever of the three methods fits the amount of data.
5. **Rank** — counted per review, so one long rant can't inflate a topic, with community upvotes
   and playtime as extra signals. Topics are also categorised, and "not actionable" ones
   (genre preference, general dislike) are separated from fixable ones.

### Design decisions

**Sentence level, not review level.** Clustering whole reviews produced *one* topic covering
3,854 negative Baldur's Gate 3 reviews, labelled `game, like, just, time` — because every review
mentions the story, the combat, a bug and the price at once, so they all look alike. Splitting
into sentences turned that into ~25 specific pain points.

**A complaint detector, not just sentiment.** The sentiment model answers "is this negative?",
which isn't the same question. Measured against hand labels, 13% of what it kept weren't
complaints ("I can't look at any component and say it's bad" scored 0.9999 negative). Raising the
threshold to 0.99 lifts precision to 96% while keeping 78% of the data; a trained binary detector
then removes more of what's left.

**Clustering for big games, a classifier for small ones.** Clustering finds issues no fixed list
would contain — "Act 3 is disjointed", a review-bombing episode — but needs volume. A fixed
aspect model works on 40 sentences but can only find what it was trained on.

## Results

### Aspect classifier

Trained by distillation: Claude labelled ~5,000 complaint sentences (one-off cost: **$0.43**),
then a logistic regression on MiniLM embeddings learned to imitate it. The model runs offline
and free. Evaluated on **4 games held out entirely** from training (16 games in, 4 unseen).

| | accuracy | macro F1 |
|---|---|---|
| Aspect level (16 classes, n=790) | | |
| &nbsp;&nbsp;majority baseline | 0.284 | 0.028 |
| &nbsp;&nbsp;**classifier** | **0.509** | **0.501** |
| Category level (6 classes) | | |
| &nbsp;&nbsp;template matching (previous method) | 0.268 | 0.290 |
| &nbsp;&nbsp;**classifier** | **0.597** | **0.589** |

The classifier roughly doubles the template baseline it replaced.

### How good are the labels?

Model-vs-model scores can flatter a distilled model, so 100 test sentences were hand-labelled
independently (without seeing Claude's answers):

| | agreement | Cohen's κ |
|---|---|---|
| Claude vs human (teacher quality) | 0.487 | 0.411 |
| Classifier vs human (honest score) | 0.359 | 0.286 |
| Classifier vs Claude (imitation) | 0.577 | 0.528 |

**The ceiling is the labels, not the model.** Claude agrees with a human on only ~49% of
sentences, so a student trained to imitate Claude can't do much better. The disagreements are
mostly genuine judgement calls, not errors:

| Sentence | Human | Claude |
|---|---|---|
| "Many of the powers are either great bordering on OP or pointless." | mechanics | difficulty |
| "it's obnoxious how many loading phases the game has." | visuals | performance |
| "A mess upon release and now a mess with more content." | content | support |

Aspect boundaries were merged (21 → 16) and their definitions sharpened based on exactly these
disagreements, which raised teacher–human agreement from 0.42 to 0.48 κ. Further gains need
multi-label annotation and more than one annotator.

## Findings

**Baldur's Gate 3** — the most-mentioned actionable pain points were Act 3's quality, combat
pacing, dice/RNG frustration, bugs and crashes, camera and controls.

The tool also surfaced something nobody asked it to look for: a cluster of complaints about the
developer rather than the game, spiking in one month. It corresponds to a review-bombing episode
after the studio's comments about using AI tools — an event visible in the data only as a sudden
change in what people complained about.

<!-- TODO: add your indie game's results here — game name, top 3 pain points, which method was
     used (it should say "classifier" or "grouped by Claude"), and a timeline screenshot if the
     spike is visible. This is the most interesting part of the README for a reader. -->

## Project structure

```
src/
├── collect/steam.py        Steam API: reviews, game search by name
├── process/clean.py        markup, language detection, sentence splitting
├── analyze/
│   ├── topics.py           sentiment filter, BERTopic clustering, ranking
│   ├── small.py            small-game strategies: LLM, classifier, templates
│   ├── label.py            topic naming (LLM or embeddings)
│   └── aspects.py          the 16 aspects and their definitions
├── train/                  dataset building, labelling, training, evaluation
├── app/dashboard.py        Streamlit dashboard
├── app/labeler.py          hand-labelling tool for evaluation
└── pipeline.py             one command, end to end
notebooks/                  exploration that led to the pipeline
models/                     trained classifier + metrics (committed, ~1 MB)
```

## Training your own classifier

Not needed to use the tool — a trained model is included. To reproduce it:

```bash
python -m src.train.build_dataset        # ~5,000 complaint sentences from 20 games
python -m src.train.label_with_claude    # Claude labels them (~$0.43, needs an API key)
python -m src.train.train_classifier     # trains, evaluates on unseen games, saves models/
streamlit run src/app/labeler.py         # hand-label 100 sentences for honest evaluation
python -m src.train.evaluate_human
python -m src.train.evaluate_filter      # precision of the complaint filter
```

## Limitations

- **Steam only, English only.** Epic has no written reviews; Steam Community discussions,
  itch.io and mobile stores would be the natural next sources.
- **One label per sentence.** Many complaints legitimately span two aspects.
- **Evaluation sets are small** — 100 hand-labelled sentences, so differences below ~6 points
  aren't meaningful.
- **A single annotator.** No inter-annotator agreement, so "human labels" means one person's
  judgement.
- **Sampling bias.** Collecting the most recent reviews over-represents whatever is happening
  right now, such as a review-bombing episode.

## Built with

Python, pandas, scikit-learn, sentence-transformers, BERTopic, Hugging Face Transformers,
Streamlit, SQLite, and the Claude API (optional).

Review data comes from Steam's public review API and belongs to its authors; only code and the
trained model are in this repository.
