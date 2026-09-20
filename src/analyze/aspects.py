"""
The fixed set of pain-point aspects used by the trained classifier.

key: (display name, dashboard category, description used when labeling data)

The descriptions double as labeling instructions, so tie-break rules live here:
ambiguous sentences were the main source of disagreement between human and model
labels, and clear rules matter more than adding more classes.
"""

from pathlib import Path

MODEL_PATH = Path("models/aspect_classifier.joblib")

ASPECTS = {
    "crashes":        ("Crashes & lost progress", "Technical",
                       "the game crashes, freezes, won't launch, or progress and saves are lost"),
    "bugs":           ("Bugs & glitches", "Technical",
                       "bugs, glitches, broken quests or features, things not working as intended"),
    "performance":    ("Performance & loading", "Technical",
                       "low frame rate, stuttering, lag, long loading times, poor optimization, "
                       "hardware demands (loading screens count here, not as visuals)"),
    "multiplayer":    ("Multiplayer & servers", "Technical",
                       "servers, netcode, matchmaking, co-op, cheaters, online connectivity"),
    "controls":       ("Controls & camera", "UX",
                       "awkward controls, keybindings, controller or mouse support, input lag, "
                       "camera angle or camera movement"),
    "ui":             ("UI & menus", "UX",
                       "user interface, menus, HUD, inventory management, readability, "
                       "quality-of-life features"),
    "onboarding":     ("Tutorial & onboarding", "UX",
                       "poor tutorial, unclear mechanics, steep learning curve, having to look up "
                       "information outside the game"),
    "difficulty":     ("Difficulty & balance", "Gameplay",
                       "too hard or too easy, unfair, badly balanced, overpowered or useless options, "
                       "randomness or RNG feels bad"),
    "mechanics":      ("Core mechanics & combat", "Gameplay",
                       "combat, movement, AI or the core gameplay loop feel bad, slow or shallow "
                       "(about how it plays, not how hard it is)"),
    "grind":          ("Grind & repetition", "Gameplay",
                       "repetitive, grindy, tedious, padded content, doing the same thing over and "
                       "over (a plain statement of hours played is not_actionable instead)"),
    "content":        ("Length & amount of content", "Content",
                       "too short, not enough content, little replayability, empty world, "
                       "feels unfinished at release"),
    "story":          ("Story & characters", "Content",
                       "story, plot, ending, writing, dialogue, lore, characters, companions, "
                       "character creation"),
    "visuals":        ("Presentation (visuals & audio)", "Content",
                       "graphics, art style, animations, visual effects, sound, music, voice acting"),
    "price":          ("Price & monetization", "Business",
                       "too expensive, not worth the money, refunds, microtransactions, DLC, "
                       "pay to win, battle passes"),
    "support":        ("Updates & developer support", "Business",
                       "abandoned game, lack of or bad patches after release, developer communication "
                       "or decisions (post-release handling, not the state of the content itself)"),
    "not_actionable": ("General dislike / not actionable", "Not actionable",
                       "vague dislike, not fun, genre not for me, plain playtime statements, "
                       "off-topic remarks, jokes"),
}

NON_COMPLAINT = "not_complaint"
NON_COMPLAINT_DESC = ("not a complaint at all: praise, neutral description, story of what the "
                      "player did, a question, or a remark about something other than the game")

# Every label the classifier can predict: the aspects plus "not a complaint", which lets
# the model filter out sentences the sentiment step wrongly kept.
LABEL_CHOICES = {**{k: v[2] for k, v in ASPECTS.items()}, NON_COMPLAINT: NON_COMPLAINT_DESC}

# Aspects that used to be separate. Labels are mapped through this, so data labeled
# with the older, finer set stays usable without relabeling.
ALIASES = {
    "camera": "controls",
    "saves": "crashes",
    "characters": "story",
    "audio": "visuals",
    "monetization": "price",
}


def canonical(aspect: str) -> str:
    """Map any label (old or new) to its current aspect key."""
    return ALIASES.get(aspect, aspect)
