"""
The fixed set of pain-point aspects used by the trained classifier.

key: (display name, dashboard category, description used when labeling data)
"""

from pathlib import Path

MODEL_PATH = Path("models/aspect_classifier.joblib")

ASPECTS = {
    "crashes":        ("Crashes & freezes", "Technical",
                       "the game crashes, freezes, won't launch or closes unexpectedly"),
    "bugs":           ("Bugs & glitches", "Technical",
                       "bugs, glitches, broken quests or features, things not working as intended"),
    "performance":    ("Performance & optimization", "Technical",
                       "low frame rate, stuttering, lag, long loading times, poor optimization, hardware demands"),
    "saves":          ("Save & progress loss", "Technical",
                       "lost progress, corrupted saves, bad checkpoint or save system"),
    "multiplayer":    ("Multiplayer & servers", "Technical",
                       "servers, netcode, matchmaking, co-op, cheaters, online connectivity"),
    "controls":       ("Controls", "UX",
                       "awkward controls, keybindings, controller or mouse support, input responsiveness"),
    "camera":         ("Camera", "UX",
                       "camera angle, camera movement, field of view, camera control"),
    "ui":             ("UI & menus", "UX",
                       "user interface, menus, HUD, inventory management, readability, quality-of-life features"),
    "onboarding":     ("Tutorial & onboarding", "UX",
                       "poor tutorial, unclear mechanics, steep learning curve, not explained to the player"),
    "difficulty":     ("Difficulty & balance", "Gameplay",
                       "too hard or too easy, unfair, unbalanced, randomness or RNG feels bad"),
    "mechanics":      ("Core mechanics & combat", "Gameplay",
                       "combat, movement, core gameplay loop or mechanics feel bad, slow or shallow"),
    "grind":          ("Grind & repetition", "Gameplay",
                       "repetitive, grindy, tedious, padding, same thing over and over"),
    "content":        ("Length & amount of content", "Content",
                       "too short, not enough content, little replayability, feels unfinished or early access"),
    "story":          ("Story & writing", "Content",
                       "story, plot, ending, writing, dialogue, lore, quests' narrative"),
    "characters":     ("Characters", "Content",
                       "characters, companions, NPCs, character creation or customization"),
    "visuals":        ("Graphics & visuals", "Content",
                       "graphics, art style, animations, visual effects look bad"),
    "audio":          ("Audio & music", "Content",
                       "sound, music, voice acting, audio mixing"),
    "price":          ("Price & value", "Business",
                       "too expensive, not worth the price, refunds, value for money"),
    "monetization":   ("Monetization & DLC", "Business",
                       "microtransactions, DLC, pay to win, battle pass, cosmetics shop"),
    "support":        ("Updates & developer support", "Business",
                       "abandoned game, lack of updates, bad patches, developer communication or decisions"),
    "not_actionable": ("General dislike / not actionable", "Not actionable",
                       "vague dislike, not fun, genre not for me, playtime statements, off-topic, jokes"),
}
