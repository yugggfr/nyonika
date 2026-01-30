import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "data/mood_log.csv"
GRAPH_DIR = "data/graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    print("No mood data available.")
    exit()

df = pd.read_csv(LOG_FILE)

if df.empty:
    print("Mood log exists but contains no rows.")
    exit()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

print(f"Loaded {len(df)} mood records.\n")

counts = df["emotion"].value_counts()

print("Emotion counts:")
for e, c in counts.items():
    print(f"  {e}: {c}")

print(f"\nMost common emotion: {counts.idxmax()}")
print("-" * 40)

# -------- Frequency --------

plt.figure(figsize=(8, 5))
counts.plot(kind="bar")
plt.title("Emotion Frequency")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/emotion_frequency.png")
plt.show()

# -------- Timeline --------

plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["emotion"], marker="o")
plt.title("Emotion Timeline")
plt.xlabel("Time")
plt.ylabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/emotion_timeline.png")
plt.show()

# -------- Hourly Activity --------

df["hour"] = df["time"].dt.hour
hourly = df.groupby("hour")["emotion"].count()

plt.figure(figsize=(8, 4))
hourly.plot(marker="o")
plt.title("Logs per Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/hourly_activity.png")
plt.show()

# -------- Mood Score --------

score_map = {
    "Happy": 2,
    "Neutral": 1,
    "Surprise": 1,
    "Sad": -1,
    "Fear": -1,
    "Disgust": -2,
    "Angry": -2
}

df["score"] = df["emotion"].map(score_map)

plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["score"], marker="o")
plt.axhline(0, linestyle="--", alpha=0.5)
plt.title("Mood Trend")
plt.xlabel("Time")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/mood_trend.png")
plt.show()

print("\nGraphs generated and saved.")
