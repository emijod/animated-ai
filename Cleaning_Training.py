# -*- coding: utf-8 -*-
"""IMDb Hybrid Movie Recommendation AI Training (notebook-style script)."""

# ============================================================
# Imports
# ============================================================
import json
import os
import pickle
import re

import kagglehub
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Download Dataset
# ============================================================
path = kagglehub.dataset_download("tiagoadrianunes/imdb-top-5000-movies")
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))

print("Dataset path:", path)
print("Files:", files)
print("Original shape:", df.shape)


# ============================================================
# Data Cleaning
# ============================================================
# Keep only columns needed by app + model.
df = df[
    [
        "primaryTitle",
        "genres",
        "directors",
        "writers",
        "averageRating",
        "startYear",
        "runtimeMinutes",
        "numVotes",
    ]
]

# Keep recent movies and remove noisy rows.
df = df[(df["startYear"] >= 2023) & (df["startYear"] <= 2026)]
df = df.dropna().drop_duplicates()
df = df[df["numVotes"] > 10000].reset_index(drop=True)

print("Cleaned shape:", df.shape)


# ============================================================
# Poster Filename Formatting
# ============================================================
def safe_filename(title, year):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(title).strip()).strip("_")
    return f"{cleaned}_{int(year)}.jpg"


df["poster"] = df.apply(
    lambda row: safe_filename(row["primaryTitle"], row["startYear"]), axis=1
)


# ============================================================
# Preprocessing
# ============================================================
# Create text feature field used by TF-IDF.
df["text_features"] = (
    df["genres"].str.lower().str.replace(",", " ", regex=False)
    + " "
    + df["directors"].str.lower().str.strip()
    + " "
    + df["writers"].str.lower().str.strip()
)

# Scale numeric features to 0-1.
numeric_features = df[
    ["runtimeMinutes", "numVotes", "averageRating", "startYear"]
].copy()
scaler = MinMaxScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Convert text to vectors.
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
text_vectors = vectorizer.fit_transform(df["text_features"])

# Combine text + numeric into hybrid feature matrix.
X = hstack([text_vectors, csr_matrix(numeric_features_scaled)])


# ============================================================
# Train Model
# ============================================================
model = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute")
model.fit(X)


# ============================================================
# Rename Columns for Frontend/Server Compatibility
# ============================================================
df = df.rename(
    columns={
        "primaryTitle": "title",
        "averageRating": "rating",
        "startYear": "year",
    }
)


# ============================================================
# Generate Recommendations
# ============================================================
recommendations = []
for i in range(X.shape[0]):
    distances, indices = model.kneighbors(X[i], n_neighbors=6)  # include self
    rec_titles = [df.iloc[idx]["title"] for idx in indices[0] if idx != i][:5]
    recommendations.append(rec_titles)

df["recommendations"] = recommendations


# ============================================================
# Save Model Artifacts + movies.json
# ============================================================
pickle.dump(model, open("movie_model_hybrid.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer_hybrid.pkl", "wb"))
pickle.dump(scaler, open("scaler_hybrid.pkl", "wb"))

movies_json = df.to_dict(orient="records")
with open("movies.json", "w", encoding="utf-8") as f:
    json.dump({"movies": movies_json}, f, indent=4)


# ============================================================
# Offline Metrics
# ============================================================
def genre_set(value):
    return {g.strip().lower() for g in str(value).split(",") if g.strip()}


n = X.shape[0]
sims = []
jaccards = []
recommended_indices = set()
all_genres = [genre_set(g) for g in df["genres"]]

for i in range(n):
    distances, indices = model.kneighbors(X[i], n_neighbors=min(6, n))
    rec_idxs = [idx for idx in indices[0] if idx != i][:5]
    rec_dists = [dist for idx, dist in zip(indices[0], distances[0]) if idx != i][:5]

    sims.extend([1.0 - float(d) for d in rec_dists])
    recommended_indices.update(rec_idxs)

    src = all_genres[i]
    for ridx in rec_idxs:
        tgt = all_genres[ridx]
        union = len(src | tgt)
        inter = len(src & tgt)
        jaccards.append((inter / union) if union else 0.0)

metrics = {
    "movies_used": int(n),
    "k": 5,
    "avg_neighbor_similarity_at_k": float(np.mean(sims)) if sims else 0.0,
    "avg_genre_jaccard_at_k": float(np.mean(jaccards)) if jaccards else 0.0,
    "catalog_coverage_at_k": float(len(recommended_indices) / n) if n else 0.0,
}

with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


# ============================================================
# Done
# ============================================================
print("Dataset cleaned and model trained.")
print("Artifacts saved: movie_model_hybrid.pkl, vectorizer_hybrid.pkl, scaler_hybrid.pkl")
print("movies.json updated with recommendations.")
print("Offline metrics:")
for key, value in metrics.items():
    print(f"  - {key}: {value}")
