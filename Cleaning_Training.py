# -*- coding: utf-8 -*-
"""IMDb Hybrid Movie Recommendation AI (2023-2026)"""

import os, re, json, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import pickle
import kagglehub

# --- Download Dataset ---
path = kagglehub.dataset_download("tiagoadrianunes/imdb-top-5000-movies")
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))

# --- Filter and clean 2023-2026 movies ---
df = df[['primaryTitle','genres','directors','writers','averageRating','startYear','runtimeMinutes','numVotes']]
df = df[(df['startYear'] >= 2023) & (df['startYear'] <= 2026)]
df = df.dropna().drop_duplicates()
df = df[df['numVotes'] > 10000].reset_index(drop=True)

# --- Safe poster filenames ---
def safe_filename(title, year):
    title = title.strip()
    filename = re.sub(r'[^A-Za-z0-9]+', '_', title)
    filename = filename.strip('_')
    filename += f"_{year}.jpg"
    return filename

df['poster'] = df.apply(lambda row: safe_filename(row['primaryTitle'], row['startYear']), axis=1)

# --- Text + numeric features for hybrid AI ---
df['text_features'] = (
    df['genres'].str.lower().str.replace(',', ' ') + " " +
    df['directors'].str.lower().str.strip() + " " +
    df['writers'].str.lower().str.strip()
)

numeric_features = df[['runtimeMinutes','numVotes','averageRating','startYear']].copy()
scaler = MinMaxScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
text_vectors = vectorizer.fit_transform(df['text_features'])

X = hstack([text_vectors, csr_matrix(numeric_features_scaled)])

# --- Train Hybrid Recommendation Model ---
model = NearestNeighbors(n_neighbors=15, metric='cosine', algorithm='brute')
model.fit(X)

# --- Save AI and preprocessing objects ---
pickle.dump(model, open("movie_model_hybrid.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer_hybrid.pkl","wb"))
pickle.dump(scaler, open("scaler_hybrid.pkl","wb"))

# --- Prepare JSON for JS ---
df.rename(columns={
    'primaryTitle':'title',
    'averageRating':'rating',
    'startYear':'year'
}, inplace=True)

movies_json = df.to_dict(orient='records')
with open("movies.json", "w") as f:
    json.dump({"movies": movies_json}, f, indent=4)

print("✅ Dataset cleaned, AI trained, JSON & posters ready!")

# --- Generate AI Recommendations ---
import numpy as np

recommendations = []

for i in range(X.shape[0]):
    distances, indices = model.kneighbors(X[i], n_neighbors=6)  # include self
    rec_titles = [df.iloc[idx]['title'] for idx in indices[0] if idx != i]  # exclude self
    recommendations.append(rec_titles)

df['recommendations'] = recommendations

# Save updated JSON
movies_json = df.to_dict(orient='records')
with open("movies.json", "w") as f:
    json.dump({"movies": movies_json}, f, indent=4)

print("✅ JSON updated with AI recommendations!")