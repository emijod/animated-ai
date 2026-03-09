import json
import pickle
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
MOVIES_JSON = ROOT / "movies.json"
MODEL_PATH = ROOT / "movie_model_hybrid.pkl"
VECTORIZER_PATH = ROOT / "vectorizer_hybrid.pkl"
SCALER_PATH = ROOT / "scaler_hybrid.pkl"


def load_movies():
    payload = json.loads(MOVIES_JSON.read_text(encoding="utf-8"))
    movies = payload.get("movies", [])
    if not isinstance(movies, list):
        return []
    return movies


def build_feature_frame(movies):
    df = pd.DataFrame(movies)

    # Normalize expected columns for hybrid model features.
    df["title"] = df.get("title", "").fillna("")
    df["genres"] = df.get("genres", "").fillna("")
    df["directors"] = df.get("directors", "").fillna("")
    df["writers"] = df.get("writers", "").fillna("")
    df["runtimeMinutes"] = pd.to_numeric(df.get("runtimeMinutes", 0), errors="coerce").fillna(0)
    df["numVotes"] = pd.to_numeric(df.get("numVotes", 0), errors="coerce").fillna(0)
    # Keep both naming variants; scaler was fit on averageRating/startYear.
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0)
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0)
    df["averageRating"] = pd.to_numeric(df.get("averageRating", df["rating"]), errors="coerce").fillna(0)
    df["startYear"] = pd.to_numeric(df.get("startYear", df["year"]), errors="coerce").fillna(0)

    df["text_features"] = (
        df["genres"].str.lower().str.replace(",", " ", regex=False) + " "
        + df["directors"].str.lower().str.strip() + " "
        + df["writers"].str.lower().str.strip()
    )
    return df


def build_matrix(df, vectorizer, scaler):
    numeric_features = df[["runtimeMinutes", "numVotes", "averageRating", "startYear"]]
    numeric_scaled = scaler.transform(numeric_features)
    text_vectors = vectorizer.transform(df["text_features"])
    return hstack([text_vectors, csr_matrix(numeric_scaled)])


def normalize_title_key(title):
    return (title or "").strip().lower()


class AIState:
    def __init__(self):
        self.movies = load_movies()
        self.df = build_feature_frame(self.movies)
        self.model = pickle.loads(MODEL_PATH.read_bytes())
        self.vectorizer = pickle.loads(VECTORIZER_PATH.read_bytes())
        self.scaler = pickle.loads(SCALER_PATH.read_bytes())
        self.text_vectors = self.vectorizer.transform(self.df["text_features"])
        self.X = build_matrix(self.df, self.vectorizer, self.scaler)
        self.title_to_idx = {
            normalize_title_key(row["title"]): idx for idx, row in self.df.iterrows()
        }

    def recommend(self, title, k=10):
        key = normalize_title_key(title)
        idx = self.title_to_idx.get(key)
        if idx is None:
            return []

        n_neighbors = min(max(int(k) + 1, 2), len(self.df))
        distances, indices = self.model.kneighbors(self.X[idx], n_neighbors=n_neighbors)

        recs = []
        for dist, rec_idx in zip(distances[0], indices[0]):
            if int(rec_idx) == int(idx):
                continue
            movie = self.movies[int(rec_idx)]
            recs.append(
                {
                    "title": movie.get("title", ""),
                    "poster": movie.get("poster", ""),
                    "score": float(1.0 - dist),
                }
            )
            if len(recs) >= int(k):
                break
        return recs

    def search(self, query, limit=10):
        q = (query or "").strip().lower()
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        sem = cosine_similarity(q_vec, self.text_vectors).ravel()

        scored = []
        for i, movie in enumerate(self.movies):
            title = (movie.get("title") or "").lower()
            if not title:
                continue

            lex = 0.0
            if title.startswith(q):
                lex = 1.0
            elif any(word.startswith(q) for word in title.replace(":", " ").split()):
                lex = 0.8
            elif len(q) >= 3 and q in title:
                lex = 0.55
            elif len(q) < 3:
                continue

            rating = float(movie.get("rating") or 0.0) / 10.0
            score = (0.62 * float(sem[i])) + (0.30 * lex) + (0.08 * rating)
            scored.append((score, movie))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, movie in scored[: max(1, min(50, int(limit)))]:
            out.append(
                {
                    "title": movie.get("title", ""),
                    "year": movie.get("year", ""),
                    "poster": movie.get("poster", ""),
                    "score": float(score),
                }
            )
        return out


STATE = AIState()


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommendations":
            params = parse_qs(parsed.query)
            title = (params.get("title") or [""])[0]
            k_raw = (params.get("k") or ["10"])[0]
            try:
                k = max(1, min(20, int(k_raw)))
            except ValueError:
                k = 10
            recs = STATE.recommend(title, k=k)
            self._send_json({"title": title, "recommendations": recs})
            return

        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            limit_raw = (params.get("limit") or ["10"])[0]
            try:
                limit = max(1, min(50, int(limit_raw)))
            except ValueError:
                limit = 10
            hits = STATE.search(query, limit=limit)
            self._send_json({"q": query, "results": hits})
            return

        if parsed.path == "/api/health":
            self._send_json({"ok": True, "movies": len(STATE.movies)})
            return

        return super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AppHandler)
    print("AI server running on http://127.0.0.1:8000")
    print("Open home page at http://127.0.0.1:8000/home.html")
    server.serve_forever()
