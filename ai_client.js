(function () {
  const PROFILE_KEY = "imdb_ai_profile_v1";

  function loadProfile() {
    try {
      const raw = localStorage.getItem(PROFILE_KEY);
      if (!raw) return { views: {}, genres: {}, directors: {}, searches: {}, recent: [] };
      const parsed = JSON.parse(raw);
      return {
        views: parsed.views || {},
        genres: parsed.genres || {},
        directors: parsed.directors || {},
        searches: parsed.searches || {},
        recent: parsed.recent || [],
      };
    } catch {
      return { views: {}, genres: {}, directors: {}, searches: {}, recent: [] };
    }
  }

  function saveProfile(profile) {
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } catch {
      // Ignore quota/storage errors.
    }
  }

  function bumpMap(map, key, amount) {
    if (!key) return;
    map[key] = (map[key] || 0) + amount;
  }

  function recordMovieView(movie) {
    if (!movie) return;
    const p = loadProfile();
    const title = movie.title || "";
    bumpMap(p.views, title, 1);

    (movie.genres || "").split(",").map(g => g.trim()).filter(Boolean).forEach(g => bumpMap(p.genres, g, 1));
    (movie.directors || "").split(",").map(d => d.trim()).filter(Boolean).forEach(d => bumpMap(p.directors, d, 1));

    p.recent = [title, ...p.recent.filter(x => x !== title)].slice(0, 30);
    saveProfile(p);
  }

  function recordSearch(query) {
    const q = (query || "").trim().toLowerCase();
    if (!q) return;
    const p = loadProfile();
    bumpMap(p.searches, q, 1);
    saveProfile(p);
  }

  function personalizedBoost(movie, p) {
    if (!movie) return 0;
    let boost = 0;
    boost += (p.views[movie.title] || 0) * 0.18;
    (movie.genres || "").split(",").map(g => g.trim()).filter(Boolean).forEach(g => {
      boost += (p.genres[g] || 0) * 0.08;
    });
    (movie.directors || "").split(",").map(d => d.trim()).filter(Boolean).forEach(d => {
      boost += (p.directors[d] || 0) * 0.06;
    });
    if (p.recent.includes(movie.title)) boost += 0.5;
    return boost;
  }

  function mapMoviesByTitle(movies) {
    const byTitle = new Map();
    movies.forEach(m => byTitle.set((m.title || "").toLowerCase(), m));
    return byTitle;
  }

  async function aiSearch(query, limit) {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    if (!res.ok) throw new Error("AI search unavailable");
    return res.json();
  }

  function localSearch(movies, query, limit) {
    const q = (query || "").toLowerCase().trim();
    if (!q) return [];
    return movies
      .map(m => {
        const title = (m.title || "").toLowerCase();
        let score = 0;
        if (title.startsWith(q)) score = 1.0;
        else if (title.split(/[^a-z0-9]+/).some(w => w.startsWith(q))) score = 0.8;
        else if (q.length >= 3 && title.includes(q)) score = 0.55;
        return { movie: m, baseScore: score };
      })
      .filter(x => x.baseScore > 0)
      .sort((a, b) => b.baseScore - a.baseScore || (b.movie.rating || 0) - (a.movie.rating || 0))
      .slice(0, limit)
      .map(x => x.movie);
  }

  async function getAISearchResults(movies, query, limit = 20) {
    const q = (query || "").trim();
    if (!q) return [];
    const p = loadProfile();

    try {
      const payload = await aiSearch(q, limit);
      const byTitle = mapMoviesByTitle(movies);
      const merged = (payload.results || [])
        .map(r => {
          const movie = byTitle.get((r.title || "").toLowerCase());
          if (!movie) return null;
          return {
            movie,
            baseScore: Number(r.score || 0),
          };
        })
        .filter(Boolean)
        .sort((a, b) => {
          const as = a.baseScore + personalizedBoost(a.movie, p);
          const bs = b.baseScore + personalizedBoost(b.movie, p);
          return bs - as;
        })
        .slice(0, limit)
        .map(x => x.movie);
      if (merged.length) return merged;
    } catch {
      // Fallback below.
    }

    const local = localSearch(movies, q, limit);
    return [...local].sort((a, b) => (personalizedBoost(b, p) - personalizedBoost(a, p)));
  }

  async function getAISuggestions(movies, query, limit = 10) {
    return getAISearchResults(movies, query, limit);
  }

  window.AIClient = {
    recordMovieView,
    recordSearch,
    getAISearchResults,
    getAISuggestions,
    personalizedBoost,
    getProfile: loadProfile,
  };
})();
