"""Flask application — Wine Buddy web portal."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from uain.web.services import (
    find_similar_wines,
    get_all_foods,
    pair_wine_to_food,
    search_wines_by_name,
)

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template("find_similar.html")


@app.route("/find-similar", methods=["GET", "POST"])
def find_similar() -> str:
    results = None
    candidates = None
    query = ""
    k = 5
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        k = int(request.form.get("k", 5))
        k = max(1, min(k, 50))
        wine_idx = request.form.get("wine_idx")

        if wine_idx is not None:
            # Step 2: user picked a wine — find similar by embeddings
            results = find_similar_wines(int(wine_idx), k=k)
            if results["query_wine"] is None:
                error = "Wine not found. Please search again."
                results = None
        elif query:
            # Step 1: fuzzy name search — show candidates to pick from
            candidates = search_wines_by_name(query, limit=20)
            if not candidates:
                error = f'No wines found matching "{query}". Try a different name.'
        else:
            error = "Please enter a wine name."

    return render_template(
        "find_similar.html",
        results=results,
        candidates=candidates,
        query=query,
        k=k,
        error=error,
    )


@app.route("/api/wines/search")
def api_wine_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    return jsonify(search_wines_by_name(q, limit=15))


@app.route("/pair-to-food", methods=["GET", "POST"])
def pair_to_food() -> str:
    results = None
    query = ""
    k = 10
    error = None
    all_foods = get_all_foods()

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        k = int(request.form.get("k", 10))
        k = max(1, min(k, 100))

        if query:
            results = pair_wine_to_food(query, k=k)
            if not results["wines"]:
                error = f'No wines matched the pairing rules for "{query}". Try a different food.'
                results = None
        else:
            error = "Please select a food."

    return render_template(
        "pair_to_food.html",
        results=results,
        all_foods=all_foods,
        query=query,
        k=k,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
