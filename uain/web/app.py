"""Flask application — Wine Buddy web portal."""

from __future__ import annotations

from flask import Flask, render_template, request

from uain.web.services import find_similar_wines

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template("find_similar.html")


@app.route("/find-similar", methods=["GET", "POST"])
def find_similar() -> str:
    results = None
    query = ""
    k = 5
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        k = int(request.form.get("k", 5))
        k = max(1, min(k, 50))

        if not query:
            error = "Please enter a wine name."
        else:
            results = find_similar_wines(query, k=k)
            if results["query_wine"] is None:
                error = f'No wines found matching "{query}". Try a different name.'
                results = None

    return render_template(
        "find_similar.html",
        results=results,
        query=query,
        k=k,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
