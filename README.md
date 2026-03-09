# Wine Buddy

Wine analysis project featuring similarity search, recommendations, and rule-based food pairing — all powered by machine learning.

## Features

1. **Data Gathering** — Scraper for collecting wine data (taste profiles, ratings, grapes, regions) from Vivino's public API
2. **Data Preparation** — Parse nested JSON fields, handle missing values, feature engineering
3. **Find Similar Wines** — KD-Tree nearest neighbor search on wine embeddings (PCA/LDA/NCA)
4. **Wine Recommendations** — Collaborative filtering via scraped user ratings
5. **Wine & Food Pairing** — Rule-based engine matching wine taste profiles to food characteristics

## Setup

```bash
uv sync
```

## Usage

```bash
uv run jupyter lab
```

Then open `wine_buddy.ipynb`.

### Scraping your own data

No pre-scraped data is included. The notebook contains scraper functions you can run to collect your own dataset. Please respect Vivino's Terms of Service and rate-limit your requests.

## Credits

- Food pairing logic inspired by [RoaldSchuring/wine_food_pairing](https://github.com/RoaldSchuring/wine_food_pairing)
- Wine sweetness references: [Wine Folly](https://winefolly.com/deep-dive/the-prosecco-wine-guide/), [Handy Wine Guide](https://handywineguide.com/wine-sweetness-chart/)
