"""
hybrid.py
---------
Hybrid Recommendation Engine.
Combines content-based and collaborative filtering scores
using a weighted average controlled by an alpha parameter.

  hybrid_score = alpha × content_score + (1 - alpha) × collab_score
"""

import pandas as pd
import numpy as np


def get_hybrid_recommendations(product_name: str,
                                 user_id: str,
                                 content_model: dict,
                                 collab_model: dict,
                                 catalogue: pd.DataFrame,
                                 alpha: float = 0.5,
                                 top_n: int = 10,
                                 category_filter: str = None,
                                 min_rating: float = 1.0,
                                 max_price: float = None) -> pd.DataFrame:
    """
    Generate hybrid recommendations for a (user, product) pair.

    Parameters
    ----------
    product_name    : str          – product used for content-based scoring
    user_id         : str          – user used for collaborative scoring
    content_model   : dict         – loaded content model
    collab_model    : dict         – loaded collab model
    catalogue       : pd.DataFrame – product catalogue
    alpha           : float        – weight for content score (0 = full collab,
                                     1 = full content)
    top_n           : int          – number of recommendations
    category_filter : str          – optional category filter
    min_rating      : float        – minimum average rating
    max_price       : float        – maximum price

    Returns
    -------
    pd.DataFrame – top-N hybrid recommended products
    """
    from src.content_filter import get_content_recommendations
    from src.collab_filter import get_collab_recommendations

    # Get a wider pool from both engines
    fetch_n = min(top_n * 5, 50)

    content_recs = get_content_recommendations(
        product_name, content_model, top_n=fetch_n,
        category_filter=category_filter, min_rating=min_rating,
        max_price=max_price,
    )
    collab_recs = get_collab_recommendations(
        user_id, collab_model, catalogue, top_n=fetch_n,
        category_filter=category_filter,
    )

    if content_recs.empty and collab_recs.empty:
        return pd.DataFrame()

    # Normalise scores to [0, 1] for fair combination
    def _norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    # Build score maps keyed by product_id
    content_score_map: dict = {}
    if not content_recs.empty:
        content_recs["norm_content"] = _norm(content_recs["similarity_score"])
        content_score_map = dict(
            zip(content_recs["product_id"], content_recs["norm_content"])
        )

    collab_score_map: dict = {}
    if not collab_recs.empty:
        collab_recs["norm_collab"] = _norm(collab_recs["predicted_score"])
        collab_score_map = dict(
            zip(collab_recs["product_id"], collab_recs["norm_collab"])
        )

    # Union of candidate products
    all_pids = set(content_score_map) | set(collab_score_map)

    rows = []
    for pid in all_pids:
        c_score = content_score_map.get(pid, 0.0)
        u_score = collab_score_map.get(pid, 0.0)
        hybrid  = alpha * c_score + (1 - alpha) * u_score
        rows.append({"product_id": pid, "hybrid_score": round(hybrid, 4),
                     "content_score": round(c_score, 4),
                     "collab_score": round(u_score, 4)})

    hybrid_df = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False)

    # Merge with catalogue for full metadata
    result = hybrid_df.merge(catalogue, on="product_id", how="left")

    # Apply remaining filters
    result = result[result["avg_rating"] >= min_rating]
    if max_price:
        result = result[result["price"] <= max_price]

    return result.head(top_n).reset_index(drop=True)
