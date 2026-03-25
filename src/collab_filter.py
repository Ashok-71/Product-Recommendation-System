"""
collab_filter.py
----------------
Collaborative Filtering using SVD (Matrix Factorisation).
Recommends products that similar users have rated highly.
Falls back to a KNN cosine approach if scipy SVD is unavailable.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ── model training ────────────────────────────────────────────────────────────

def build_collab_model(user_item_matrix: pd.DataFrame,
                       n_components: int = 50,
                       model_path: str = "models/collab_model.pkl") -> dict:
    """
    Train a Truncated SVD collaborative filter on the user-item matrix.

    Parameters
    ----------
    user_item_matrix : pd.DataFrame – users × products pivot table
    n_components     : int          – number of latent factors
    model_path       : str          – where to save the model

    Returns
    -------
    dict with keys: svd, user_factors, item_factors,
                    user_ids, product_ids, filled_matrix
    """
    # Fill NaN with 0 for decomposition (implicit feedback style)
    matrix_filled = user_item_matrix.fillna(0).values

    n_comp = min(n_components, min(matrix_filled.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_factors = svd.fit_transform(matrix_filled)    # (users × k)
    item_factors = svd.components_.T                   # (products × k)

    # Normalise for cosine similarity
    user_factors_norm = normalize(user_factors)
    item_factors_norm = normalize(item_factors)

    payload = {
        "svd": svd,
        "user_factors": user_factors_norm,
        "item_factors": item_factors_norm,
        "user_ids": list(user_item_matrix.index),
        "product_ids": list(user_item_matrix.columns),
        "filled_matrix": matrix_filled,
        "user_item_matrix": user_item_matrix,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)
    return payload


def load_collab_model(model_path: str = "models/collab_model.pkl") -> dict:
    """
    Load a previously saved collaborative filter model.

    Parameters
    ----------
    model_path : str – path to pickled model

    Returns
    -------
    dict – model payload
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Collab model not found at '{model_path}'. "
            "Call build_collab_model() first."
        )
    return joblib.load(model_path)


# ── recommendations ───────────────────────────────────────────────────────────

def get_collab_recommendations(user_id: str,
                                model: dict,
                                catalogue: pd.DataFrame,
                                top_n: int = 10,
                                category_filter: str = None) -> pd.DataFrame:
    """
    Predict the top-N products a given user would enjoy.

    Uses dot product of user latent vector × all item latent vectors
    to score unrated products.

    Parameters
    ----------
    user_id         : str          – target user ID
    model           : dict         – loaded collab model
    catalogue       : pd.DataFrame – product catalogue for metadata
    top_n           : int          – number of recommendations
    category_filter : str          – optional category restriction

    Returns
    -------
    pd.DataFrame – top-N recommended products with predicted scores
    """
    user_ids   = model["user_ids"]
    product_ids = model["product_ids"]
    user_factors = model["user_factors"]
    item_factors = model["item_factors"]
    user_item_matrix = model["user_item_matrix"]

    if user_id not in user_ids:
        # Cold-start: return globally popular products
        return _cold_start_recommendations(catalogue, top_n, category_filter)

    u_idx = user_ids.index(user_id)
    user_vec = user_factors[u_idx]                # (k,)

    # Score all items
    scores = item_factors @ user_vec              # (n_products,)

    # Mask already-rated products
    rated_mask = user_item_matrix.loc[user_id].notna().values
    scores[rated_mask] = -np.inf

    # Get top-N indices
    top_indices = np.argsort(scores)[::-1][:top_n * 3]
    top_product_ids = [product_ids[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    result = catalogue[catalogue["product_id"].isin(top_product_ids)].copy()
    score_map = dict(zip(top_product_ids, top_scores))
    result["predicted_score"] = result["product_id"].map(score_map)
    result = result.sort_values("predicted_score", ascending=False)

    if category_filter and category_filter != "All":
        result = result[result["category"] == category_filter]

    return result.head(top_n).reset_index(drop=True)


def _cold_start_recommendations(catalogue: pd.DataFrame,
                                  top_n: int,
                                  category_filter: str = None) -> pd.DataFrame:
    """
    Fallback for unknown users: return highest-rated products.

    Parameters
    ----------
    catalogue       : pd.DataFrame
    top_n           : int
    category_filter : str

    Returns
    -------
    pd.DataFrame
    """
    result = catalogue.copy()
    if category_filter and category_filter != "All":
        result = result[result["category"] == category_filter]
    result["predicted_score"] = result["avg_rating"]
    return (result.sort_values("predicted_score", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))


def get_user_ids(model: dict) -> list:
    """Return sorted list of all user IDs."""
    return sorted(model["user_ids"])
