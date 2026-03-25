"""
evaluation.py
-------------
Model evaluation utilities.
Computes RMSE, MAE, Precision@K, and Recall@K
for both the collaborative filter and content-based model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ── collaborative filter evaluation ──────────────────────────────────────────

def evaluate_collab(df: pd.DataFrame,
                    collab_model: dict,
                    test_size: float = 0.2,
                    seed: int = 42) -> dict:
    """
    Split ratings into train/test and evaluate predicted vs actual ratings.

    Parameters
    ----------
    df            : pd.DataFrame – full cleaned dataframe
    collab_model  : dict         – loaded collab model
    test_size     : float        – fraction of rows used for testing
    seed          : int

    Returns
    -------
    dict with keys: rmse, mae, n_test_samples
    """
    test_df = df.sample(frac=test_size, random_state=seed)

    user_factors = collab_model["user_factors"]
    item_factors = collab_model["item_factors"]
    user_ids     = collab_model["user_ids"]
    product_ids  = collab_model["product_ids"]

    actual, predicted = [], []

    for _, row in test_df.iterrows():
        uid = row["user_id"]
        pid = row["product_id"]
        if uid in user_ids and pid in product_ids:
            u_idx = user_ids.index(uid)
            p_idx = product_ids.index(pid)
            pred = float(user_factors[u_idx] @ item_factors[p_idx])
            # Re-scale from latent space to [1, 5]
            pred = np.clip(pred * 5, 1, 5)
            actual.append(row["rating"])
            predicted.append(pred)

    if not actual:
        return {"rmse": None, "mae": None, "n_test_samples": 0}

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "n_test_samples": len(actual),
    }


# ── precision & recall @ K ────────────────────────────────────────────────────

def precision_recall_at_k(df: pd.DataFrame,
                           collab_model: dict,
                           catalogue: pd.DataFrame,
                           k: int = 10,
                           relevance_threshold: float = 3.5,
                           n_users: int = 50,
                           seed: int = 42) -> dict:
    """
    Compute mean Precision@K and Recall@K across a sample of users.

    A product is considered 'relevant' if actual rating >= relevance_threshold.

    Parameters
    ----------
    df                  : pd.DataFrame
    collab_model        : dict
    catalogue           : pd.DataFrame
    k                   : int   – cutoff rank
    relevance_threshold : float – minimum rating to be 'relevant'
    n_users             : int   – number of users to sample
    seed                : int

    Returns
    -------
    dict with keys: precision_at_k, recall_at_k, k
    """
    from src.collab_filter import get_collab_recommendations

    np.random.seed(seed)
    all_users = collab_model["user_ids"]
    sample_users = np.random.choice(all_users,
                                     min(n_users, len(all_users)),
                                     replace=False)

    precisions, recalls = [], []

    for uid in sample_users:
        # Ground truth: products this user actually rated ≥ threshold
        user_ratings = df[df["user_id"] == uid]
        relevant_pids = set(
            user_ratings[user_ratings["rating"] >= relevance_threshold]["product_id"]
        )
        if not relevant_pids:
            continue

        # Predicted top-K
        recs = get_collab_recommendations(uid, collab_model, catalogue, top_n=k)
        if recs.empty:
            continue
        rec_pids = set(recs["product_id"])

        hits = len(rec_pids & relevant_pids)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant_pids))

    return {
        "precision_at_k": round(np.mean(precisions), 4) if precisions else 0.0,
        "recall_at_k":    round(np.mean(recalls), 4)    if recalls    else 0.0,
        "k": k,
        "n_users_evaluated": len(precisions),
    }


# ── content model evaluation ──────────────────────────────────────────────────

def evaluate_content_diversity(content_model: dict,
                                 sample_n: int = 20,
                                 top_n: int = 10,
                                 seed: int = 42) -> dict:
    """
    Measure intra-list diversity of content recommendations.
    Higher diversity means recommendations are not all identical.

    Parameters
    ----------
    content_model : dict
    sample_n      : int – number of products to sample
    top_n         : int – recommendation list length
    seed          : int

    Returns
    -------
    dict with keys: mean_diversity, mean_similarity
    """
    from src.content_filter import get_content_recommendations

    np.random.seed(seed)
    catalogue  = content_model["catalogue"]
    cosine_sim = content_model["cosine_sim"]

    sample = catalogue.sample(min(sample_n, len(catalogue)), random_state=seed)
    diversities = []

    for _, row in sample.iterrows():
        recs = get_content_recommendations(
            row["product_name"], content_model, top_n=top_n
        )
        if len(recs) < 2:
            continue
        indices = recs.index.tolist()
        sim_vals = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ii = catalogue[catalogue["product_id"] == recs.iloc[i]["product_id"]].index
                jj = catalogue[catalogue["product_id"] == recs.iloc[j]["product_id"]].index
                if len(ii) and len(jj):
                    sim_vals.append(cosine_sim[ii[0], jj[0]])
        if sim_vals:
            diversities.append(1 - np.mean(sim_vals))

    mean_div = round(np.mean(diversities), 4) if diversities else 0.0
    return {
        "mean_diversity": mean_div,
        "mean_similarity": round(1 - mean_div, 4),
    }
