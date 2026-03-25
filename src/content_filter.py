"""
content_filter.py
-----------------
Content-Based Filtering using TF-IDF + Cosine Similarity.
Recommends products similar to a given product based on
their textual features (name, category, reviews).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── model training ────────────────────────────────────────────────────────────

def build_content_model(catalogue: pd.DataFrame,
                        model_path: str = "models/content_model.pkl"
                        ) -> tuple:
    """
    Build and save the TF-IDF + cosine similarity model.

    Parameters
    ----------
    catalogue   : pd.DataFrame – product catalogue (one row per product)
    model_path  : str          – path to save the pickled model

    Returns
    -------
    tuple : (tfidf_matrix, cosine_sim_matrix, vectorizer, catalogue)
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,           # dampens high-frequency terms
    )

    tfidf_matrix = vectorizer.fit_transform(catalogue["combined_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    payload = {
        "cosine_sim": cosine_sim,
        "vectorizer": vectorizer,
        "product_ids": catalogue["product_id"].tolist(),
        "catalogue": catalogue,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)
    return cosine_sim, vectorizer, catalogue


def load_content_model(model_path: str = "models/content_model.pkl") -> dict:
    """
    Load a previously saved content model from disk.

    Parameters
    ----------
    model_path : str – path to the pickled model file

    Returns
    -------
    dict with keys: cosine_sim, vectorizer, product_ids, catalogue
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Content model not found at '{model_path}'. "
            "Call build_content_model() first."
        )
    return joblib.load(model_path)


# ── recommendations ───────────────────────────────────────────────────────────

def get_content_recommendations(product_name: str,
                                 model: dict,
                                 top_n: int = 10,
                                 category_filter: str = None,
                                 min_rating: float = 1.0,
                                 max_price: float = None) -> pd.DataFrame:
    """
    Return top-N products most similar to the given product_name.

    Parameters
    ----------
    product_name    : str   – name of the query product
    model           : dict  – loaded content model dict
    top_n           : int   – number of recommendations
    category_filter : str   – optional category to restrict results
    min_rating      : float – minimum average rating filter
    max_price       : float – maximum price filter

    Returns
    -------
    pd.DataFrame – top-N recommended products with similarity scores
    """
    catalogue = model["catalogue"]
    cosine_sim = model["cosine_sim"]
    product_ids = model["product_ids"]

    # Find the index of the queried product
    matches = catalogue[
        catalogue["product_name"].str.lower() == product_name.lower()
    ]
    if matches.empty:
        # Fuzzy fallback – partial match
        matches = catalogue[
            catalogue["product_name"].str.lower().str.contains(
                product_name.lower(), na=False
            )
        ]

    if matches.empty:
        return pd.DataFrame()          # no match found

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the query product itself
    sim_scores = [(i, s) for i, s in sim_scores if i != idx]

    # Build result dataframe
    top_indices = [i for i, _ in sim_scores[:top_n * 3]]   # over-fetch to allow filtering
    top_scores  = [s for _, s in sim_scores[:top_n * 3]]

    result = catalogue.iloc[top_indices].copy()
    result["similarity_score"] = [round(s, 4) for s in top_scores]

    # Apply filters
    result = result[result["avg_rating"] >= min_rating]
    if category_filter and category_filter != "All":
        result = result[result["category"] == category_filter]
    if max_price:
        result = result[result["price"] <= max_price]

    return result.head(top_n).reset_index(drop=True)


def get_product_names(model: dict) -> list:
    """Return sorted list of all product names for autocomplete."""
    return sorted(model["catalogue"]["product_name"].unique().tolist())


def get_similarity_heatmap_data(model: dict, top_n: int = 20) -> tuple:
    """
    Return cosine similarity submatrix for the top-N most reviewed products.

    Parameters
    ----------
    model  : dict – loaded content model
    top_n  : int  – number of products to include

    Returns
    -------
    tuple : (similarity_matrix_df, product_name_list)
    """
    catalogue = model["catalogue"]
    cosine_sim = model["cosine_sim"]

    top_products = catalogue.nlargest(top_n, "num_reviews")
    indices = top_products.index.tolist()
    names = top_products["product_name"].tolist()

    sub_matrix = cosine_sim[np.ix_(indices, indices)]
    sim_df = pd.DataFrame(sub_matrix, index=names, columns=names)
    return sim_df, names
