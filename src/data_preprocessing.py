"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, and feature engineering
for the Product Recommendation System.
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import MinMaxScaler

# ── constants ────────────────────────────────────────────────────────────────
CATEGORIES = ["Electronics", "Books", "Clothing", "Sports", "Home & Kitchen"]
SAMPLE_PRODUCTS = {
    "Electronics": [
        "Wireless Bluetooth Headphones", "USB-C Fast Charger", "4K Smart TV",
        "Gaming Mechanical Keyboard", "Portable Power Bank 20000mAh",
        "Noise Cancelling Earbuds", "Smart Watch Fitness Tracker",
        "Laptop Stand Adjustable", "Webcam 1080p HD", "RGB Gaming Mouse",
    ],
    "Books": [
        "Python Machine Learning Guide", "Deep Learning with TensorFlow",
        "Data Science Handbook", "Clean Code by Robert Martin",
        "The Pragmatic Programmer", "Designing Data-Intensive Applications",
        "Introduction to Algorithms", "Fluent Python", "Statistics for ML",
        "Hands-On ML with Scikit-Learn",
    ],
    "Clothing": [
        "Men's Running Shoes", "Women's Yoga Pants", "Casual Denim Jacket",
        "Waterproof Hiking Boots", "Cotton Polo T-Shirt",
        "Slim Fit Chinos", "Sports Compression Shorts",
        "Winter Puffer Jacket", "Formal Oxford Shoes", "Printed Kurti",
    ],
    "Sports": [
        "Adjustable Dumbbell Set", "Resistance Bands Kit",
        "Yoga Mat Non-Slip 6mm", "Whey Protein Powder Chocolate",
        "Jump Rope Speed Cable", "Pull-Up Bar Doorway",
        "Badminton Racket Set", "Cricket Bat Kashmir Willow",
        "Football Size 5 Pro", "Cycling Helmet Safety",
    ],
    "Home & Kitchen": [
        "Air Fryer 5L Digital", "Stainless Steel Water Bottle",
        "Non-Stick Cookware Set", "Electric Kettle 1.5L",
        "Bamboo Cutting Board", "Coffee Maker Drip",
        "Vacuum Cleaner Cordless", "Blender 600W", "Instant Pot Pressure Cooker",
        "Microwave Oven 20L",
    ],
}

REVIEW_TEMPLATES = [
    "Excellent product, highly recommend to everyone.",
    "Good quality and fast delivery, very satisfied.",
    "Worth every rupee, performs as described.",
    "Average product but serves the purpose well.",
    "Great build quality and easy to use.",
    "Decent product for the price point.",
    "Absolutely love this product, bought two already.",
    "Okay quality but packaging could be better.",
    "Best in its category, no complaints at all.",
    "Solid performance and great value for money.",
]


# ── mock data generation ──────────────────────────────────────────────────────

def generate_mock_data(n_products: int = 500, n_users: int = 1000,
                       n_reviews: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic mock Amazon-style product review dataset.

    Parameters
    ----------
    n_products : int  – number of unique products
    n_users    : int  – number of unique users
    n_reviews  : int  – total number of review rows
    seed       : int  – random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        user_id, product_id, product_name, category, rating,
        review_text, price, num_reviews
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build product catalogue
    products = []
    pid = 1
    for cat, names in SAMPLE_PRODUCTS.items():
        per_cat = n_products // len(SAMPLE_PRODUCTS)
        for i in range(per_cat):
            base_name = names[i % len(names)]
            suffix = f" v{i // len(names) + 1}" if i >= len(names) else ""
            price = round(np.random.uniform(199, 49999), 2)
            products.append({
                "product_id": f"P{pid:04d}",
                "product_name": base_name + suffix,
                "category": cat,
                "price": price,
            })
            pid += 1

    product_df = pd.DataFrame(products)

    # Generate reviews
    rows = []
    for _ in range(n_reviews):
        prod = product_df.sample(1).iloc[0]
        uid = f"U{random.randint(1, n_users):04d}"
        rating = float(np.clip(np.random.normal(3.8, 1.0), 1, 5))
        rating = round(rating * 2) / 2          # round to nearest 0.5
        review = random.choice(REVIEW_TEMPLATES)
        rows.append({
            "user_id": uid,
            "product_id": prod["product_id"],
            "product_name": prod["product_name"],
            "category": prod["category"],
            "price": prod["price"],
            "rating": rating,
            "review_text": review,
        })

    df = pd.DataFrame(rows)
    # Add aggregated review count per product
    counts = df.groupby("product_id")["rating"].count().reset_index()
    counts.columns = ["product_id", "num_reviews"]
    df = df.merge(counts, on="product_id")
    return df


# ── cleaning & preprocessing ─────────────────────────────────────────────────

def load_or_generate(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Load processed data if it exists; otherwise generate mock data,
    clean it, and save both raw + processed versions.

    Parameters
    ----------
    raw_path       : str – path to save/load raw CSV
    processed_path : str – path to save/load cleaned CSV

    Returns
    -------
    pd.DataFrame – cleaned dataset ready for modelling
    """
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)

    # Generate or load raw
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        df = generate_mock_data()
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)

    df = clean_data(df)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop duplicates (same user × product)
      2. Fill / drop nulls
      3. Clip ratings to [1, 5]
      4. Normalize price
      5. Create combined text feature for TF-IDF

    Parameters
    ----------
    df : pd.DataFrame – raw dataframe

    Returns
    -------
    pd.DataFrame – cleaned dataframe
    """
    df = df.copy()

    # 1. Remove duplicate user-product pairs (keep highest rating)
    df = df.sort_values("rating", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "product_id"], keep="first")

    # 2. Handle nulls
    df["review_text"] = df["review_text"].fillna("")
    df["price"] = df["price"].fillna(df["price"].median())
    df = df.dropna(subset=["user_id", "product_id", "rating"])

    # 3. Clip ratings
    df["rating"] = df["rating"].clip(1, 5)

    # 4. Normalise price to [0, 1] for feature use
    scaler = MinMaxScaler()
    df["price_norm"] = scaler.fit_transform(df[["price"]])

    # 5. Combined text feature for content-based filtering
    df["combined_text"] = (
        df["product_name"].str.lower() + " "
        + df["category"].str.lower() + " "
        + df["review_text"].str.lower()
    )

    df = df.reset_index(drop=True)
    return df


def get_product_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per product with aggregated stats.

    Parameters
    ----------
    df : pd.DataFrame – cleaned full dataframe

    Returns
    -------
    pd.DataFrame – product-level dataframe
    """
    catalogue = (
        df.groupby(["product_id", "product_name", "category", "price",
                    "combined_text"])
        .agg(avg_rating=("rating", "mean"),
             num_reviews=("rating", "count"),
             price_norm=("price_norm", "mean"))
        .reset_index()
    )
    catalogue["avg_rating"] = catalogue["avg_rating"].round(2)
    return catalogue


def get_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user × product pivot table of ratings.
    Missing values are left as NaN (not filled) so SVD handles them.

    Parameters
    ----------
    df : pd.DataFrame – cleaned full dataframe

    Returns
    -------
    pd.DataFrame – pivot table (users as rows, products as columns)
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean",
    )
    return matrix
