"""
app.py
------
Main Streamlit entry point for the Product Recommendation System.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    load_or_generate, get_product_catalogue, get_user_item_matrix
)
from src.content_filter import (
    build_content_model, load_content_model,
    get_content_recommendations, get_product_names, get_similarity_heatmap_data
)
from src.collab_filter import (
    build_collab_model, load_collab_model,
    get_collab_recommendations, get_user_ids
)
from src.hybrid import get_hybrid_recommendations
from src.evaluation import evaluate_collab, precision_recall_at_k, evaluate_content_diversity

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main { background: #0f0f14; color: #e8e8f0; }
.stApp { background: #0f0f14; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.product-card {
    background: linear-gradient(135deg, #1c1c2e 0%, #1a1a30 100%);
    border: 1px solid #3a3a6a;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}
.product-card:hover { border-color: #7c7cff; }
.badge {
    display: inline-block;
    background: #2e2e5e;
    color: #a0a0ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}
.score-pill {
    background: linear-gradient(90deg, #6c63ff, #a855f7);
    color: white;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #8888bb;
}
.stTabs [aria-selected="true"] { color: #a0a0ff !important; }
</style>
""", unsafe_allow_html=True)

# ── data + model loading (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = load_or_generate(
        raw_path="data/raw/products.csv",
        processed_path="data/processed/cleaned_data.csv",
    )
    catalogue = get_product_catalogue(df)
    matrix    = get_user_item_matrix(df)
    return df, catalogue, matrix


@st.cache_resource(show_spinner=False)
def load_models(catalogue, matrix):
    # Content model
    if os.path.exists("models/content_model.pkl"):
        c_model = load_content_model()
    else:
        _, _, _ = build_content_model(catalogue)
        c_model = load_content_model()

    # Collab model
    if os.path.exists("models/collab_model.pkl"):
        u_model = load_collab_model()
    else:
        build_collab_model(matrix)
        u_model = load_collab_model()

    return c_model, u_model


# ── helpers ───────────────────────────────────────────────────────────────────
def star_rating(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "⯨" * half + "☆" * empty


def render_product_cards(recs: pd.DataFrame, score_col: str = None):
    if recs.empty:
        st.warning("No recommendations found. Try adjusting filters.")
        return
    cols_per_row = 2
    for i in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(recs):
                break
            row = recs.iloc[idx]
            score_html = ""
            if score_col and score_col in recs.columns:
                score_html = f'<span class="score-pill">Score: {row[score_col]:.3f}</span>'
            col.markdown(f"""
            <div class="product-card">
                <strong style="font-size:1rem; font-family:'Syne',sans-serif;">{row['product_name']}</strong><br>
                <span class="badge">{row['category']}</span>
                {score_html}<br>
                <span style="color:#f5c518; font-size:1.1rem;">{star_rating(row['avg_rating'])}</span>
                <span style="color:#aaa; font-size:0.85rem;"> {row['avg_rating']}/5</span><br>
                <span style="color:#7cfc8a; font-weight:600;">₹{row['price']:,.0f}</span>
                <span style="color:#666; font-size:0.8rem; margin-left:8px;">
                    {int(row.get('num_reviews', 0))} reviews
                </span>
            </div>
            """, unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ Product Recommender")
    st.markdown("---")
    st.markdown("**Filters**")
    cat_filter  = st.selectbox("Category", ["All", "Electronics", "Books",
                                             "Clothing", "Sports", "Home & Kitchen"])
    min_rating  = st.slider("Min Rating", 1.0, 5.0, 3.0, 0.5)
    max_price   = st.number_input("Max Price (₹)", min_value=0,
                                   max_value=100000, value=50000, step=500)
    top_n       = st.slider("Results to show", 4, 20, 8, 2)
    st.markdown("---")
    st.markdown("<small style='color:#666'>Built during Elevate Labs Internship</small>",
                unsafe_allow_html=True)

# ── load everything ───────────────────────────────────────────────────────────
with st.spinner("🚀 Loading data & models..."):
    df, catalogue, matrix = load_data()
    c_model, u_model = load_models(catalogue, matrix)

product_names = get_product_names(c_model)
user_ids_list = get_user_ids(u_model)

# ── tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Home", "🔍 Content-Based", "👤 Collaborative",
    "⚡ Hybrid", "📊 Evaluation", "📈 EDA"
])

# ─────────────── TAB 1: HOME ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("# 🛍️ Product Recommendation System")
    st.markdown("### A hybrid ML-powered recommender built with TF-IDF + SVD")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Products", f"{catalogue['product_id'].nunique():,}")
    c2.metric("👥 Users", f"{df['user_id'].nunique():,}")
    c3.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")
    c4.metric("🗂 Categories", df['category'].nunique())

    st.markdown("---")
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🧮 Content-Based</h4>
        <p>Uses TF-IDF on product names, categories, and reviews.
        Computes cosine similarity to find the most similar products.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🤝 Collaborative</h4>
        <p>Builds a user-item rating matrix and applies
        SVD matrix factorisation to discover latent preferences.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>⚡ Hybrid</h4>
        <p>Combines both scores with an adjustable alpha weight,
        giving you the best of both approaches in one ranked list.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Sample Products")
    st.dataframe(
        catalogue[["product_name", "category", "price", "avg_rating", "num_reviews"]]
        .sort_values("num_reviews", ascending=False)
        .head(10)
        .reset_index(drop=True),
        width='stretch',
    )

# ─────────────── TAB 2: CONTENT-BASED ────────────────────────────────────────
with tabs[1]:
    st.markdown("## 🔍 Content-Based Recommendations")
    st.markdown("Find products similar to a product you like.")

    query = st.selectbox("Select or search a product", [""] + product_names,
                          key="cb_product")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        surprise = st.button("🎲 Surprise Me", key="cb_surprise")
    if surprise:
        query = np.random.choice(product_names)
        st.info(f"Random pick: **{query}**")

    if query:
        with st.spinner("Finding similar products..."):
            recs = get_content_recommendations(
                product_name=query,
                model=c_model,
                top_n=top_n,
                category_filter=cat_filter if cat_filter != "All" else None,
                min_rating=min_rating,
                max_price=max_price if max_price > 0 else None,
            )

        st.markdown(f"### Results for: *{query}*")
        render_product_cards(recs, score_col="similarity_score")

        if not recs.empty:
            csv = recs.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv,
                               "content_recommendations.csv", "text/csv")

        # Heatmap
        st.markdown("---")
        st.markdown("### 🔥 Similarity Heatmap (Top 20 Products)")
        sim_df, names = get_similarity_heatmap_data(c_model, top_n=20)
        fig = px.imshow(
            sim_df,
            color_continuous_scale="Viridis",
            title="Cosine Similarity Between Top 20 Products",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="#0f0f14", plot_bgcolor="#0f0f14",
            font_color="#e8e8f0", height=500,
        )
        st.plotly_chart(fig, width='stretch')

# ─────────────── TAB 3: COLLABORATIVE ────────────────────────────────────────
with tabs[2]:
    st.markdown("## 👤 Collaborative Recommendations")
    st.markdown("Personalised picks based on what similar users liked.")

    uid = st.selectbox("Select User ID", [""] + user_ids_list[:200], key="cf_user")
    col_b1, _ = st.columns([1, 5])
    with col_b1:
        rand_user = st.button("🎲 Random User", key="cf_rand")
    if rand_user:
        uid = np.random.choice(user_ids_list)
        st.info(f"Random user: **{uid}**")

    if uid:
        with st.spinner("Generating personalised recommendations..."):
            recs = get_collab_recommendations(
                user_id=uid,
                model=u_model,
                catalogue=catalogue,
                top_n=top_n,
                category_filter=cat_filter if cat_filter != "All" else None,
            )

        st.markdown(f"### Recommendations for: *{uid}*")
        render_product_cards(recs, score_col="predicted_score")

        if not recs.empty:
            csv = recs.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv,
                               "collab_recommendations.csv", "text/csv")

        # User's rating history
        user_history = df[df["user_id"] == uid][
            ["product_name", "category", "rating"]
        ].sort_values("rating", ascending=False).head(10)
        if not user_history.empty:
            st.markdown("---")
            st.markdown("### 📋 User's Rating History")
            st.dataframe(user_history.reset_index(drop=True),
                         width='stretch')

# ─────────────── TAB 4: HYBRID ───────────────────────────────────────────────
with tabs[3]:
    st.markdown("## ⚡ Hybrid Recommendations")
    st.markdown("The best of both worlds — content + collaborative combined.")

    h_product = st.selectbox("Product for content signal",
                              [""] + product_names, key="h_prod")
    h_user    = st.selectbox("User for collaborative signal",
                              [""] + user_ids_list[:200], key="h_user")
    alpha     = st.slider("Alpha (0 = full collaborative, 1 = full content)",
                           0.0, 1.0, 0.5, 0.05)

    st.markdown(f"""
    <div class="metric-card">
    📐 <b>Current blend:</b> &nbsp;
    <span class="score-pill">{alpha:.0%} Content</span> &nbsp;+&nbsp;
    <span class="score-pill">{1-alpha:.0%} Collaborative</span>
    </div>
    """, unsafe_allow_html=True)

    if h_product and h_user:
        with st.spinner("Blending recommendations..."):
            recs = get_hybrid_recommendations(
                product_name=h_product,
                user_id=h_user,
                content_model=c_model,
                collab_model=u_model,
                catalogue=catalogue,
                alpha=alpha,
                top_n=top_n,
                category_filter=cat_filter if cat_filter != "All" else None,
                min_rating=min_rating,
                max_price=max_price if max_price > 0 else None,
            )

        st.markdown("### 🏆 Hybrid Results")
        render_product_cards(recs, score_col="hybrid_score")

        if not recs.empty:
            csv = recs.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv,
                               "hybrid_recommendations.csv", "text/csv")

            # Score breakdown chart
            st.markdown("---")
            st.markdown("### 📊 Score Breakdown")
            fig = go.Figure()
            fig.add_bar(name="Content Score",  x=recs["product_name"],
                        y=recs["content_score"], marker_color="#6c63ff")
            fig.add_bar(name="Collab Score",   x=recs["product_name"],
                        y=recs["collab_score"],  marker_color="#a855f7")
            fig.add_bar(name="Hybrid Score",   x=recs["product_name"],
                        y=recs["hybrid_score"],  marker_color="#22d3ee")
            fig.update_layout(
                barmode="group", paper_bgcolor="#0f0f14",
                plot_bgcolor="#0f0f14", font_color="#e8e8f0",
                xaxis_tickangle=-30, height=400,
                legend=dict(bgcolor="#1a1a2e"),
            )
            st.plotly_chart(fig, width='stretch')

# ─────────────── TAB 5: EVALUATION ───────────────────────────────────────────
with tabs[4]:
    st.markdown("## 📊 Model Evaluation")

    if st.button("▶️ Run Evaluation (takes ~30s)", key="run_eval"):
        with st.spinner("Evaluating collaborative filter..."):
            collab_metrics = evaluate_collab(df, u_model)
            pk_metrics     = precision_recall_at_k(df, u_model, catalogue, k=10)
            content_div    = evaluate_content_diversity(c_model)

        st.markdown("### Collaborative Filter — Rating Prediction")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE",  collab_metrics["rmse"] or "N/A")
        m2.metric("MAE",   collab_metrics["mae"] or "N/A")
        m3.metric("Test Samples", collab_metrics["n_test_samples"])

        st.markdown("### Precision & Recall @ K=10")
        p1, p2, p3 = st.columns(3)
        p1.metric("Precision@10",        pk_metrics["precision_at_k"])
        p2.metric("Recall@10",           pk_metrics["recall_at_k"])
        p3.metric("Users Evaluated",     pk_metrics["n_users_evaluated"])

        st.markdown("### Content-Based — Intra-List Diversity")
        d1, d2 = st.columns(2)
        d1.metric("Mean Diversity",   content_div["mean_diversity"])
        d2.metric("Mean Similarity",  content_div["mean_similarity"])

        # Rating distribution for predicted vs actual
        st.markdown("---")
        st.markdown("### Rating Distribution in Dataset")
        fig = px.histogram(df, x="rating", nbins=9,
                           color="category",
                           title="Rating Distribution by Category",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor="#0f0f14", plot_bgcolor="#1a1a2e",
                          font_color="#e8e8f0")
        st.plotly_chart(fig, width='stretch')

    else:
        st.info("Click **Run Evaluation** to compute metrics.")
        st.markdown("""
        | Metric | What it measures |
        |--------|-----------------|
        | RMSE | Root Mean Squared Error on predicted ratings |
        | MAE  | Mean Absolute Error on predicted ratings |
        | Precision@K | Fraction of top-K recs that are relevant |
        | Recall@K | Fraction of relevant items found in top-K |
        | Diversity | How varied the recommendations are |
        """)

# ─────────────── TAB 6: EDA ───────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 📈 Exploratory Data Analysis")

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        fig = px.bar(
            df.groupby("category")["product_id"].count().reset_index()
             .rename(columns={"product_id": "count"})
             .sort_values("count", ascending=True),
            x="count", y="category", orientation="h",
            title="📦 Reviews per Category",
            color="count", color_continuous_scale="Plasma",
        )
        fig.update_layout(paper_bgcolor="#0f0f14", plot_bgcolor="#1a1a2e",
                          font_color="#e8e8f0", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with row1_c2:
        fig = px.box(df, x="category", y="rating",
                     title="⭐ Rating Distribution per Category",
                     color="category",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor="#0f0f14", plot_bgcolor="#1a1a2e",
                          font_color="#e8e8f0", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        top_products = (
            df.groupby("product_name")["rating"].count()
              .reset_index()
              .rename(columns={"rating": "review_count"})
              .sort_values("review_count", ascending=False)
              .head(15)
        )
        fig = px.bar(top_products, x="review_count", y="product_name",
                     orientation="h",
                     title="🏆 Most Reviewed Products",
                     color="review_count",
                     color_continuous_scale="Teal")
        fig.update_layout(paper_bgcolor="#0f0f14", plot_bgcolor="#1a1a2e",
                          font_color="#e8e8f0", showlegend=False, height=450)
        st.plotly_chart(fig, width='stretch')

    with row2_c2:
        price_cat = catalogue.groupby("category")["price"].mean().reset_index()
        fig = px.pie(price_cat, names="category", values="price",
                     title="💰 Average Price Share by Category",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor="#0f0f14", font_color="#e8e8f0")
        st.plotly_chart(fig, width='stretch')

    # Price vs Rating scatter
    st.markdown("### 💡 Price vs Average Rating")
    fig = px.scatter(
        catalogue.sample(min(300, len(catalogue))),
        x="price", y="avg_rating",
        color="category", size="num_reviews",
        hover_name="product_name",
        title="Price vs Rating (bubble size = review count)",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(paper_bgcolor="#0f0f14", plot_bgcolor="#1a1a2e",
                      font_color="#e8e8f0")
    st.plotly_chart(fig, width='stretch')
