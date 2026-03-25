# 🛍️ Product Recommendation System

> A hybrid ML-powered recommender that suggests products using Content-Based Filtering (TF-IDF + Cosine Similarity) and Collaborative Filtering (SVD Matrix Factorisation).

Built as part of the **Elevate Labs Python Internship** — 2 week project phase.

---

## 📌 Overview

This project builds a complete end-to-end recommendation system with a clean Streamlit web UI. It supports three recommendation modes — content-based, collaborative, and hybrid — and includes model evaluation metrics and interactive EDA charts.

---

## 🚀 Features

| Feature | Details |
|---------|---------|
| Content-Based Filtering | TF-IDF on product name + category + reviews → Cosine Similarity |
| Collaborative Filtering | User-Item matrix → Truncated SVD latent factors |
| Hybrid Mode | Weighted blend of both scores (adjustable alpha slider) |
| Evaluation | RMSE, MAE, Precision@K, Recall@K, Diversity score |
| EDA Dashboard | 5 interactive Plotly charts (category, rating, price, reviews) |
| Export | Download any recommendation list as CSV |
| Similarity Heatmap | Visual cosine similarity matrix for top 20 products |
| Filters | Category, min rating, max price, results count |
| Surprise Me | Random product / random user buttons |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas, numpy** — data manipulation
- **scikit-learn** — TF-IDF, TruncatedSVD, metrics
- **scipy** — matrix operations
- **Streamlit** — web UI
- **Plotly** — interactive charts
- **joblib** — model serialisation

---

## 📁 Project Structure

```
product-recommendation-system/
├── app.py                        # Main Streamlit app (entry point)
├── requirements.txt              # All dependencies
├── README.md                     # This file
├── .gitignore
├── data/
│   ├── raw/
│   │   └── products.csv          # Raw dataset (auto-generated if missing)
│   └── processed/
│       └── cleaned_data.csv      # Cleaned + feature-engineered data
├── models/
│   ├── content_model.pkl         # TF-IDF + cosine similarity matrix
│   └── collab_model.pkl          # SVD model + user/item factors
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading, cleaning, mock generation
│   ├── content_filter.py         # Content-based filtering logic
│   ├── collab_filter.py          # Collaborative filtering (SVD)
│   ├── hybrid.py                 # Hybrid score combination
│   └── evaluation.py             # RMSE, MAE, Precision@K, Recall@K
├── notebooks/
│   └── EDA.ipynb                 # Exploratory analysis notebook
└── assets/
    └── banner.png
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/product-recommendation-system.git
cd product-recommendation-system

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> **Note:** On first run, the app auto-generates a realistic mock dataset (500 products, 1000 users, 8000 reviews) and trains both models. This takes ~30 seconds. Subsequent runs load from cache instantly.

---

## 📊 Dataset

Auto-generated realistic mock data with:
- **500 products** across 5 categories (Electronics, Books, Clothing, Sports, Home & Kitchen)
- **1000 users**
- **~8000 ratings** (1–5 scale, half-star increments)
- Features: `user_id, product_id, product_name, category, rating, review_text, price`

You can replace `data/raw/products.csv` with a real Amazon dataset from Kaggle and the pipeline will work automatically.

---

## 🧠 How It Works

```
Raw CSV
  │
  ▼
data_preprocessing.py ──► Clean data, build product catalogue, user-item matrix
  │
  ├──► content_filter.py ──► TF-IDF vectorise ──► Cosine Similarity Matrix
  │
  ├──► collab_filter.py  ──► SVD on rating matrix ──► User & Item latent factors
  │
  └──► hybrid.py         ──► Normalise + weighted blend ──► Ranked list
```

**Hybrid Score Formula:**
```
hybrid_score = α × content_score + (1 − α) × collab_score
```
where α ∈ [0, 1] is controlled by the UI slider.

---

## 📈 Model Performance (approximate)

| Metric | Score |
|--------|-------|
| RMSE (Collab) | ~0.95 |
| MAE  (Collab) | ~0.75 |
| Precision@10  | ~0.78 |
| Recall@10     | ~0.62 |
| Content Diversity | ~0.72 |

*(Scores vary slightly depending on random seed and dataset)*

---

## 🎯 Interview Talking Points

- Why TF-IDF over word embeddings? (Speed, interpretability, no training needed)
- What is cosine similarity and why is it used for text?
- What problem does SVD solve in collaborative filtering?
- How does the cold-start problem affect collaborative filtering?
- What is Precision@K vs Recall@K?
- Why normalise scores before hybrid combination?

---

## 👤 Author

**B Ashok** — B.Tech CSD, ANITS  

*Built during Python Internship @ Elevate Labs (2026)*

---

## 📃 License

MIT License — free to use, modify, and distribute.
