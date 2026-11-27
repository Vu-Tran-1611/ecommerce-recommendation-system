"""Train the TF-IDF + KNN cosine model used for similar products.

Rebuilds data/cleaned_data (metadata.joblib, cleaned_products.*) from
data/raw_data/products.csv, then saves the model to models/tfidf_knn_cosine_v2/v1.

Run from the project root:
    python -m training.train_tfidf_knn_cosine
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = Path("data/raw_data/products.csv")
CLEAN_DIR = Path("data/cleaned_data")
MODEL_DIR = Path("models/tfidf_knn_cosine_v2/v1")

TEXT_COLS = ["product_name", "slug", "short_description", "long_description", "sku", "seo_title", "seo_description"]
TFIDF_COLS = ["product_name", "short_description", "long_description", "seo_title", "seo_description"]
CATEGORICAL_COLS = ["brand_name", "category_name", "sub_category_name", "gender", "product_type", "status", "is_approved"]
ONE_HOT_COLS = ["brand_name", "category_name", "sub_category_name", "gender"]
DATETIME_COLS = ["offer_start_price", "offer_end_price", "created_at"]
N_NEIGHBORS = 11


# Strip HTML tags from a description
def remove_html(text):
    if pd.isna(text):
        return "unavailable"
    return re.sub(r"<.*?>", "", str(text))


# Clean raw products the same way as notebooks/data_cleaning.ipynb
def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"id": "product_id", "name": "product_name"}).copy()
    df["gender"] = df.get("gender", pd.Series(index=df.index, dtype=object)).fillna("unspecified")

    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["short_description", "long_description"]:
        df[col] = df[col].apply(remove_html)
    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


# Use plain Python string types so loading does not need pyarrow
def to_plain_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            categories = pd.Index(list(df[col].cat.categories), dtype=object)
            df[col] = pd.Categorical(df[col].astype(object), categories=categories)
        elif pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].astype(object)
    df.columns = pd.Index(list(df.columns), dtype=object)
    return df


# Keep the previous row order so other models' feature matrices stay aligned
def align_with_previous(df: pd.DataFrame) -> pd.DataFrame:
    previous_path = CLEAN_DIR / "metadata.joblib"
    if not previous_path.exists():
        return df.reset_index(drop=True)

    previous_ids = list(joblib.load(previous_path)["product_id"])
    position = {pid: i for i, pid in enumerate(previous_ids)}
    df = df.assign(_order=df["product_id"].map(lambda pid: position.get(pid, len(previous_ids) + pid)))
    removed = set(previous_ids) - set(df["product_id"])
    if removed:
        raise ValueError(f"Products removed since last training {sorted(removed)}; retrain all content models together")
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# Build TF-IDF text, one-hot and scaled price features
def build_features(df: pd.DataFrame):
    text = df[TFIDF_COLS].astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(text)
    one_hot = OneHotEncoder(handle_unknown="ignore").fit_transform(df[ONE_HOT_COLS].astype(str))
    price = csr_matrix(StandardScaler().fit_transform(df[["price"]].astype(float)))
    return hstack([tfidf, one_hot, price]).tocsr()


# Share of top-10 neighbours shared with the previous model
def overlap_with_previous(df: pd.DataFrame, knn, features) -> float | None:
    try:
        old_features = joblib.load(MODEL_DIR / "feature_matrix.joblib")
        old_knn = joblib.load(MODEL_DIR / "knn_model.joblib")
    except FileNotFoundError:
        return None
    count = min(old_features.shape[0], features.shape[0])
    old = old_knn.kneighbors(old_features[:count], return_distance=False)[:, 1:11]
    new = knn.kneighbors(features[:count], return_distance=False)[:, 1:11]
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(old, new)]))


def main():
    df = to_plain_types(align_with_previous(clean_products(pd.read_csv(RAW_PATH))))
    features = build_features(df)
    knn = NearestNeighbors(metric="cosine", n_neighbors=N_NEIGHBORS).fit(features)

    overlap = overlap_with_previous(df, knn, features)
    if overlap is not None:
        print(f"Top-10 overlap with previous model: {overlap:.1%}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / "cleaned_products.csv", index=False)
    df.to_parquet(CLEAN_DIR / "cleaned_products.parquet", index=False)
    joblib.dump(df, CLEAN_DIR / "metadata.joblib")
    joblib.dump(features, MODEL_DIR / "feature_matrix.joblib")
    joblib.dump(knn, MODEL_DIR / "knn_model.joblib")
    print(f"Saved {len(df)} products, {features.shape[1]} features")


if __name__ == "__main__":
    main()
