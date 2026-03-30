import joblib
import pandas as pd
import scipy.sparse as sp
from sqlalchemy import create_engine
from implicit.als import AlternatingLeastSquares

DB_URL = "mysql+pymysql://root:123456@127.0.0.1:3306/onlineshop"

interaction_weight = {
    "click": 0.001,
    "wishlist_add": 0.7,
    "cart_add": 1,
    "R5": 2.0,
    "R4": 1.8,
    "R3": 1.6,
    "R2": 1.4,
    "R1": 1.2,
    "R0": 1.0,
}


def load_data(engine):
    return pd.read_sql(
        "SELECT user_id, product_id, interaction_type FROM user_product_interactions",
        engine
    )


def preprocess(df):
    user_product = df[["user_id", "product_id"]].copy()

    user_product["score"] = df["interaction_type"].map(interaction_weight)

    user_product = (
        user_product
        .groupby(["user_id", "product_id"])["score"]
        .sum()
        .reset_index()
    )

    user_product["user_idx"] = user_product["user_id"].astype("category").cat.codes
    user_product["product_idx"] = user_product["product_id"].astype("category").cat.codes

    return user_product


def build_matrix(user_product):
    return sp.csr_matrix(
        (user_product["score"], (user_product["user_idx"], user_product["product_idx"]))
    )

def get_mappings(user_product):
    product_idx_to_id = dict(zip(user_product["product_idx"], user_product["product_id"]))
    product_id_to_idx = dict(zip(user_product["product_id"], user_product["product_idx"]))
    user_id_to_idx = dict(zip(user_product["user_id"], user_product["user_idx"]))
    return product_idx_to_id, product_id_to_idx, user_id_to_idx

def train_als(matrix):
    model = AlternatingLeastSquares(
        factors=50,
        regularization=0.05,
        iterations=100,
        alpha=10
    )
    model.fit(matrix)
    return model


def save_artifacts(user_product, matrix, model, product_idx_to_id, product_id_to_idx):
    joblib.dump(matrix, "./models/matrix_factorization/feature_matrix.joblib")
    joblib.dump(model, "./models/matrix_factorization/matrix_factorization_model.joblib")
    joblib.dump(product_idx_to_id,
                "./models/matrix_factorization/product_idx_to_id_mapping.joblib")
    joblib.dump(product_id_to_idx,
                "./models/matrix_factorization/product_id_to_idx_mapping.joblib") 
    joblib.dump(model.item_factors, "./models/matrix_factorization/item_factors.joblib")

def train_model():

    engine = create_engine(DB_URL)

    df = load_data(engine)

    if df.empty:
        print("No data found")
        return

    user_product = preprocess(df)
    matrix = build_matrix(user_product) 
    model = train_als(matrix)
    product_idx_to_id, product_id_to_idx, user_id_to_idx = get_mappings(user_product)
    save_artifacts(user_product, matrix, model, product_idx_to_id, product_id_to_idx)

    return model, matrix, product_idx_to_id, product_id_to_idx, user_id_to_idx

if __name__ == "__main__":
    train_model()