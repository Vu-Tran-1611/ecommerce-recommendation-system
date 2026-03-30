import joblib 

class ModelLoader:
    def __init__(self):
        #  Metadata
        self.metadata = joblib.load("data/cleaned_data/metadata.joblib")
        #  KNN Euclidean 
        self.knn_model = joblib.load("models/knn_euclidean/knn_model.joblib")
        self.knn_feature_matrix = joblib.load("models/knn_euclidean/feature_matrix.joblib")

        # TFIDF + Cosine Similarity 
        self.tfidf_model = joblib.load("models/tfidf_cosine/tfidf_model.joblib")

        # TFIDF + KNN + Cosine Similarity
        self.tfidf_knn_model = joblib.load("models/tfidf_knn_cosine/tfidf_knn_model.joblib")
        self.tfidf_knn_feature_matrix = joblib.load("models/tfidf_knn_cosine/feature_matrix.joblib")

        # Matrix Factorization
        self.matrix_factorization_model = joblib.load("models/matrix_factorization/matrix_factorization_model.joblib")
        self.matrix_factorization_item_factors = joblib.load("models/matrix_factorization/item_factors.joblib")
        self.matrix_factorization_feature_matrix = joblib.load("models/matrix_factorization/feature_matrix.joblib")
        self.matrix_factorization_product_idx_to_id_mapping = joblib.load("models/matrix_factorization/product_idx_to_id_mapping.joblib")
        self.matrix_factorization_user_id_to_idx_mapping = joblib.load("models/matrix_factorization/user_id_to_idx_mapping.joblib") 
        self.matrix_factorization_product_id_to_idx_mapping = joblib.load("models/matrix_factorization/product_id_to_idx_mapping.joblib")