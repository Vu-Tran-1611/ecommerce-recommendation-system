import joblib 
import torch 
class ModelLoader:
    def __init__(self):
        #  Metadata
        self.metadata = joblib.load("data/cleaned_data/metadata.joblib")

        # # # ------------------------ KNN Euclidean  ------------------------

        # ---------------- Vu ----------------
    
        # self.knn_model = joblib.load("models/knn_euclidean/knn_model.joblib")
        # self.knn_feature_matrix = joblib.load("models/knn_euclidean/feature_matrix.joblib")
        # ---------------- Vu ----------------

        # # # ---------------- Blake ----------------
        self.knn_model = joblib.load("models/blake/v1/knn_model.joblib")
        self.knn_feature_matrix = joblib.load("models/blake/v1/feature_matrix.joblib")
        # # # ---------------- Blake ----------------



        # # TFIDF + Cosine Similarity 
        # # ---------------- Vu ----------------
        # self.tfidf_model = joblib.load("models/tfidf_cosine/tfidf_model.joblib")
        # # ---------------- Vu ----------------


         # # # ---------------- Brandon ----------------
        # V1 
        self.tfidf_model_v1 = joblib.load("models/brandon/v1/knn_model.joblib")
        self.tfidf_feature_matrix_v1 = joblib.load("models/brandon/v1/feature_matrix.joblib")

        # V2 
        self.tfidf_model_v2 = joblib.load("models/brandon/v2/knn_model.joblib")
        self.tfidf_feature_matrix_v2 = joblib.load("models/brandon/v2/feature_matrix.joblib")
        
        # # # ---------------- Brandon ----------------




        # TFIDF + KNN + Cosine Similarity

         # # ---------------- Vu ----------------
        # self.tfidf_knn_model = joblib.load("models/tfidf_knn_cosine/tfidf_knn_model.joblib")
        # self.tfidf_knn_feature_matrix = joblib.load("models/tfidf_knn_cosine/feature_matrix.joblib")
        # # ---------------- Vu ----------------


        # # ---------------- Arjun ----------------

        self.tfidf_knn_model = joblib.load("models/arjun/v1/knn_model.joblib")
        self.tfidf_knn_feature_matrix = joblib.load("models/arjun/v1/feature_matrix.joblib")
        
        # # ---------------- Arjun ----------------






        # Matrix Factorization
        # self.matrix_factorization_model = joblib.load("models/matrix_factorization/matrix_factorization_model.joblib")
        self.matrix_factorization_item_factors = joblib.load("models/matrix_factorization/item_factors.joblib")
        self.matrix_factorization_user_factors = joblib.load("models/matrix_factorization/user_factors.joblib")
        self.matrix_factorization_feature_matrix = joblib.load("models/matrix_factorization/feature_matrix.joblib")
        self.matrix_factorization_product_idx_to_id_mapping = joblib.load("models/matrix_factorization/product_idx_to_id_mapping.joblib")
        self.matrix_factorization_user_id_to_idx_mapping = joblib.load("models/matrix_factorization/user_id_to_idx_mapping.joblib") 
        self.matrix_factorization_product_id_to_idx_mapping = joblib.load("models/matrix_factorization/product_id_to_idx_mapping.joblib")



        # LightGCN
        self.lightgcn_item_embeddings = joblib.load("models/lightgcn/item_embeddings.joblib")
        self.lightgcn_user_embeddings = joblib.load("models/lightgcn/user_embeddings.joblib")
        self.lightgcn_user_id_to_idx_mapping = joblib.load("models/lightgcn/user_id_to_idx_mapping.joblib") 
        self.lightgcn_user_idx_to_id_mapping = joblib.load("models/lightgcn/user_idx_to_id_mapping.joblib")
        self.lightgcn_product_id_to_idx_mapping = joblib.load("models/lightgcn/product_id_to_idx_mapping.joblib")
        self.lightgcn_product_idx_to_id_mapping = joblib.load("models/lightgcn/product_idx_to_id_mapping.joblib")

        # SASRec
        self.sasrec_checkpoint = torch.load("models/sasrec/sasrec_checkpoint.pth")
        self.sasrec_item_id_to_index = torch.load("models/sasrec/item_id_to_index.pth")
        self.sasrec_item_index_to_id = torch.load("models/sasrec/item_index_to_id.pth")
        self.sasrec_user_id_to_index = torch.load("models/sasrec/user_id_to_index.pth") 

        # Bert4Rec
        self.bert4rec_checkpoint = torch.load("models/bert4rec/bert4rec_checkpoint.pth")
        self.bert4rec_item_id_to_index = torch.load("models/bert4rec/item_id_to_index.pth")
        self.bert4rec_item_index_to_id = torch.load("models/bert4rec/item_index_to_id.pth")
        self.bert4rec_user_id_to_index = torch.load("models/bert4rec/user_id_to_index.pth")

        # ComiRec
        self.comirec_checkpoint = torch.load("models/comirec/comirec_checkpoint.pth")
        self.comirec_item_id_to_index = torch.load("models/comirec/item_id_to_index.pth")
        self.comirec_item_index_to_id = torch.load("models/comirec/item_index_to_id.pth")
        self.comirec_user_id_to_index = torch.load("models/comirec/user_id_to_index.pth") 

        # TwoTower
        self.twotower_vector_embeddings = torch.load("models/twotower/vector_embeddings.pth")
        self.twotower_user_id_to_index = torch.load("models/twotower/user_id_to_index.pth")
        self.twotower_user_index_to_id = torch.load("models/twotower/user_index_to_id.pth") 
        self.twotower_product_id_to_index = torch.load("models/twotower/product_id_to_index.pth")
        self.twotower_product_index_to_id = torch.load("models/twotower/product_index_to_id.pth")
        
