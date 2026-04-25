from app.services.knn_euclidean_service import KNNEuclideanService
from app.services.light_gcn_service import LightGCNService
from app.services.tfidf_cosine_service import TFIDFCosineService
from app.services.tfidf_knn_cosine_service import TFIDFKNNCosineService 
from app.services.matrix_factorization_service import MatrixFactorizationRecentService
class RecommendationServiceFactory: 
    def __init__(self,model_loader):
        self.model_loader = model_loader 
    def get_service(self,model_name:str,version=None): 
        if model_name == "knn_euclidean":
            return KNNEuclideanService(self.model_loader.knn_model, 
                                       self.model_loader.knn_feature_matrix,
                                       self.model_loader.metadata) 
        elif model_name == "tfidf_cosine": 
            if version == "v1" or version is None:
                tfidf_model = self.model_loader.tfidf_model_v1
                tfidf_feature_matrix = self.model_loader.tfidf_feature_matrix_v1
            elif version == "v2":
                tfidf_model = self.model_loader.tfidf_model_v2
                tfidf_feature_matrix = self.model_loader.tfidf_feature_matrix_v2
            else:
                raise ValueError(f"Version {version} not supported for model {model_name}")
            return TFIDFCosineService(tfidf_model,
                                       tfidf_feature_matrix,
                                       self.model_loader.metadata)
        elif model_name == "tfidf_knn_cosine": 
            return TFIDFKNNCosineService(
                self.model_loader.tfidf_knn_model, 
                self.model_loader.tfidf_knn_feature_matrix,
                self.model_loader.metadata
            )
        elif model_name == "matrix_factorization":
            if version == "recent" or version is None: 
                return MatrixFactorizationRecentService(
                    self.model_loader.matrix_factorization_item_factors,
                    self.model_loader.matrix_factorization_user_factors,
                    self.model_loader.matrix_factorization_product_idx_to_id_mapping, 
                    self.model_loader.matrix_factorization_product_id_to_idx_mapping,
                    self.model_loader.matrix_factorization_user_id_to_idx_mapping,
                )
        elif model_name == "light_gcn":
            return LightGCNService(
                self.model_loader.lightgcn_item_embeddings,
                self.model_loader.lightgcn_user_embeddings,
                self.model_loader.lightgcn_user_id_to_idx_mapping,
                self.model_loader.lightgcn_user_idx_to_id_mapping,
                self.model_loader.lightgcn_product_id_to_idx_mapping,
                self.model_loader.lightgcn_product_idx_to_id_mapping)

        else: 
            raise ValueError(f"Model {model_name} not supported")
        