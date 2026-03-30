from app.services.knn_euclidean_service import KNNEuclideanService
from app.services.tfidf_cosine_service import TFIDFCosineService
from app.services.tfidf_knn_cosine_service import TFIDFKNNCosineService 
# from app.services.matrix_factorization_service import MatrixFactorizationService
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
            return TFIDFCosineService(self.model_loader.tfidf_model,
                                       self.model_loader.metadata)
        elif model_name == "tfidf_knn_cosine": 
            return TFIDFKNNCosineService(
                self.model_loader.tfidf_knn_model, 
                self.model_loader.tfidf_knn_feature_matrix,
                self.model_loader.metadata
            )
        elif model_name == "matrix_factorization":
            if version == "recent": 
                return MatrixFactorizationRecentService(
                    self.model_loader.matrix_factorization_model,
                    self.model_loader.matrix_factorization_product_idx_to_id_mapping, 
                    self.model_loader.matrix_factorization_product_id_to_idx_mapping,
                )
            # else:
            #     return MatrixFactorizationService()
        else: 
            raise ValueError(f"Model {model_name} not supported")
        