from app.services.base_service import BaseRecommendationService
from app.utils.similarity_utils import get_product_index, get_product_ids

class TFIDFKNNCosineService(BaseRecommendationService):
    def __init__(self, tfidf_knn_model, tfidf_knn_feature_matrix, metadata):
        self.tfidf_knn_model = tfidf_knn_model
        self.tfidf_knn_feature_matrix = tfidf_knn_feature_matrix
        self.metadata = metadata

    def recommend(self, product_id:int, top_k:int = 10):
        id_to_index = get_product_index(product_id, self.metadata)
        indices = self.tfidf_knn_model.kneighbors(self.tfidf_knn_feature_matrix[id_to_index], return_distance=False)
        indices_to_product_ids = get_product_ids(indices[0][1:top_k+1], self.metadata)
        return indices_to_product_ids