from app.services.base_service import BaseRecommendationService 
from app.utils.similarity_utils import get_product_index, get_product_ids

class KNNEuclideanService(BaseRecommendationService):
    def __init__(self, knn_model, feature_matrix,metadata):
        self.knn_model = knn_model
        self.feature_matrix = feature_matrix
        self.metadata = metadata

    def recommend(self, product_id:int, top_k:int = 10):
        id_to_index = get_product_index(product_id, self.metadata)
        indices = self.knn_model.kneighbors(self.feature_matrix[id_to_index], return_distance=False)
        indices_to_product_ids = get_product_ids(indices[0][1:top_k+1], self.metadata)
        return indices_to_product_ids
