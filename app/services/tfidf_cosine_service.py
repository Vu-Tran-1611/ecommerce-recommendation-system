from app.services.base_service import BaseRecommendationService
from app.utils.similarity_utils import get_product_index, get_product_ids

class TFIDFCosineService(BaseRecommendationService):
    # def __init__(self, tfidf_model,metadata):
    #     self.tfidf_model = tfidf_model
    #     self.metadata = metadata

    # def recommend(self, product_id:int, top_k:int = 10):
    #     scores = self.tfidf_model[get_product_index(product_id, self.metadata)]
    #     similar_indices = scores.argsort()[::-1][1:top_k+1]
    #     indices_to_product_ids = get_product_ids(similar_indices, self.metadata)
    #     return indices_to_product_ids
    def __init__(self, knn_model, feature_matrix,metadata):
        self.knn_model = knn_model
        self.feature_matrix = feature_matrix
        self.metadata = metadata

    def recommend(self, product_id:int, top_k:int = 10):
        id_to_index = get_product_index(product_id, self.metadata)
        indices = self.knn_model.kneighbors(self.feature_matrix[id_to_index], return_distance=False)
        indices_to_product_ids = get_product_ids(indices[0][1:top_k+1], self.metadata)
        return indices_to_product_ids