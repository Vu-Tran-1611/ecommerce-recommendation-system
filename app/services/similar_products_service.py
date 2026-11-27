from app.services.base_service import BaseRecommendationService
from app.utils.similarity_utils import get_product_index, get_product_ids


# Content-based similar products: TF-IDF text + one-hot attributes + price, cosine KNN
class SimilarProductsService(BaseRecommendationService):
    def __init__(self, knn_model, feature_matrix, metadata):
        self.knn_model = knn_model
        self.feature_matrix = feature_matrix
        self.metadata = metadata

    # Return the products most similar to a product
    def recommend(self, product_id: int, top_k: int = 10):
        index = get_product_index(product_id, self.metadata)
        indices = self.knn_model.kneighbors(self.feature_matrix[index], return_distance=False)
        # Skip the first neighbour, which is the product itself
        return get_product_ids(indices[0][1:top_k + 1], self.metadata)
