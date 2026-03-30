from app.schemas.recommend_schema import Interaction
from app.services.base_service import BaseRecommendationService
from app.utils.similarity_utils import get_product_index, get_product_ids
import numpy as np
# from training.train_matrix_factorization import train_model
class MatrixFactorizationRecentService(BaseRecommendationService):
    def __init__(self, matrix_factorization_model, matrix_factorization_product_idx_to_id, matrix_factorization_product_id_to_idx):
        self.matrix_factorization_model = matrix_factorization_model
        self.matrix_factorization_product_idx_to_id = matrix_factorization_product_idx_to_id
        self.matrix_factorization_product_id_to_idx = matrix_factorization_product_id_to_idx
    
    def get_mapped_interactions(self, interactions:list[Interaction]):
        mapped_interactions = []
        for interaction in interactions:
            product_id = interaction.product_id
            interaction_type = interaction.interaction_type
            if interaction_type == "click":
                mapped_interactions.append((product_id, 0.05))
            elif interaction_type == "wishlist_add":
                mapped_interactions.append((product_id, 0.7))
            elif interaction_type == "cart_add":
                mapped_interactions.append((product_id, 1)) 
            elif interaction_type == "R5":
                mapped_interactions.append((product_id, 0.9))
            elif interaction_type == "R4":
                mapped_interactions.append((product_id, 0.8))
            elif interaction_type == "R3":
                mapped_interactions.append((product_id, 0.6))
            elif interaction_type == "R2":
                mapped_interactions.append((product_id, 0.4))
            elif interaction_type == "R1":
                mapped_interactions.append((product_id, 0.2))
        return mapped_interactions
    def build_temp_user_vector(self, interactions:list[dict]):
        weighted_vectors = []
        weights = [] 
        for product_id,weight in interactions:
            idx = self.matrix_factorization_product_id_to_idx[product_id] 
            weighted_vectors.append(self.matrix_factorization_model.item_factors[idx] * weight)
            weights.append(weight)
        return np.sum(weighted_vectors, axis=0) / np.sum(weights)

    def get_seen_product_ids(self, interactions:list[dict]):
        seen_product_ids = set()
        for product_id, _ in interactions:
            seen_product_ids.add(product_id)
        return seen_product_ids

    def recommend(self, interactions:list[dict], top_k:int = 10):
        mapped_interactions = self.get_mapped_interactions(interactions)
        scores = self.matrix_factorization_model.item_factors.dot(self.build_temp_user_vector(mapped_interactions)).flatten()
        rank_indices = np.argsort(scores)[::-1]
        seen_product_ids = self.get_seen_product_ids(mapped_interactions)
        recs = []
        for idx in rank_indices:
            product_id = self.matrix_factorization_product_idx_to_id[int(idx)]
            if product_id not in seen_product_ids:
                recs.append(product_id)
            if len(recs) == top_k:
                break
        return recs

# class MatrixFactorizationService(BaseRecommendationService):
#     def recommend(self, user_id:int, top_k:int = 10): 
#             model,matrix,product_idx_to_id,product_id_to_idx,user_id_to_idx = train_model()
#             user_idx = user_id_to_idx[user_id]
#             product_indices,scores = model.recommend(
#                 userid = user_idx, 
#                 user_items = matrix[user_idx], 
#                 N=top_k
#             )           
#             recommended_product_ids = [product_idx_to_id[i] for i in product_indices]
#             return recommended_product_ids