from math import pi

import numpy
import torch 
import torch.nn.functional as F

from app.utils import similarity_utils
class TwoTowerService: 
    def __init__(self, 
                 user_id_to_index, 
                 user_index_to_id, 
                 product_id_to_index, 
                 product_index_to_id, 
                 vector_embeddings):
        self.user_id_to_index = user_id_to_index
        self.user_index_to_id = user_index_to_id
        self.product_id_to_index = product_id_to_index
        self.product_index_to_id = product_index_to_id
        self.vector_embeddings = vector_embeddings

    def build_temp_user_vector(self, interactions):
        product_indices = [self.product_id_to_index[interaction.product_id] 
                           for interaction in interactions 
                           if interaction.product_id in self.product_id_to_index] 
        #Remove dublicate product indices
        seen_product_indices = list(set(product_indices))

        if not product_indices:
            return None 
        product_vectors = self.vector_embeddings[seen_product_indices]
        temp_user_vector = torch.mean(product_vectors, dim=0,keepdim=True)
        temp_user_vector = F.normalize(temp_user_vector, dim=1)
        return temp_user_vector,seen_product_indices
    
    def recommend(self,user_id, interactions, top_k=10):
        temp_user_vector, seen_product_indices = self.build_temp_user_vector(interactions)

        if temp_user_vector is None:
            return []
        all_product_vectors = self.vector_embeddings
        similarity_scores = temp_user_vector @ all_product_vectors.T #[1,num_products] <= [1,embedding_dim] @ [embedding_dim,num_products]
        similarity_scores = similarity_scores.squeeze(0) #[num_products] <= [1,num_products]

        # Remove products the user has already interacted with
        similarity_scores[seen_product_indices] = float("-inf")
     
        top_scores, top_indices = torch.topk(similarity_scores, k=top_k) 
        recommended_product_ids = [self.product_index_to_id[idx.item()] for idx in top_indices ]

        # Test
        print("Top Id", recommended_product_ids)
        # Test
        
        return recommended_product_ids
    

