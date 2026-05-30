from math import pi
from re import sub

import numpy
import torch 
import torch.nn.functional as F
import random
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
    
    # Classify interactions based on its category_id 

    def group_interactions_by_category(self, interactions):
        category_interactions = {}
        for interaction in interactions:
            category_id = interaction.category_id
            if category_id not in category_interactions:
                category_interactions[category_id] = []
            category_interactions[category_id].append(interaction)
        return category_interactions

    def recommend(self,user_id, interactions, top_k=10):
        
        category_interactions = self.group_interactions_by_category(interactions)
        all_recommended_product_indices = set() 
        catergory_based_recommendations = {}
        divided_k = max(10, top_k // len(category_interactions))  
        for category_id, interactions in category_interactions.items():
            temp_user_vector, seen_product_indices = self.build_temp_user_vector(interactions)
            if temp_user_vector is None:
                continue 
            similarity_scores = temp_user_vector @ self.vector_embeddings.T #[1,num_products] <= [1,embedding_dim] @ [embedding_dim,num_products]
            similarity_scores = similarity_scores.squeeze(0) #[num_products] <= [1,num_products]
            # Remove products the user has already interacted with in this sub-category
            similarity_scores[seen_product_indices] = float("-inf")
            # Remove products the user has already interacted with in other sub-categories as well
            similarity_scores[list(all_recommended_product_indices)] = float("-inf")
            top_k_indices = torch.topk(similarity_scores, k=top_k).indices.tolist()
            catergory_based_recommendations[category_id] = top_k_indices
        
        for category_id, recommended_indices in catergory_based_recommendations.items():
            all_recommended_product_indices.update(recommended_indices[:divided_k])  # Add top-k recommendations from this sub-category
        all_recommended_product_indices = list(all_recommended_product_indices)
        random.shuffle(all_recommended_product_indices)  # Shuffle the final recommendations to mix products from different sub-categories
        all_recommended_product_indices = all_recommended_product_indices[:top_k]  # Limit to top_k recommendations
        recommended_product_ids = [self.product_index_to_id[idx] for idx in all_recommended_product_indices]    
        
        return recommended_product_ids
        

        # Old code without sub-category grouping
        # temp_user_vector, seen_product_indices = self.build_temp_user_vector(interactions)
            
        # if temp_user_vector is None:
        #     return []
        # all_product_vectors = self.vector_embeddings
        # similarity_scores = temp_user_vector @ all_product_vectors.T #[1,num_products] <= [1,embedding_dim] @ [embedding_dim,num_products]
        # similarity_scores = similarity_scores.squeeze(0) #[num_products] <= [1,num_products]

        # # Remove products the user has already interacted with
        # similarity_scores[seen_product_indices] = float("-inf")
     
        # top_scores, top_indices = torch.topk(similarity_scores, k=top_k) 
        # recommended_product_ids = [self.product_index_to_id[idx.item()] for idx in top_indices ]


    

