from numpy import rec
import pandas as pd 
import numpy as np
from app.services.base_service import BaseRecommendationService
from app.schemas.recommend_schema import Interaction
from sklearn.model_selection import train_test_split
class LightGCNService(BaseRecommendationService):
    def __init__(self, lightgcn_item_embeddings, lightgcn_user_embeddings, lightgcn_user_id_to_idx_mapping, lightgcn_user_idx_to_id_mapping,
                 lightgcn_product_id_to_idx_mapping, lightgcn_product_idx_to_id_mapping):
        self.item_embeddings = lightgcn_item_embeddings
        self.user_embeddings = lightgcn_user_embeddings
        self.user_id_to_idx_mapping = lightgcn_user_id_to_idx_mapping
        self.user_idx_to_id_mapping = lightgcn_user_idx_to_id_mapping
        self.product_id_to_idx_mapping = lightgcn_product_id_to_idx_mapping
        self.product_idx_to_id_mapping = lightgcn_product_idx_to_id_mapping


    def cleaned_interactions(self,interactions):
        interactions_df = pd.DataFrame([{"product_id": i.product_id} for i in interactions])
        interactions_df.drop_duplicates(inplace=True) 
        #Split interactions into train and test sets
        # train_interactions, test_interactions = train_test_split(interactions_df, test_size=0.4, random_state=42)
        # return  interactions_df,train_interactions, test_interactions
        return interactions_df

    def build_temp_user_vector(self,interactions):
        embeddings = []
        for id in interactions:
            product_idx = self.product_id_to_idx_mapping.get(id)
            if product_idx is not None:
                embeddings.append(self.item_embeddings[product_idx])
        return np.mean(embeddings, axis=0)


    def get_seen_product_ids(self, interactions):
        seen_product_ids = set()
        for product_id in interactions:
            seen_product_ids.add(product_id)
        return seen_product_ids
    
    def metrics(self,ground_truth,recommendations, top_k:int=10):
        hits = 0
        for item in recommendations[:top_k]:
            if item in ground_truth:
                hits += 1
        precision = hits / top_k
        recall = hits / len(ground_truth) if ground_truth else 0
        return precision,recall,hits


    def recommend(self,user_id,interactions, top_k:int=10):
        # cleaned_interactions, train_interactions, test_interactions = self.cleaned_interactions(interactions)
        cleaned_interactions = self.cleaned_interactions(interactions)
        user_vector = None
        user_idx = self.user_id_to_idx_mapping.get(user_id)  
        if user_idx is None:
            user_vector = self.build_temp_user_vector(cleaned_interactions["product_id"].tolist())
        else: 
            user_vector = self.user_embeddings[user_idx]
        scores = np.dot(self.item_embeddings, user_vector).flatten()
        rank_indices = np.argsort(scores)[::-1]
        seen_product_ids = self.get_seen_product_ids(cleaned_interactions["product_id"].tolist())
        recs = []
        # all_recs = []
        for idx in rank_indices:
            product_id = self.product_idx_to_id_mapping.get(int(idx))
            # all_recs.append(product_id)
            if product_id not in seen_product_ids:
                recs.append(product_id)
            if len(recs) == top_k:
                break 
        # precision,recall,hits = self.metrics(test_interactions["product_id"].tolist(), all_recs, top_k=top_k)
        # return recs,precision,recall,hits
        return recs



