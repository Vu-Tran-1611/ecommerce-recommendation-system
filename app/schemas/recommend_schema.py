from unicodedata import category

from pydantic import BaseModel 
from typing import Literal
# Content-based recommendation request and response schemas
class RecommendationRequest(BaseModel): 
    product_id:int 
    model_name:Literal[
        "knn_euclidean",
        "tfidf_cosine",
        "tfidf_knn_cosine",
    ]
    top_k:int = 10,
    version:Literal[
        "v1",
        "v2", 
        None
    ] = None

class RecommendationResponse(BaseModel): 
    product_id:int 
    model_name:str 
    recommendations:list[int] 

# User-based recommendation request and response schemas 
# Recent Interactions-based recommendation
class Interaction(BaseModel):
        product_id:int
        category_id:int
        interaction_type:Literal[
            "click",
            "wishlist_add",
            "cart_add",
            "R5",
            "R4",
            "R3",
            "R2",
            "R1"
        ]
class UserRecentRecommendationRequest(BaseModel):
    user_id:int
    interactions:list[Interaction]  
    model_name:Literal[
        "matrix_factorization",
        "light_gcn", 
        "sasrec", 
        "bert4rec", 
        "comirec", 
        "twotower"
    ]
    top_k:int = 10 

class UserRecommendationResponse(BaseModel):
    model_name:str
    recommendations:list[int]
    user_id:int

    # precision:float
    # recall:float
    # hits:int




#2 Retrained-interactions-based recommendation 

# class UserRetrainingRecommendationRequest(BaseModel):
#     user_id:int
#     model_name:Literal[
#         "matrix_factorization",
#         "light_gcn"
#     ]
#     top_k:int = 10 