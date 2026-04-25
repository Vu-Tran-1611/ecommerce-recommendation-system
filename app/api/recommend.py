from fastapi import APIRouter, HTTPException 
from app.services.service_factory import RecommendationServiceFactory
from app.loaders.model_loader import ModelLoader 
from app.schemas.recommend_schema import RecommendationRequest,RecommendationResponse, UserRecentRecommendationRequest, UserRecommendationResponse
router = APIRouter(prefix="/api", tags=["recommendations"])  
model_loader = ModelLoader() 
service_factory = RecommendationServiceFactory(model_loader) 
# Content-based recommendation endpoint

@router.post("/recommend",response_model = RecommendationResponse) 
def recommend(request: RecommendationRequest): 
    try: 
        service = service_factory.get_service(request.model_name,request.version)
        recommendations = service.recommend(
            product_id = request.product_id, 
            top_k = request.top_k
        )
        return RecommendationResponse(
            product_id = request.product_id,
            model_name = request.model_name,
            recommendations = recommendations,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
#user-based recommendation endpoint based on recent interactions

# 1. Recent Interactions-based recommendation
@router.post("/recommend/recent",response_model = UserRecommendationResponse)
def recommend_recent(request: UserRecentRecommendationRequest):
    try:
        service = service_factory.get_service(request.model_name, version="recent")
        recommendations = service.recommend(
            user_id = request.user_id,
            interactions = request.interactions,
            top_k = request.top_k
        )
        return UserRecommendationResponse(
            model_name = request.model_name,
            recommendations = recommendations[0],
            # precision = recommendations[1],
            # recall = recommendations[2], 
            # hits = recommendations[3],
            user_id = request.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        




        
# #2. Retrained-interactions-based recommendation
# @router.post("/recommend/retrain", response_model=UserRecommendationResponse)
# def recommend_retrain(request: UserRetrainingRecommendationRequest):
#     try:
#         service = service_factory.get_service(request.model_name, version="retrain")
#         recommendations = service.recommend(
#             user_id=request.user_id,
#             top_k=request.top_k
#         )
#         return UserRecommendationResponse(
#             model_name=request.model_name,
#             recommendations=recommendations[0],
#             precision = recommendations[1],
#             recall = recommendations[2],
#             hits = recommendations[3],
#             user_id = request.user_id

#         )
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))