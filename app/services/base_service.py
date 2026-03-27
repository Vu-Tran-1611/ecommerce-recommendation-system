from abc import ABC,abstractmethod 

class BaseRecommendationService(ABC): 
    @abstractmethod 
    def recommend(self,product_id:int,top_k:int = 10): 
        pass 