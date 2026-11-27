from app.services.light_gcn_service import LightGCNService
from app.services.sasrec_service import SasRecService
from app.services.similar_products_service import SimilarProductsService
from app.services.matrix_factorization_service import MatrixFactorizationRecentService 
from app.services.bert4rec_service import Bert4RecService
from app.services.comirec_service import ComiRecService 
from app.services.twotower_service import TwoTowerService
class RecommendationServiceFactory: 
    def __init__(self,model_loader):
        self.model_loader = model_loader 
    def get_service(self,model_name:str,version=None): 
        # "tfidf_knn_cosine" is a temporary alias until Laravel uses "similar_products"
        if model_name in ("similar_products", "tfidf_knn_cosine"):
            return SimilarProductsService(
                self.model_loader.similar_products_model,
                self.model_loader.similar_products_feature_matrix,
                self.model_loader.similar_products_metadata,
            )
        elif model_name == "matrix_factorization":
            if version == "recent" or version is None: 
                return MatrixFactorizationRecentService(
                    self.model_loader.matrix_factorization_item_factors,
                    self.model_loader.matrix_factorization_user_factors,
                    self.model_loader.matrix_factorization_product_idx_to_id_mapping, 
                    self.model_loader.matrix_factorization_product_id_to_idx_mapping,
                    self.model_loader.matrix_factorization_user_id_to_idx_mapping,
                )
        elif model_name == "light_gcn":
            return LightGCNService(
                self.model_loader.lightgcn_item_embeddings,
                self.model_loader.lightgcn_user_embeddings,
                self.model_loader.lightgcn_user_id_to_idx_mapping,
                self.model_loader.lightgcn_user_idx_to_id_mapping,
                self.model_loader.lightgcn_product_id_to_idx_mapping,
                self.model_loader.lightgcn_product_idx_to_id_mapping)
        elif model_name == "sasrec":
            return SasRecService( 
                self.model_loader.sasrec_checkpoint,
                self.model_loader.sasrec_user_id_to_index,
                self.model_loader.sasrec_item_id_to_index,
                self.model_loader.sasrec_item_index_to_id
            )
        elif model_name == "bert4rec":
            return Bert4RecService(
                self.model_loader.bert4rec_checkpoint,
                self.model_loader.bert4rec_user_id_to_index,
                self.model_loader.bert4rec_item_id_to_index,
                self.model_loader.bert4rec_item_index_to_id
            )
        elif model_name == "comirec":
            return ComiRecService(
                self.model_loader.comirec_checkpoint,
                self.model_loader.comirec_user_id_to_index,
                self.model_loader.comirec_item_id_to_index,
                self.model_loader.comirec_item_index_to_id
            )
        elif model_name == "twotower":
            return TwoTowerService(
                self.model_loader.twotower_user_id_to_index,
                self.model_loader.twotower_user_index_to_id,
                self.model_loader.twotower_product_id_to_index,
                self.model_loader.twotower_product_index_to_id,
                self.model_loader.twotower_vector_embeddings
            )
        else: 
            raise ValueError(f"Model {model_name} not supported")
        