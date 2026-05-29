
from numpy import pad
import torch

from app.services.base_service import BaseRecommendationService
from models.comirec import comirec_model
class ComiRecService(BaseRecommendationService):
    def __init__(self, comirec_checkpoint, user_id_to_index, item_id_to_index, item_index_to_id):
        self.comirec_checkpoint = comirec_checkpoint
        self.user_id_to_index = user_id_to_index
        self.item_id_to_index = item_id_to_index
        self.item_index_to_id = item_index_to_id
        self.max_seq_len = int(self.comirec_checkpoint["max_seq_len"])
        self.pad_id = int(self.comirec_checkpoint["pad_id"])

        checkpoint_state = self.comirec_checkpoint["model_state_dict"]

        self.model = comirec_model.ComiRecSAModel(
            num_items= int(self.comirec_checkpoint["num_items"]),
            max_seq_len=int(self.comirec_checkpoint["max_seq_len"]),
            embedding_dim=int(self.comirec_checkpoint["embedding_dim"]),
            hidden_dim=int(self.comirec_checkpoint["hidden_dim"]),
            num_interests=int(self.comirec_checkpoint["num_interests"]),
            dropout_rate=float(self.comirec_checkpoint["dropout_rate"]),
            pad_id=self.pad_id,
        )
        self.model.load_state_dict(checkpoint_state)

    def build_interactions_sequence(self, interactions):
        user_sequence = [self.item_id_to_index[item.product_id] for item in interactions]
        #Remove similar items from the sequence
        user_sequence = list(dict.fromkeys(user_sequence))
        #Reverse the sequence to prioritize recent interactions
        user_sequence.reverse()
        
        if len(user_sequence) > self.max_seq_len: 
            user_sequence = user_sequence[-self.max_seq_len:] 
        padding_length = self.max_seq_len - len(user_sequence)
        padded_sequence = [self.pad_id] * padding_length + user_sequence
        return padded_sequence,user_sequence

    def recommend(self, user_id, interactions, top_k=10):
        self.model.to("cpu")
        self.model.eval()
        padded_sequence, user_sequence = self.build_interactions_sequence(interactions)
        
        input_tensor = torch.tensor([padded_sequence],dtype=torch.long).to("cpu") #[1,L] <= [L]

        with torch.no_grad():
            V, A = self.model(input_tensor) #[1,K,D],[1,K,L] 

            all_item_emb = self.model.item_embedding.weight #[num_items+1, D]
            all_item_emb = all_item_emb.T #[D, num_items+1]

            V = V.squeeze(0) #[K,D] <= [1,K,D] 

            scores_by_interest = V @ all_item_emb #[K,num_items+1] <= [K,D] @ [D,num_items+1] 

            final_scores,best_interest_for_item = torch.max(scores_by_interest,dim=0) #[num_items+1] <= max([K,num_items+1],dim=0) 
        final_scores[self.pad_id] = float("-inf")  # Mask the padding index 

        for item_id in user_sequence: 
            final_scores[item_id] = float("-inf")  # Mask already interacted items
    
        top_scores,top_items = torch.topk(final_scores,k=top_k)
        
        # Testing  
        print("scores:", final_scores)
        print("padded_sequence:", padded_sequence) 
        print("top_items:", top_items.tolist())
        # Testing 

        return [self.item_index_to_id[item_index] for item_index in top_items.tolist()]