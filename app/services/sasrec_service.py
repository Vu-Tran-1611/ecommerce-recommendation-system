
import torch

from app.services.base_service import BaseRecommendationService
from models.sasrec import sasrec_model


class SasRecService(BaseRecommendationService):
    def __init__(self, sasrec_checkpoint, user_id_to_index, item_id_to_index, item_index_to_id):
        self.sasrec_checkpoint = sasrec_checkpoint
        self.user_id_to_index = user_id_to_index
        self.item_id_to_index = item_id_to_index
        self.item_index_to_id = item_index_to_id
        self.max_seq_len = int(self.sasrec_checkpoint["max_seq_len"])
        self.pad_id = int(self.sasrec_checkpoint["pad_id"])
        checkpoint_state = self.sasrec_checkpoint["model_state_dict"]

        self.model = sasrec_model.SASRecModel(
            num_items= int(self.sasrec_checkpoint["num_items"]),
            max_seq_len=int(self.sasrec_checkpoint["max_seq_len"]),
            embedding_dim=int(self.sasrec_checkpoint["embedding_dim"]),
            num_attention_heads=int(self.sasrec_checkpoint["num_attention_heads"]),
            num_transformer_blocks=int(self.sasrec_checkpoint["num_transformer_blocks"]),
            dropout_rate=float(self.sasrec_checkpoint["dropout_rate"]),
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
        input_tensor = torch.tensor(
            [padded_sequence], 
            dtype = torch.long,
            device = "cpu"
        )
        with torch.no_grad():
            logits = self.model(input_tensor) 
            scores = logits[0] 
        
        scores[self.pad_id] = float("-inf")

        for item_index in set(user_sequence): 
            scores[item_index] = float("-inf")
        top_scores,top_items = torch.topk(scores,k=top_k)
        
        # Testing 
        print("padded_sequence:", padded_sequence) 
        print("top_items:", top_items.tolist())
        # Testing 

        return [self.item_index_to_id[item_index] for item_index in top_items.tolist()]