from torch import nn
import torch 
import torch.nn.functional as F
class MultiInterestExtractorSA(nn.Module): 
    def __init__(self,embedding_dim,hidden_dim,num_interests,dropout_rate=0.2): 
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_interests = num_interests

        self.W1 = nn.Linear(embedding_dim, hidden_dim, bias=False) # D => hD 
        self.W2 = nn.Linear(hidden_dim,num_interests, bias=False) # hD => K

        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,H,padding_mask=None): 
        hidden = torch.tanh(self.W1(H)) #[B,L,hD]
        hidden = self.dropout(hidden) 

        hidden = self.W2(hidden) #[B,L,K] 
        hidden = hidden.transpose(1,2) #[B,K,L] 

        if padding_mask is not None: 
            padding_mask = padding_mask.unsqueeze(1) #[B,1,L] <= [B,L] 
            padding_mask = padding_mask.expand(-1,self.num_interests,-1) #[B,K,L] <= [B,1,L]
            hidden = hidden.masked_fill(padding_mask,float("-inf")) 
        
        A = F.softmax(hidden,dim=-1) #[B,K,L]
        
        V = torch.bmm(A,H) # [B,K,D] <= [B,K,L] * [B,L,D] 
        return V,A 
class ComiRecSAModel(nn.Module): 
    def __init__(self,num_items,max_seq_len,embedding_dim,hidden_dim,num_interests,dropout_rate=0.2,pad_id=0): 
        super().__init__()
        self.num_items = num_items 
        self.max_seq_len = max_seq_len 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        self.num_interests = num_interests 
        self.pad_id = pad_id 
        self.vocab_size = num_items + 1 

        self.item_embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_dim,
            padding_idx = self.pad_id
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings = self.max_seq_len,
            embedding_dim = self.embedding_dim
        )

        self.dropout = nn.Dropout(dropout_rate) 
        self.layer_norm = nn.LayerNorm(self.embedding_dim) 

        self.interest_extractor = MultiInterestExtractorSA(
            embedding_dim = self.embedding_dim,
            hidden_dim = self.hidden_dim,
            num_interests = self.num_interests,
            dropout_rate = dropout_rate
        )
    def forward(self,input_sequences):
        # Input: input_sequences [B,L]
        device = input_sequences.device
        batch_size, seq_len = input_sequences.size() 

        positions = torch.arange(seq_len, device = device)  #[L]
        positions = positions.unsqueeze(0) # [1,L] <= [L]
        positions = positions.expand(batch_size,-1) #[B,L] <= [1,L] 

        item_emb = self.item_embedding(input_sequences) #[B,L,D]
        pos_emb = self.positional_embedding(positions) #[B,L,D] 

        H = item_emb + pos_emb #[B,L,D]
        H = self.dropout(H)
        H = self.layer_norm(H) #[B,L,D] 

        padding_mask = input_sequences.eq(self.pad_id) #[B,L] 
        V,A = self.interest_extractor(H,padding_mask) #[B,K,D],[B,K,L]
        
        return V,A
    def compute_loss(self, input_sequences, target_items):  
        # Input: input_sequences [B,L], target_items [B]
        V, A = self.forward(input_sequences) #[B,K,D],[B,K,L] 
        target_emb = self.item_embedding(target_items) #[B,D]

        # 1. Compute the dot product between target_emb and each interest vector in V
        target_emb = target_emb.unsqueeze(1) #[B,1,D] <= [B,D]
        target_emb = target_emb.expand(-1,self.num_interests,-1) #[B,K,D] <= [B,1,D]
        interest_scores = torch.sum(V * target_emb, dim=-1) #[B,K] <= sum([B,K,D] * [B,K,D], dim=-1)

        # 2. Select the best interest vector that has the highest score for each sample/user 
        best_interest_idx = torch.argmax(interest_scores, dim=-1) #[B] <= argmax([B,K], dim=-1)
        batch_indices = torch.arange(V.size(0), device=V.device) #[B] 
        best_interest = V[batch_indices,best_interest_idx,:] #[B,D] <= V[[B],[B],:] 

        # 3. Compute the loss by dot product between best_interest and all item embeddings
        all_item_emb = self.item_embedding.weight #[num_items+1,D] 
        all_item_emb = all_item_emb.T #[D,num_items+1]
        logits = best_interest @ all_item_emb #[B,num_items+1] <= [B,D] @ [D,num_items+1] 
        logits[:,self.pad_id] = float("-inf")  # Mask the padding index
        loss = F.cross_entropy(logits, target_items) # scalar <= cross_entropy([B,num_items+1],[B])

        return loss
