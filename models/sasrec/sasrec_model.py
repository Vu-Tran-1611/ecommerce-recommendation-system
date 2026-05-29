from torch import nn
import torch 
class SASRecModel(nn.Module):
    def __init__(
        self,
        num_items,
        max_seq_len,
        embedding_dim=64,
        num_attention_heads=2,
        num_transformer_blocks=2,
        dropout_rate=0.1, 
        pad_id = 0
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.pad_id = pad_id
        self.num_attention_heads = num_attention_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        # 1. Item Embedding Layer
        self.item_embedding = nn.Embedding(
            num_embeddings = num_items + 1, 
            embedding_dim = embedding_dim, 
            padding_idx = 0 # Embedding for padding will be a zero vector and not updated during training
        )
        # 2. Position Embedding Layer
        self.position_embedding = nn.Embedding(
            num_embeddings = max_seq_len,
            embedding_dim = embedding_dim
        )

        self.dropout = nn.Dropout(dropout_rate) # Dropout layer for regularization

        # Transformer Blocks
        transformer_block = nn.TransformerEncoderLayer(
            # Multi-Head Self-Attention
            d_model = embedding_dim,  # Embedding dimension for the transformer, define the dim for matrices Q, K, V
            nhead = num_attention_heads, # Number of attention heads in the multi-head attention mechanism

            # Multi Layer Perceptron (MLP)
            dropout = dropout_rate, # Dropout rate for regularization
            dim_feedforward = embedding_dim * 4, # Expansion Weights 
            activation = "gelu", # Activation function for the feedforward network

            #Supplementary Arguments
            batch_first = True, # Input shape (batch_size, seq_len, embedding_dim)
            norm_first = True # Apply layer normalization before the attention and feedforward layers

        )
        # 3. Transformer Layer
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = transformer_block, # Base transformer block to be repeated
            num_layers = num_transformer_blocks, # Number of transformer blocks
            enable_nested_tensor=False
        )

        self.layer_norm = nn.LayerNorm(embedding_dim) # Layer normalization to stabilize training

        # 4. Output Layer
        self.output_layer = nn.Linear(
            embedding_dim, # Input dimension from the transformer output
            num_items + 1 # Output dimension equal to number of items
        )
    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self,input_sequences):
        device = input_sequences.device
        batch_size,seq_len = input_sequences.size() 

        positions = torch.arange(seq_len,device=device)
        positions = positions.unsqueeze(0) 
        positions = positions.expand(batch_size,seq_len) 

        item_emb = self.item_embedding(input_sequences) 
        pos_emb = self.position_embedding(positions) 
        
        x = item_emb + pos_emb 
        x = self.dropout(x) 

        # Create causal mask for self-attention to prevent attending to future positions
        causal_mask = self.create_causal_mask(seq_len, device)

        # Create padding mask to ignore padded positions in the input sequences
        padding_mask_bool = input_sequences.eq(self.pad_id) #[batch_size,seq_len]
        padding_mask = torch.zeros(batch_size, seq_len, device=device, dtype=x.dtype) #[batch_size,seq_len]
        padding_mask = padding_mask.masked_fill(padding_mask_bool, float("-inf"))  #[batch_size,seq_len]

 
        for layer in self.transformer_encoder.layers:
            x = layer(
                x, 
                src_mask=causal_mask, #(seq_len, seq_len) mask for causal attention
                src_key_padding_mask=padding_mask #(batch_size, seq_len) mask for padding positions
            )
            x = torch.nan_to_num(x, nan=0.0)

        x = self.layer_norm(x)

        # Extract the last hidden state corresponding to the last item in the sequence
        last_hidden_embeddings = x[:,-1,:] 

        # Using the last hidden state to predict the next item in the sequence
        logits = self.output_layer(last_hidden_embeddings) 

        return logits 
