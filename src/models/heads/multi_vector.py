import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultiVectorHead(nn.Module):
    def __init__(self, num_vectors=3,  input_dim=384):
        super().__init__()

        # k개의 학습 가능한 쿼리 토큰 생성
        self.query_tokens = nn.Parameter( # 일단 텐서랑 다르게 학습 가능함
            torch.randn(1, num_vectors, input_dim) # (1, K, D) 차원의 랜덤값
        )
        # nn.init.normal_(self.query_tokens, std=0.02)
        # nn.init.orthogonal_(self.query_tokens)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, seq_out, attn_mask):
        """
        # shapes
        query = queries  # (B, K, D)
        key = seq_out    # (B, L, D)
        value = seq_out  # (B, L, D)
        """
        batch_size = seq_out.shape[0] # (B)
        queries = self.query_tokens.repeat(batch_size, 1, 1)  # Query 확장 (1, K, D) -> (B, K, D)
        key_padding_mask = ~attn_mask.bool()

        vectors, _ = self.attention( # (B, K, D)
            query=queries,
            key=seq_out,
            value=seq_out,
            key_padding_mask=key_padding_mask
        )

        # 2. 🔥 [Pro Tip] 잔차 연결 (Residual) + LayerNorm
        # "새로 배운 정보(vectors)에 원래 내 자아(queries)를 섞는다"
        # 이렇게 하면 학습이 훨씬 안정적으로 변합니다.
        # vectors = self.norm(queries + vectors)
        return vectors

class MultiVectorHead(nn.Module): # transformer에서 self_attn만 제거함
    def __init__(self, num_vectors=3,  input_dim=384):
        super().__init__()

        # k개의 학습 가능한 쿼리 토큰 생성
        self.query_tokens = nn.Parameter( # 일단 텐서랑 다르게 학습 가능함
            torch.randn(1, num_vectors, input_dim) # (1, K, D) 차원의 랜덤값
        )
        # nn.init.normal_(self.query_tokens, std=0.02)

        # Cross-Attention (정보 수집)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # FFN (수집한 정보 가공)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(0.1)
        )

    def forward(self, seq_out, attn_mask):
        """
        # shapes
        query = queries  # (B, K, D)
        key = seq_out    # (B, L, D)
        value = seq_out  # (B, L, D)
        """
        batch_size = seq_out.shape[0] # (B)
        queries = self.query_tokens.repeat(batch_size, 1, 1)  # Query 확장 (1, K, D) -> (B, K, D)
        key_padding_mask = ~attn_mask.bool()

        attn_out, _ = self.attention(
            query=queries,
            key=seq_out,
            value=seq_out,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(queries + self.dropout(attn_out))

        # vectors = x + self.ffn(self.norm2(x))
        ffn_out = self.ffn(x)
        vectors = self.norm2(x + ffn_out)
        return vectors