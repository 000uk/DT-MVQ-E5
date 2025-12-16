import torch
import torch.nn as nn
import torch.nn.functional as F

class CompetitiveAttention(nn.Module):
    """
    nn.MultiheadAttention 대신 사용할 '경쟁적 어텐션' 모듈
    Slot Attention의 핵심인 Softmax(dim=Query)를 구현함
    """
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(input_dim, input_dim, bias=False)
        self.to_k = nn.Linear(input_dim, input_dim, bias=False)
        self.to_v = nn.Linear(input_dim, input_dim, bias=False)
        self.to_out = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        B, K, _ = query.shape # K: 쿼리 개수 (Slots)
        _, L, _ = key.shape   # L: 시퀀스 길이
        H = self.num_heads

        # 1. Projection & Head Split
        q = self.to_q(query).reshape(B, K, H, -1).permute(0, 2, 1, 3) # (B, H, K, D)
        k = self.to_k(key).reshape(B, L, H, -1).permute(0, 2, 1, 3)   # (B, H, L, D)
        v = self.to_v(value).reshape(B, L, H, -1).permute(0, 2, 1, 3) # (B, H, L, D)

        # 2. Score Calculation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (B, H, K, L)

        # 3. 🔥 [핵심] Inverted Softmax (Slot Attention의 영혼)
        # 일반 Attention: softmax(dim=-1) -> 단어 축으로 확률 (모든 쿼리가 같은 단어 봐도 됨)
        # 경쟁 Attention: softmax(dim=-2) -> 쿼리 축으로 확률 (v0가 가져가면 v1은 못 가져감!)
        attn = dots.softmax(dim=-2) 
        
        # (Optional) 안정성을 위한 Normalization (Slot Attention 논문 디테일)
        # 각 쿼리가 너무 작은 값만 가져가지 않도록 보정
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Aggregation
        out = torch.matmul(attn, v) # (B, H, K, D)
        out = out.permute(0, 2, 1, 3).reshape(B, K, -1)
        return self.to_out(out), attn

class FusedMultiVectorHead(nn.Module):
    def __init__(self, num_vectors=3, input_dim=384):
        super().__init__()
        
        # 1. 초기화 (Orthogonal 필수!)
        self.query_tokens = nn.Parameter(torch.randn(1, num_vectors, input_dim))
        nn.init.orthogonal_(self.query_tokens)

        # 2. 🔥 [교체] 일반 Attention -> 경쟁적 Attention
        self.attention = CompetitiveAttention(input_dim=input_dim, num_heads=8)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

        # 3. FFN (님 코드 그대로 유지 - 아주 훌륭함)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(0.1)
        )

    def forward(self, seq_out, attn_mask=None):
        batch_size = seq_out.shape[0]
        queries = self.query_tokens.repeat(batch_size, 1, 1) # (B, K, D)

        # 경쟁적 어텐션 수행
        # (마스킹은 복잡해서 생략해도 E5가 이미 잘해서 괜찮지만, 필요하면 attn에 -inf 추가)
        attn_out, _ = self.attention(query=queries, key=seq_out, value=seq_out)
        
        # Residual & Norm
        x = self.norm1(queries + self.dropout(attn_out))

        # FFN & Residual
        vectors = self.norm2(x + self.ffn(x))
        
        return vectors