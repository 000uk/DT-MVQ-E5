import torch
import torch.nn as nn
import torch.nn.functional as F

# Slot Attention (Google DeepMind, 2020)에서 영향 받음
# 단 GRU는 뺌! text로 검색하는 task니까 굳이 싶어서...
class CompetitiveVectorHead(nn.Module):
    def __init__(self, num_vectors=2, input_dim=384, num_heads=8):
        super().__init__()
        self.num_vectors = num_vectors
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_tokens = nn.Parameter(torch.randn(1, num_vectors, input_dim))
        nn.init.orthogonal_(self.query_tokens)

        self.to_q = nn.Linear(input_dim, input_dim, bias=False)
        self.to_k = nn.Linear(input_dim, input_dim, bias=False)
        self.to_v = nn.Linear(input_dim, input_dim, bias=False)

        self.to_out = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(0.1)
        )

    def forward(self, seq_out, attention_mask=None):
        """
        seq_out: (B, L, D) - Backbone Output
        attention_mask: (B, L) - 1 for content, 0 for padding (HuggingFace Style)
        """
        B, L, D = seq_out.shape
        K = self.num_vectors
        q = self.to_q(self.query_tokens.repeat(B, 1, 1)).reshaㅋpe(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(seq_out).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(seq_out).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-2) 
        mask_expanded = attention_mask.view(B, 1, 1, L).float()
        attn = attn * mask_expanded

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, K, D)
        out = self.to_out(out)

        queries = self.query_tokens.repeat(B, 1, 1)
        x = self.norm1(queries + out)
        vectors = self.norm2(x + self.ffn(x))

        return vectors