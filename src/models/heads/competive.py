import torch
import torch.nn as nn
import torch.nn.functional as F

class CompetitiveVectorHead(nn.Module):
    def __init__(self, num_vectors=2, input_dim=384, num_heads=8):
        super().__init__()
        self.num_vectors = num_vectors
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. 학습 가능한 쿼리 (Orthogonal Init 필수!)
        self.query_tokens = nn.Parameter(torch.randn(1, num_vectors, input_dim))
        nn.init.orthogonal_(self.query_tokens)

        # 2. Linear Layers (Q, K, V)
        self.to_q = nn.Linear(input_dim, input_dim, bias=False)
        self.to_k = nn.Linear(input_dim, input_dim, bias=False)
        self.to_v = nn.Linear(input_dim, input_dim, bias=False)

        # 3. Output Projection & Norm
        self.to_out = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 4. FFN (기존과 동일)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, seq_out, attn_mask=None):
        """
        seq_out: (B, L, D) - E5 Output
        """
        B, L, D = seq_out.shape
        K = self.num_vectors

        # 1. Q, K, V 생성 및 Head 분리
        # (B, K, H, Dh) 형태로 변환
        q = self.to_q(self.query_tokens.repeat(B, 1, 1)).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(seq_out).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(seq_out).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. Attention Score 계산
        # (B, H, K, Dh) @ (B, H, Dh, L) -> (B, H, K, L)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # ---------------------------------------------------------
        # 🍒 [핵심] 여기가 바로 Cherry-Pick 포인트! 🍒
        # ---------------------------------------------------------
        # 일반 Attention: softmax(dim=-1) -> 단어(L) 축으로 확률 계산 (모두가 같은 단어 봐도 됨)
        # Competitive:    softmax(dim=-2) -> 쿼리(K) 축으로 확률 계산 (단어 하나를 두고 K개가 싸움)
        
        # 해석: "입력 단어 하나(Key)가 1.0의 정보를 가지고 있을 때, v0와 v1이 나눠 가져라!"
        # v0가 0.9 가져가면 v1은 0.1밖에 못 가져감 -> 강제 분리 효과
        attn = dots.softmax(dim=-2) 
        
        # (선택) 안정성을 위한 정규화 (Slot Attention 논문 디테일)
        # 각 쿼리가 가져간 정보 총량으로 나눠줌 (너무 커지지 않게)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Weighted Sum
        # (B, H, K, L) @ (B, H, L, Dh) -> (B, H, K, Dh)
        out = torch.matmul(attn, v)

        # 4. Reshape & Projection
        out = out.permute(0, 2, 1, 3).reshape(B, K, D)
        out = self.to_out(out)

        # 5. Residual & FFN
        # 기존 쿼리에 더해줌 (Perceiver 방식)
        queries = self.query_tokens.repeat(B, 1, 1)
        x = self.norm1(queries + out)
        vectors = self.norm2(x + self.ffn(x))

        return vectors