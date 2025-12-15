class SlotAttentionHead(nn.Module):
    def __init__(self, num_vectors=2, input_dim=384, iters=3, hidden_dim=384):
        super().__init__()
        self.num_vectors = num_vectors
        self.iters = iters # 보통 3번 정도 반복해서 경쟁시킴
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.scale = hidden_dim ** -0.5

        # 1. 학습 가능한 초기 슬롯 (님 코드의 query_tokens와 동일)
        # 중요: mu(평균)와 sigma(분산)를 학습해서 매번 샘플링하는 게 원본이지만,
        # 텍스트에서는 그냥 Parameter로 고정해도 잘 됩니다.
        self.slots_mu = nn.Parameter(torch.randn(1, num_vectors, input_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, num_vectors, input_dim))
        
        # 2. Linear Projections
        self.to_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(input_dim, hidden_dim, bias=False)

        # 3. GRU (반복 업데이트의 핵심)
        # 슬롯이 정보를 먹고 -> 업데이트하고 -> 다시 정보를 먹는 과정
        self.gru = nn.GRUCell(hidden_dim, input_dim)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, inputs, attn_mask=None):
        """
        inputs: (B, L, D) - E5의 출력
        """
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        
        # Key, Value는 미리 만들어둠
        k = self.to_k(inputs) # (B, L, D)
        v = self.to_v(inputs) # (B, L, D)

        # 초기 슬롯 생성 (Gaussian Sampling)
        mu = self.slots_mu.expand(b, self.num_vectors, -1)
        sigma = self.slots_log_sigma.expand(b, self.num_vectors, -1).exp()
        slots = mu + sigma * torch.randn_like(mu)

        # Iterative Routing (경쟁 시작!)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Query 생성
            q = self.to_q(slots) # (B, K, D)

            # Attention Score
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            # 🔥 여기가 핵심 차이점! 🔥
            # 일반 어텐션: Softmax(dim=-1) -> Key(입력 단어) 축으로 확률 계산
            # Slot 어텐션: Softmax(dim=1)  -> Slot(쿼리) 축으로 확률 계산
            # 의미: "이 단어는 내꺼야!" 라고 슬롯끼리 경쟁함
            attn = dots.softmax(dim=1) + 1e-8 # (B, K, L)
            
            # Weighted Sum (근데 이제 정규화를 곁들인)
            # 특정 슬롯이 정보를 너무 독점하지 않게 나눠줌
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            updates = torch.einsum('bjd,bij->bid', v, attn)

            # GRU로 슬롯 업데이트 (잔차 연결 느낌)
            # GRUCell은 (Batch * Num_Slots, Dim) 형태를 받음
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_vectors, d)
            
            # Optional MLP
            slots = slots + self.mlp(self.norm_slots(slots))

        return slots # (B, 2, D) -> v0, v1