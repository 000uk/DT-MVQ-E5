import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, neg_ratio=0.1, noise_scale=0.0):
        super().__init__()
        self.temperature = temperature
        self.neg_ratio = neg_ratio
        self.noise_scale = noise_scale

    def forward(self, embeddings, labels):
        """
        anchor: 닻(기준점)
        pos_sim: anchor와 pos 샘플들의 유사도 벡터
        neg_sim: anchor와 neg 샘플들의 유사도 벡터
        """
        similarity = torch.matmul(embeddings, embeddings.T) # 임배딩값 self 내적 -> 각 샘플당 유사도
        
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0) # 브로드캐스팅
        identity_mask = torch.eye(len(labels), device=labels.device).bool() # 자기 자신 제거 mask
        pos_mask = (labels_eq & (~identity_mask)).float()
        pos_sim = similarity * pos_mask
        
        sim_for_neg = similarity.clone()
        sim_for_neg.masked_fill_(labels_eq, -1e9)
        neg_sim, _ = sim_for_neg.topk(k, dim=1)

        # loss 확대: 정답(0.8/0.05=16), 오답(0.7/0.05=14) => exp(16) ≈ 8,886,110 vs exp(14) ≈ 1,202,604 7배 이상 차이남
        # => 0.1 차이도 크게 만들어 모델이 pos를 더더더 1에 가깝도록 맞춤
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature

        loss = -torch.log(
            torch.exp(pos_sim).sum(dim=1) /
            (torch.exp(pos_sim).sum(dim=1) + torch.exp(neg_sim).sum(dim=1))
        ).mean()
        # sum해주는 이유는.. 한 anchor(기준 샘플)당 여러 개의 pos/neg 쌍이 존재할 수 있기 때문
        # 그림 mean은 왜 하는거지... 배치내 모든 loss를 평균 내는것임! -> 이번 배치의 loss는 0.36다~

        return loss
    

        

    # [필살기 1] Orthogonality Check (학습 때만 계산해서 리턴)
        if self.training:
            v_genre = embeddings[:, 0, :]
            v_content = embeddings[:, 1, :]
            
            # 두 벡터가 얼마나 닮았는지(섞였는지) 계산
            ortho_loss = self.calculate_ortho_loss(v_genre, v_content)
            return embeddings, ortho_loss
            
        return embeddings

    def calculate_ortho_loss(self, v1, v2):
        # 벡터가 정규화되어 있다고 가정하면 내적(Dot Product)이 곧 코사인 유사도
        # 서로 수직이면 내적값은 0이 됨
        dot_product = torch.sum(v1 * v2, dim=-1)
        return torch.mean(dot_product ** 2)