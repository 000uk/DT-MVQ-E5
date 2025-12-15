import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, neg_ratio=0.2):
        super().__init__()
        self.temperature = temperature
        self.neg_ratio = neg_ratio

    def forward(self, embeddings, labels):
        """
        anchor: 닻(기준점)
        pos_sim: anchor와 pos 샘플들의 유사도 벡터
        neg_sim: anchor와 neg 샘플들의 유사도 벡터
        """
        k = max(3, int(embeddings.size(0) * self.neg_ratio))

        similarity = torch.matmul(embeddings, embeddings.T) # 임배딩값 self 내적 -> 각 샘플당 유사도
        
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0) # 브로드캐스팅
        identity_mask = torch.eye(len(labels), device=labels.device).bool() # 자기 자신 제거 mask
        pos_mask = (labels_eq & (~identity_mask)).float()
        pos_sim = similarity * pos_mask
        
        sim_for_neg = similarity.clone()
        sim_for_neg.masked_fill_(labels_eq, -1e9)

        # neg_sim = similarity[labels.unsqueeze(0) != labels.unsqueeze(1)]
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

class DDSLLoss(nn.Module): # 이건 서로 도와주는 loss일때 좋은데 우리 태스트랑은 안맞네
    def __init__(self, num_tasks=3):
        super().__init__()
        # 학습 가능한 파라미터 (초기값 0.0 -> variance 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_cont, loss_kd_genre, loss_kd_content):
        # 1. Contrastive Loss (Genre Vector)
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * loss_cont + 0.5 * self.log_vars[0]

        # 2. Genre KD Loss (Genre Vector -> Teacher)
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * loss_kd_genre + 0.5 * self.log_vars[1]
        
        # 3. Content KD Loss (Content Vector -> Teacher)
        precision3 = torch.exp(-self.log_vars[2])
        loss3 = precision3 * loss_kd_content + 0.5 * self.log_vars[2]

        return loss1 + loss2 + loss3