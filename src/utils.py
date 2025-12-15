import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)            # 기본 Python random 고정
    np.random.seed(seed)         # NumPy 랜덤 고정
    torch.manual_seed(seed)      # CPU 연산 랜덤 고정
    torch.cuda.manual_seed(seed) # GPU 모든 디바이스 랜덤 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 때

    # 연산 재현성
    torch.backends.cudnn.deterministic = True  # cuDNN 연산을 determinisitc으로 강제
    torch.backends.cudnn.benchmark = False     # CUDA 성능 자동 튜닝 기능 끔 → 완전 재현 가능

def calculate_mrr(final_similarity, all_labels, k=10):
    # 1. 자기 자신 제외 (유사도 행렬이 이미 만들어진 상태에서 시작)
    final_similarity.fill_diagonal_(-1e9) 

    # 2. Top-k 순위 및 라벨 획득 (기존 코드와 동일)
    _, topk_idx = final_similarity.topk(k, dim=1)
    nn_labels_topk = all_labels[topk_idx]

    # 3. 정답 순위 계산
    ranks = (nn_labels_topk == all_labels.unsqueeze(1)).float()
    reciprocal_rank = []

    for i in range(ranks.size(0)):
        pos_positions = torch.nonzero(ranks[i]).flatten()
        if len(pos_positions) == 0:
            reciprocal_rank.append(0.0)
        else:
            reciprocal_rank.append(1.0 / (pos_positions[0].item() + 1))

    return sum(reciprocal_rank) / len(reciprocal_rank)