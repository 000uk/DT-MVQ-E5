from tqdm import tqdm
import torch
import torch.nn.functional as F
import math
from .loss import SupervisedContrastiveLoss

class DualDistillationTrainer:
    """
    Dual-Target Strong Knowledge Distillation을 수행하는 Trainer
    GradNorm을 통한 Gradient Balancing과 Dynamic Curriculum Learning을 포함
    """
    def __init__(self, model, teacher, train_loader, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.teacher = teacher
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Loss 함수
        self.scl = SupervisedContrastiveLoss(temperature=0.05)

    def train_epoch(self, epoch):        
        self.model.train()
        train_loss = 0

        for batch_inputs, labels in tqdm(self.train_loader, desc = f"Epoch: {epoch+1}"):
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            labels = labels.to(self.device)

            with torch.no_grad():
                teacher_outputs = self.teacher(**batch_inputs)
                # teacher_embeddings = teacher_outputs.last_hidden_state.mean(dim=1)
                hidden = teacher_outputs.last_hidden_state # B, L, D
                mask = batch_inputs['attention_mask'].unsqueeze(-1) # B, L, 1
                teacher_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)  # (B, D)

            student_vectors = self.model(**batch_inputs) # 얜 head에서 attn_mask 따로 써서 mean_pooling 안해줘도 됨

            genre_vector = student_vectors[:, 0, :] # Vector 0: 장르
            content_vectors = student_vectors[:, 1:, :] # Vector 1: 내용

            # loss 계산
            # genre_vector는 이미 E5MultiVectorHead에서 정규화 했으니 ㄱㅊ
            # student_vector 자체가 이미 길이가 1인 벡터임
            loss_cont = self.scl(genre_vector, labels)

            # 정규화(Normalize) 후 MSE 계산? 왜냐만 길이가 1인 벡터를 평균내서 l2-norm 구하면 1보다 작아지잖아
            # (1,0), (0,1) -> (0.5, 0), (0, 0.5) -> l2-norm(길이) = 0.707
            genre_norm = F.normalize(genre_vector, p=2, dim=1)
            loss_kd_genre = F.mse_loss(genre_norm, teacher_norm)

            content_mean = content_vectors.mean(dim=1)
            content_norm = F.normalize(content_mean, p=2, dim=1)
            loss_kd_content = F.mse_loss(content_norm, teacher_norm)
            
            total_loss = self.criterion(loss_cont, loss_kd_genre, loss_kd_content)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += total_loss.item()

        return train_loss / len(self.train_loader)

    def validation(self, valid_loader, k=10):
        # Vector 0 (Genre Vector)만 사용하여 카테고리 검색 성능을 측정
        self.model.eval()
        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for batch_inputs, labels in valid_loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

                genre_vector = self.model(**batch_inputs)[:, 0, :]
                genre_embeddings = F.normalize(genre_vector, p=2, dim=1)

                embeddings_list.append(genre_embeddings.cpu()) # cpu로 이동
                labels_list.append(labels)

        # 전체 데이터 합치기
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        similarity = torch.matmul(all_embeddings, all_embeddings.T) # 유사도 계산
        similarity.fill_diagonal_(-1e9) # 자기 자신 제외 (-무한대 마스킹)

        _, topk_idx = similarity.topk(k, dim=1)  # top-k neighbor 인덱스
        nn_labels_topk = all_labels[topk_idx] # 이웃들의 라벨 가져오기 (N, k)

        ranks = (nn_labels_topk == all_labels.unsqueeze(1)).float() # 정답 label 위치 찾기
        reciprocal_rank = []

        for i in range(ranks.size(0)):
            # torch.nonzero는 인덱스를 반환
            pos_positions = torch.nonzero(ranks[i]).flatten()
            if len(pos_positions) == 0:
                reciprocal_rank.append(0.0)
            else:
                # 0-index이므로 +1 해줘야 순위가 됨
                reciprocal_rank.append(1.0 / (pos_positions[0].item() + 1))

        return sum(reciprocal_rank) / len(reciprocal_rank)