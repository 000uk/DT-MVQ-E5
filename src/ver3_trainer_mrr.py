from tqdm import tqdm
import torch
import torch.nn.functional as F
import math
from .loss import SupervisedContrastiveLoss
from .utils import calculate_mrr

class DualDistillationTrainer:
    """
    Dual-Target Strong Knowledge Distillation을 수행하는 Trainer
    GradNorm을 통한 Gradient Balancing과 Dynamic Curriculum Learning을 포함
    """
    def __init__(self, model, teacher, train_loader, optimizer, scheduler, device, config):
        self.model = model
        self.teacher = teacher
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scl = SupervisedContrastiveLoss(temperature=0.05, neg_ratio=0.2)
        
        # 하이퍼파라미터 (Config에서 관리 추천)
        self.alpha = config.get('alpha', 0.8)  # Genre vs Content 비중
        self.beta = config.get('beta', 0.9)    # Moving Average 계수
        self.target_scale = config.get('target_scale', 0.6)
        
        # [핵심] 상태(State) 관리 변수
        self.running_ratio = 1.0  # 초기값

    def calc_grad_norm(self, loss):
        """
        장르, 내용 벡터가 head에서 부터 나눠지니까,
        head에서 파라미터 가져와서 Gradiend 계산함!
        """
        # requires_grad=True인 파라미터만 추적
        params = [p for p in self.model.head.attention.parameters() if p.requires_grad]
        
        # retain_graph=True: 뒤에 진짜 backward()를 또 해야 하므로 그래프를 날리면 안 됨
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

        if not grads or grads[0] is None: 
            return 0.0

        total_norm = 0.0
        for g in grads:
            if g is not None:
                total_norm += g.data.norm(2).item() ** 2

        return total_norm ** 0.5

    def train_epoch(self, epoch):        
        self.model.train()
        train_loss = 0

        for batch_inputs, labels in tqdm(self.train_loader, desc = f"Epoch: {epoch+1}"):    
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            labels = labels.to(self.device)

            student_vectors = self.model(**batch_inputs) # 얜 head에서 attn_mask 따로 써서 mean_pooling 안해줘도 됨

            genre_vector = student_vectors[:, 0, :] # Vector 0: 장르
            content_vector = student_vectors[:, 1, :] # Vector 1: 내용

            # 1. scl loss
            # genre_vector는 이미 E5MultiVectorHead에서 정규화 했으니 ㄱㅊ
            # student_vector 자체가 이미 길이가 1인 벡터임
            loss_scl = self.scl(genre_vector, labels)

            # 2. kd loss
            with torch.no_grad():
                teacher_outputs = self.teacher(**batch_inputs)
                # teacher_embeddings = teacher_outputs.last_hidden_state.mean(dim=1)
                hidden = teacher_outputs.last_hidden_state # B, L, D
                mask = batch_inputs['attention_mask'].unsqueeze(-1) # B, L, 1
                teacher_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)  # (B, D)
                
            content_norm = F.normalize(content_vector, p=2, dim=1)
            loss_kd = F.mse_loss(content_norm, teacher_norm)

            # 3. 직교 loss
            sim_orth = F.cosine_similarity(genre_vector, content_vector, dim=1)
            loss_orth = torch.abs(sim_orth).mean()

            # w_cont = 0.5 # SCL은 보통 값이 2.0~5.0으로 큽니다. 너무 세면 KD를 무시하니 절반으로 줄입니다.
            # # KD(MSE)는 값이 0.0x~0.1 수준으로 작습니다. 중요하므로 가중치를 둬야 하지만, AdamW가 스케일은 어느 정도 잡아줍니다.
            # w_kd = 1.0 # 1.0으로 두되, SCL 힘을 뺐으니 상대적으로 강조됩니다.
            # w_orth = 0.1 # Orthogonal은 보조 제약조건입니다. 너무 세면 학습을 방해하므로 '넛지(Nudge)' 정도만 줍니다.
            total_loss = (0.01 * loss_scl) + (1.0  * loss_kd) + (0.05 * loss_orth)

            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += total_loss.item()

        return train_loss / len(self.train_loader)
    
    def validation(self, valid_loader, k=10, mrr_ratio=1.0):
        self.model.eval()
        embeddings_list_v0 = []
        embeddings_list_v1 = []
        labels_list = []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_inputs, labels in valid_loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                labels = labels.to(self.device)

                student_vectors = self.model(**batch_inputs)
                genre_vector = student_vectors[:, 0, :]
                content_vector = student_vectors[:, 1, :]
                
                # ============================
                loss_scl = self.scl(genre_vector, labels)
                with torch.no_grad():
                    teacher_outputs = self.teacher(**batch_inputs)
                    hidden = teacher_outputs.last_hidden_state # B, L, D
                    mask = batch_inputs['attention_mask'].unsqueeze(-1) # B, L, 1
                    teacher_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)  # (B, D)
                content_norm = F.normalize(content_vector, p=2, dim=1)
                loss_kd = F.mse_loss(content_norm, teacher_norm)
                sim_orth = F.cosine_similarity(genre_vector, content_vector, dim=1)
                loss_orth = torch.abs(sim_orth).mean()
                # total_loss = (0.5 * loss_scl) + (1.0  * loss_kd) + (0.1 * loss_orth)
                total_loss = (0.01 * loss_scl) + (1.0  * loss_kd) + (0.05 * loss_orth)
                total_val_loss += total_loss.item()
                # ============================
                    
                # MRR 계산을 위한 데이터 수집은 CPU로
                embeddings_list_v0.append(genre_vector.cpu())
                embeddings_list_v1.append(content_vector.cpu())
                labels_list.append(labels.cpu()) # CPU로 이동
                
        # 데이터 합치기
        all_embeddings_v0 = torch.cat(embeddings_list_v0, dim=0)
        all_embeddings_v1 = torch.cat(embeddings_list_v1, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
    
        # 유사도 계산 (기존 코드와 동일)
        similarity_v0 = torch.matmul(all_embeddings_v0, all_embeddings_v0.T)
        similarity_v1 = torch.matmul(all_embeddings_v1, all_embeddings_v1.T)
        final_similarity = (mrr_ratio * similarity_v0) + ((1 - mrr_ratio) * similarity_v1)
        
        # MRR 계산 (유틸리티 함수 호출)
        avg_mrr = calculate_mrr(final_similarity, all_labels, k=k)
        
        # Validation Loss 계산
        avg_val_loss = total_val_loss / len(valid_loader)
    
        # 두 지표를 모두 반환
        return avg_val_loss, avg_mrr