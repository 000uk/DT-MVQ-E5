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
        # self.alpha = config.get('alpha', 0.8)  # Genre vs Content 비중
        # self.beta = config.get('beta', 0.9)    # Moving Average 계수
        # self.target_scale = config.get('target_scale', 0.6)
        self.t = 200  # 충돌 시작 지점
        self.base_alpha = 0.8   # 초반: 장르(Genre) 정보를 확실히 잡음
        self.target_alpha = 0.2 # 후반: 충돌 회피를 위해 장르 비중을 낮춤 (본문 집중)
        self.steps_per_epoch = len(train_loader)
        self.beta = 0.95  # 관성 계수 (클수록 변화가 부드러움)
        self.mrr_ratio = 0.5
        self.running_ratio = 1.0
        self.target_scale = 0.6

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
    
        for step, (batch_inputs, labels) in enumerate(tqdm(self.train_loader, desc = f"Epoch: {epoch+1}")):
            global_step = epoch * self.steps_per_epoch + step
    
            if global_step < self.t:
                alpha = self.base_alpha
            else:
                x = 5 * (global_step - self.t) / self.t
                sigmoid_x = 1 / (1 + math.exp(-x))
    
                decay_ratio = (sigmoid_x - 0.5) * 2
                if decay_ratio > 1.0: decay_ratio = 1.0
    
                alpha = self.base_alpha - (self.base_alpha - self.target_alpha) * decay_ratio
    
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            labels = labels.to(self.device)
    
            student_vectors = self.model(**batch_inputs)
            
            genre_vector = student_vectors[:, 0, :]
            content_vector = student_vectors[:, 1, :]
            
            loss_cont = self.scl(genre_vector, labels)
        
            with torch.no_grad():
                teacher_outputs = self.teacher(**batch_inputs)
                hidden = teacher_outputs.last_hidden_state       # (B, L, D)
                mask = batch_inputs['attention_mask'].unsqueeze(-1)  # (B, L, 1)
                teacher_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
    
            genre_norm = F.normalize(genre_vector, p=2, dim=1)
            loss_kd_genre = F.mse_loss(genre_norm, teacher_norm)
            
            content_norm = F.normalize(content_vector, p=2, dim=1)
            loss_kd_content = F.mse_loss(content_norm, teacher_norm)
            
            loss_kd_combined = alpha * loss_kd_genre + (1 - alpha) * loss_kd_content
    
            norm_main = self.calc_grad_norm(loss_cont)
            norm_sub = self.calc_grad_norm(loss_kd_combined)
            
            current_ratio = norm_main / (norm_sub + 1e-8) * self.target_scale
            if current_ratio > 1000.0: current_ratio = 1000.0
            self.running_ratio = self.beta * self.running_ratio + (1 - self.beta) * current_ratio
    
            total_loss = loss_cont + self.running_ratio * loss_kd_combined
    
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