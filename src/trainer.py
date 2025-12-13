from tqdm import tqdm
import torch
import torch.nn.functional as F
from .loss import SupervisedContrastiveLoss

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
        self.scheduler = scheduler
        self.device = device
        
        # Loss 함수
        self.scl = SupervisedContrastiveLoss(temperature=0.07)
        
        # 하이퍼파라미터 (Config에서 관리 추천)
        self.alpha = config.get('alpha', 0.8)  # Genre vs Content 비중
        self.beta = config.get('beta', 0.9)    # Moving Average 계수
        self.target_scale = config.get('target_scale', 0.6)
        
        # [핵심] 상태(State) 관리 변수
        self.running_ratio = 1.0  # 초기값

    # def alpha_scheduling(self, ):
        # global_step = epoch * steps_per_epoch + step

        # # --- 1. Alpha Scheduling (Decay 전략 적용) ---
        # # 충돌 시점(t) 이후부터 장르 비중(alpha)을 줄여서 충돌을 피합니다.
        # if global_step < t:
        #     alpha = base_alpha
        # else:
        #     # t 이후부터 Alpha Decay (장르 비중 감소)
        #     # global_step이 t보다 커질수록 x가 증가 -> sigmoid 증가 -> alpha 감소
        #     # x 계산: 변화 속도 조절 (숫자 5는 기울기, 필요시 조절)
        #     x = 5 * (global_step - t) / t
        #     sigmoid_x = 1 / (1 + math.exp(-x))

        #     # sigmoid_x는 0.5부터 시작해서 1.0으로 감
        #     # 이를 0.0 ~ 1.0 비율로 정규화하여 사용
        #     decay_ratio = (sigmoid_x - 0.5) * 2
        #     if decay_ratio > 1.0: decay_ratio = 1.0

        #     # base(0.8)에서 target(0.2)으로 서서히 이동
        #     alpha = base_alpha - (base_alpha - target_alpha) * decay_ratio
        #     # # (혹은 간단하게 step > t 일 때 바로 target_alpha로 고정해도 효과는 봅니다)
        #     # if alpha < target_alpha: alpha = target_alpha

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

        # for step, (batch_inputs, labels) in enumerate(tqdm(self.train_loader, desc = f"Epoch: {epoch+1}")):
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
            content_vectors = student_vectors[:, 1, :] # Vector 1: 내용

            # loss 계산
            loss_cont = self.scl(genre_vector, labels)
        
            genre_norm = F.normalize(genre_vector, p=2, dim=1)
            loss_kd_genre = F.mse_loss(genre_norm, teacher_norm)

            content_norm = F.normalize(content_vectors, p=2, dim=1)
            loss_kd_content = F.mse_loss(content_norm, teacher_norm)
            loss_kd_combined = self.alpha * loss_kd_genre + (1 - self.alpha) * loss_kd_content

            # gradient 계산
            norm_main = self.calc_grad_norm(loss_cont)
            norm_sub = self.calc_grad_norm(loss_kd_combined)

            current_ratio = norm_main / (norm_sub + 1e-8) * self.target_scale
            if current_ratio > 1000.0: current_ratio = 1000.0 # clipping

            self.running_ratio = self.beta * self.running_ratio + (1 - self.beta) * current_ratio

            total_loss = loss_cont + self.running_ratio * loss_kd_combined

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
            pos_positions = torch.nonzero(ranks[i], as_tuple=False).flatten()
            if len(pos_positions) == 0:
                reciprocal_rank.append(0.0)
            else:
                # 0-index이므로 +1 해줘야 순위가 됨
                reciprocal_rank.append(1.0 / (pos_positions[0].item() + 1))

        return sum(reciprocal_rank) / len(reciprocal_rank)