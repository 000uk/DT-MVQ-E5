import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# 사용자 정의 모듈 임포트 (경로에 맞게 수정)
from src.utils import set_seed
from src.models.model import BookEmbeddingModel
from src.dataset import get_loader
# from src.trainer_so_complecated import DualDistillationTrainer
from src.ver4_trainer import DualDistillationTrainer

def fix_grad_ratio(train_loader, device, optimizer, model, trainer, teacher_model):
    batch = next(iter(train_loader)) # 데이터 배치 하나만 가져옴
    inputs = {k: v.to(device) for k, v in batch[0].items()}
    labels = batch[1].to(device)
    
    optimizer.zero_grad()
    outputs = model(**inputs)
    genre_vector = outputs[:, 0, :]
    content_vector = outputs[:, 1, :]
    
    loss_scl = trainer.scl(genre_vector, labels)
    norm_scl = trainer.calc_grad_norm(loss_scl) 

    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        hidden = teacher_outputs.last_hidden_state # B, L, D
        mask = inputs['attention_mask'].unsqueeze(-1) # B, L, 1
        teacher_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)  # (B, D)
    content_norm = F.normalize(content_vector, p=2, dim=1)
    loss_kd = F.mse_loss(content_norm, teacher_norm)
    norm_kd = trainer.calc_grad_norm(loss_kd)
    
    recommended_ratio = norm_kd / (norm_scl + 1e-8)
    
    print(f"🔥 SCL Power (Grad Norm): {norm_scl:.4f}")
    print(f"💧 KD Power (Grad Norm): {norm_kd:.4f}")
    print(f"⚖️ 수학적 추천 비율 (SCL Weight): {recommended_ratio:.6f}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    config = load_config(args.config)
    exp_name = config['exp_name']

    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_backup.yaml"), "w") as f:
        yaml.dump(config, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config['seed'])
    print(f"🚀 Start Experiment: {exp_name}")
    print(f"📂 Results will be saved at: {save_dir}")
    
    print("🤖 Initializing Models...")
    teacher_model = AutoModel.from_pretrained(config['model']['backbone'])
    teacher_model.eval()
    teacher_model.to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    model = BookEmbeddingModel(
        model_name=config['model']['backbone'],
        lora_config=config['model']['lora']
    ).to(device)

    print("📚 Loading Data...")
    data_path = config['data_path']
    batch_size = config['train']['batch_size']
    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
    train_loader, valid_loader = get_loader(data_path, batch_size, tokenizer)

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))
    total_steps = len(train_loader) * config['train']['epochs']

    # linear.. /// CosineAnnealingWarmRestarts 이런것도 있대
    scheduler = get_cosine_schedule_with_warmup( 
        optimizer,
        num_warmup_steps=int(total_steps * config['train']['warmup_ratio']),
        num_training_steps=total_steps
    )

    trainer = DualDistillationTrainer(
        model=model,
        teacher=teacher_model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    fix_grad_ratio(train_loader, device, optimizer, model, trainer, teacher_model)
        
    best_mrr = 0.0
    history = [] # 로그 저장용 리스트
    for epoch in range(config['train']['epochs']):
        train_loss = trainer.train_epoch(epoch)
        val_loss, val_mrr = trainer.validation(valid_loader, k=10, mrr_ratio=config['valid']['mrr_ratio'])
        print(f"📊 [Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MRR: {val_mrr:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "val_mrr": val_mrr,
        })

        if val_mrr > best_mrr:
            print(f"✅ Best Model Updated! ({best_mrr:.4f} -> {val_mrr:.4f})")
            best_mrr = val_mrr
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        pd.DataFrame(history).to_csv(os.path.join(save_dir, "logs.csv"), index=False)
        
    print("✨ Experiment Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 실행할 때 --config 옵션으로 yaml 파일 경로를 받음
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    main(args)