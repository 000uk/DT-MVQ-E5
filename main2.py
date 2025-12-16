import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

# 사용자 정의 모듈 임포트 (경로에 맞게 수정)
from src.utils import set_seed, calc_grad_norm
from src.models.model import BookEmbeddingModel
from src.dataset import get_loader
# from src.trainer_so_complecated import DualDistillationTrainer
from src.ver1_trainer_so_complecated import DualDistillationTrainer

def fix_grad_ratio(train_loader, device, optimizer, model, trainer, teacher_model):
    batch = next(iter(train_loader)) # 데이터 배치 하나만 가져옴
    inputs = {k: v.to(device) for k, v in batch[0].items()}
    labels = batch[1].to(device)
    
    optimizer.zero_grad()
    outputs = model(**inputs)
    genre_vector = outputs[:, 0, :]
    content_vector = outputs[:, 1, :]

    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        hidden = teacher_outputs.last_hidden_state # B, L, D
        mask = inputs['attention_mask'].unsqueeze(-1) # B, L, 1
        teacher_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)  # (B, D)
    content_norm = F.normalize(content_vector, p=2, dim=1)
    loss_kd = F.mse_loss(content_norm, teacher_norm)
    norm_kd = calc_grad_norm(loss_kd, model)
    loss_scl = trainer.scl(genre_vector, labels)
    norm_scl = calc_grad_norm(loss_scl, model) 
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
    
    
    book_path = 'data/book_meta.parquet'
    books = pd.read_parquet(book_path)
    
    def build_text(row): # 입력 텍스트 생성 (타이틀 + 설명 + 저자 등 결합)
        parts = [
            f"Title: {row['title']} |",
            # f"Category: {row['category']} |", # oracle
            f"Description: {row['description']}"
        ]
        return " ".join( # 리스트의 문자열들을 공백으로 연결할건데.....
            [p for p in parts if isinstance(p, str)] # NaN이나 None이 있으면 제외함
        ) # 최종적으로 하나의 문장 형태로 반환한다고 함!! "Title: ... Category: ... Description: ..."

    books["text"] = books.apply(build_text, axis=1) # 새 컬럼 text에 대해서.... 문장 만듦
    
    # 100개 미만인 카테고리는 노이즈로 간주하고 삭제
    counts = books['category'].value_counts()
    valid_categories = counts[counts > 100].index
    books = books[books['category'].isin(valid_categories)]
    
    dataset = Dataset.from_pandas(books)
    
    le = LabelEncoder()
    le.fit(dataset['category'])   # 전체 데이터로 학습
    
    def encode_label(x):
        return {"label": le.transform([x["category"]])[0]}
    
    dataset = dataset.map(encode_label)
    
    num_classes = len(le.classes_)
    
    # Transformer 모델은 이런 raw 텍스트를 바로 처리 못 하고
    # 토크나이저를 거쳐 tensor(batch_input_ids, batch_attention_mask) 형태가 필요함.
    def collate_fn(batch): # DataLoader가 batch마다 호출
        # texts = [f"passage: {x['text']}" for x in batch]
        texts = [f"query: {x['text']}" for x in batch]
        labels = torch.tensor([x['label'] for x in batch])  # 라벨을 int 리스트 → torch.tensor 로 변환
    
        """
        토크나이저:
        텍스트를 token id로 변환 (input_ids), attention_mask 생성,
        batch의 최대 length에 맞춰 패딩, 출력 타입은 PyTorch tensor
    
        { 'input_ids': tensor([[101,  ... , 102], ...]),
          'attention_mask': tensor([[1,1,1,0,0...], ...) }
        """
        inputs = tokenizer(
          texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
        return inputs, labels
    
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    valid_len = total_len - train_len
    
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])
    
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    print("📚 Loading Trainer...")
    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))
    total_steps = len(train_loader) * config['train']['epochs']
    scheduler = get_cosine_schedule_with_warmup( # linear.. /// CosineAnnealingWarmRestarts 이런것도 있대
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

    # fix_grad_ratio(train_loader, device, optimizer, model, trainer, teacher_model)
        
    best_mrr = 0.0
    history = [] # 로그 저장용 리스트
    for epoch in range(config['train']['epochs']):
        train_loss = trainer.train_epoch(epoch)
        # val_loss, val_mrr = trainer.validation(valid_loader, k=10, mrr_ratio=config['valid']['mrr_ratio'])
        val_mrr = trainer.validation(valid_loader, k=10)
        print(f"📊 [Epoch {epoch+1}] Train Loss: {train_loss:.4f} | MRR: {val_mrr:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            # "valid_loss": val_loss,
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