from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn

"""
Transformer 내부의 특정 Layer(Wq, Wk, Wv 등)에 "LoRA 모듈"을 주입!
E5 Model (pretrained)
    ├── LayerNorm
    ├── 24× Transformer Layers
    │       ├── Attention (Q,K,V,O)
    │       ├── FFN Layers
    │       └── ... (원래 weight 고정)
    └── Pooler → embedding vector   # e.g., CLS embedding

LoRA Injected:
    ├── W_q = W_q + A_q B_q
    ├── W_k = W_k + A_k B_k
    ├── W_v = W_v + A_v B_v
    ├── FFN = FFN + A_ffn B_ffn
    └── (small learnable matrices only)
"""
class E5LoRABackbone(nn.Module):
    def __init__(self, model_name: str, lora_cfg: dict):
        super().__init__()
        
        base_model = AutoModel.from_pretrained(model_name)

        # Linear(d → d) -→ Linear(d → d) + LoRA(d → d)
        lora_config = LoraConfig( 
            task_type=TaskType.FEATURE_EXTRACTION, # 임베딩 fine-tuning
            # LoRA가 분류기와 같은 output head에 적용되는 것이 아니라
            # 모델의 Transformer 블록(encoder)에만 적용되도록
            r=lora_cfg["r"],    # LoRA rank
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            bias="none"
        )

        self.encoder = get_peft_model(base_model, lora_config)
        
        self.config = self.encoder.config # hidden_size 같은거 head에서 알아야함
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # return outputs.last_hidden_state[:, 0] single vector 테스트할땐 일케 했는디..
        return outputs.last_hidden_state # head에 넣을거라 CLS pooling 안함