import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.e5_lora import E5LoRABackbone
from .heads.multi_vector import SimpleMultiVectorHead, MultiVectorHead

class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, lora_config: dict):
        super().__init__()
        self.backbone = E5LoRABackbone(model_name, lora_config)
        # self.head = MultiVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size)
        self.head = SimpleMultiVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size)
    
    def forward(self, input_ids, attention_mask, **kargs):
        sequence_output = self.backbone(input_ids, attention_mask) # (B, L, D)
        embeddings = self.head(sequence_output, attention_mask) # (B, k, D)
        return F.normalize(embeddings, p=2, dim=2) # contrastive loss 계산하려면 필수

class AdvancedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained('intfloat/e5-small')
        
        # 🔥 [간지 포인트] 추가적인 Transformer Block (1~2층)
        # E5의 출력을 한 번 더 정제해서 "내 데이터셋 맞춤형"으로 만듦
        encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8, batch_first=True)
        self.context_block = nn.TransformerEncoder(encoder_layer, num_layers=2) # 딱 2층만!

        # 우리가 만든 멋진 Head
        self.head = CompetitiveVectorHead(num_vectors=2)

    def forward(self, input_ids, attention_mask):
        # 1. E5 (Giant)
        outputs = self.backbone(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state # (B, L, 384)

        # 2. Context Block (Adapter)
        # 여기서 토큰끼리 한 번 더 섞이면서 "책 추천 특화" 문맥을 만듦
        sequence_output = self.context_block(sequence_output, src_key_padding_mask=~attention_mask.bool())

        # 3. Head (Specialist)
        vectors = self.head(sequence_output)
        return vectors