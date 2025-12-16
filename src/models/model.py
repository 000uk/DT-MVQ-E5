import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.e5_lora import E5LoRABackbone
from .heads.multi_vector import SimpleMultiVectorHead, MultiVectorHead
from .heads.competitive import CompetitiveVectorHead

# class BookEmbeddingModel(nn.Module):
#     def __init__(self, model_name: str, lora_config: dict):
#         super().__init__()
#         self.backbone = E5LoRABackbone(model_name, lora_config)
        
#         # encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8, batch_first=True)
#         # self.context_block = nn.TransformerEncoder(encoder_layer, num_layers=2) # 딱 2층만!
        
#         # 1. self.head = SimpleMultiVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size)
#         # 2. self.head = MultiVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size)
#         self.head = CompetitiveVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size, num_heads=8)
    
#     def forward(self, input_ids, attention_mask, **kargs):
#         sequence_output = self.backbone(input_ids, attention_mask) # (B, L, D)
#         # sequence_output = self.context_block(sequence_output, src_key_padding_mask=~attention_mask.bool())
#         embeddings = self.head(sequence_output, attention_mask) # (B, k, D)
#         return F.normalize(embeddings, p=2, dim=2) # contrastive loss 계산하려면 필수

class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, lora_config: dict):
        super().__init__()
        self.backbone = E5LoRABackbone(model_name, lora_config)
        hidden_size = self.backbone.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=4, batch_first=True)
        self.context_block = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.genre_head = MultiVectorHead(num_vectors=1, input_dim=hidden_size)
        self.content_head = MultiVectorHead(num_vectors=1, input_dim=hidden_size)
    
    def forward(self, input_ids, attention_mask, **kargs):
        # 1. 백본 통과
        sequence_output = self.backbone(input_ids, attention_mask) # (B, L, D)
        # sequence_output = self.context_block(sequence_output, src_key_padding_mask=~attention_mask.bool())
        context_out = self.context_block(sequence_output, src_key_padding_mask=~attention_mask.bool())
        sequence_output = sequence_output + 0.1 * context_out

        # 2. 각자 정보 추출 (Decoupled)
        # 각각 (B, 1, D) 모양의 텐서가 나옴
        genre_vec = self.genre_head(sequence_output, attention_mask)
        content_vec = self.content_head(sequence_output, attention_mask)
        
        # 3. 다시 하나로 합치기 (Concatenate)
        # (B, 1, D) 두 개를 이어붙여서 -> (B, 2, D)로 만듦
        # 이렇게 하면 기존 코드의 리턴값 모양과 완전히 동일해짐!
        embeddings = torch.cat([genre_vec, content_vec], dim=1) 
        
        # 4. 정규화 (Contrastive Loss용)
        return F.normalize(embeddings, p=2, dim=2)