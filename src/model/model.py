import torch
import torch.nn as nn
from .backbones.e5_lora import E5LoRABackbone
from .heads.multi_vector import MultiVectorHead

class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, lora_config: dict):
        self.backbone = E5LoRABackbone(model_name, lora_config)
        self.head = MultiVectorHead(num_vectors=2, input_dim=self.backbone.config.hidden_size)
    
    def forward(self, input_ids, attention_mask, **kargs):
        sequence_output = self.backbone(input_ids, attention_mask) # (B, L, D)
        embeddings = self.head(sequence_output, attention_mask) # (B, k, D)
        return embeddings