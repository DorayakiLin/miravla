import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.cross_attn_action import CrossAttentionActionHead
import torch

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
embedder = InternVL3Embedder(device=device)
action_head = CrossAttentionActionHead().to(device)

fused_tokens = embedder.get_fused_image_text_embedding(
    "model/internvl3/example.jpg", "请描述图中物体", return_cls_only=False
).to(dtype=torch.float32)



# 图像 → hidden_tokens
# hidden_tokens = embedder.get_image_embedding("model/internvl3/example.jpg").to(device)  # [B, N, D]
# hidden_tokens = hidden_tokens.to(dtype=torch.float32)

print("fused_tokens shape:", fused_tokens.shape)

# Token → 动作
predicted_action = action_head(fused_tokens)
print("Predicted action:", predicted_action.shape)
