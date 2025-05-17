import torch
import torch.nn as nn

class CrossAttentionActionHead(nn.Module):
    def __init__(self, embed_dim=1536, hidden_dim=1024, action_dim=7, num_heads=4):
        super().__init__()
        self.act_query = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable [ACT] token

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vision_tokens, robot_state=None):
        B = vision_tokens.size(0)
        device = vision_tokens.device
        act_query = self.act_query.expand(B, -1, -1).to(device)  # ← 确保 query 和 key/value 同设备
        attn_out, _ = self.cross_attn(query=act_query, key=vision_tokens, value=vision_tokens)
        return self.mlp_head(attn_out.squeeze(1))
