import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_Decoder(nn.Module):

    def __init__(self, hidden_dim, dropout, num_heads, label_size, device, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output = nn.Linear(hidden_dim, label_size)

    def forward(self, name, h_V_baseline, mask):
        B = h_V_baseline.size(0)
        query = self.query.repeat(B, 1, 1)
        tgt = query
        attn_weights_all = []
        for layer in self.layers:
            attn_output, attn_weights = layer.multihead_attn(query=tgt, key=h_V_baseline, value=h_V_baseline, key_padding_mask=mask == 0, need_weights=True, average_attn_weights=False)
            tgt2 = layer.dropout1(attn_output)
            tgt = layer.norm1(tgt + tgt2)
            tgt2 = layer.self_attn(tgt, tgt, tgt)[0]
            tgt = layer.norm2(tgt + layer.dropout2(tgt2))
            tgt2 = layer.linear2(layer.dropout(F.relu(layer.linear1(tgt))))
            tgt = layer.norm3(tgt + layer.dropout3(tgt2))
            attn_weights_all.append(attn_weights)
        logits = self.output(tgt.squeeze(1))
        last_attn = attn_weights_all[-1]
        avg_attn = last_attn.mean(dim=1).squeeze(1)
        topk_idx = avg_attn.topk(k=20, dim=1).indices
        return (logits, topk_idx)
