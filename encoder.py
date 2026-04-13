import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import TransformerConv, GATConv, GCNConv
from data import *

class GraphConvLayer(nn.Module):

    def __init__(self, hidden_dim, dropout=0.2, num_heads=4):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
        self.trans_attn = TransformerConv(hidden_dim, int(hidden_dim / num_heads), heads=num_heads, dropout=dropout, edge_dim=hidden_dim, root_weight=False)
        self.gat_attn = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.fusion_weights = nn.Parameter(torch.randn(3))
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.ReLU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.edge_updater = EdgeUpdateModule(hidden_dim, dropout)
        self.node_context = NodeContextModule(hidden_dim)
        self.dim_adapter = nn.Linear(1024, 256)

    def forward(self, node_feat, edge_links, edge_feat, batch_idx):
        trans_out = self.trans_attn(node_feat, edge_links, edge_feat)
        trans_res = self.layer_norms[0](node_feat + self.drop(trans_out))
        gat_out = self.gat_attn(node_feat, edge_links)
        gat_out = self.dim_adapter(gat_out)
        gat_res = self.layer_norms[1](node_feat + self.drop(gat_out))
        gcn_out = self.gcn(node_feat, edge_links)
        gcn_res = self.layer_norms[2](node_feat + self.drop(gcn_out))
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_feat = (
            weights[0] * gat_res
            + weights[1] * gcn_res
            + weights[2] * trans_res
        )
        ffn_out = self.ffn(fused_feat)
        node_feat = self.layer_norms[3](fused_feat + self.drop(ffn_out))
        edge_feat = self.edge_updater(node_feat, edge_links, edge_feat)
        node_feat = self.node_context(node_feat, batch_idx)
        return (node_feat, edge_feat)

class EdgeUpdateModule(nn.Module):

    def __init__(self, dim_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(dim_size)
        self.lin1 = nn.Linear(3 * dim_size, dim_size, bias=True)
        self.lin2 = nn.Linear(dim_size, dim_size, bias=True)
        self.activate = nn.GELU()

    def forward(self, node_feat, edge_links, edge_feat):
        src, dst = (edge_links[0], edge_links[1])
        combined = torch.cat([node_feat[src], edge_feat, node_feat[dst]], -1)
        message = self.lin2(self.activate(self.lin1(combined)))
        return self.norm(edge_feat + self.drop(message))

class NodeContextModule(nn.Module):

    def __init__(self, dim_size):
        super().__init__()
        self.modulator = nn.Sequential(nn.Linear(dim_size, dim_size), nn.ReLU(), nn.Linear(dim_size, dim_size), nn.Sigmoid())

    def forward(self, node_feat, batch_idx):
        batch_mean = scatter_mean(node_feat, batch_idx, dim=0)
        return node_feat * self.modulator(batch_mean[batch_idx])

class GraphFeatureEncoder(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=4, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim, bias=True), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim, bias=True), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.conv_stack = nn.ModuleList([GraphConvLayer(hidden_dim, dropout, 4) for _ in range(num_layers)])

    def forward(self, node_init, edge_links, edge_init, batch_idx):
        node_feat = self.node_proj(node_init)
        edge_feat = self.edge_proj(edge_init)
        for conv in self.conv_stack:
            node_feat, edge_feat = conv(node_feat, edge_links, edge_feat, batch_idx)
        return node_feat

class Graph_encoder(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.feature_encoder = GraphFeatureEncoder(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.input_processor = nn.Sequential(nn.LayerNorm(1024 + 9, eps=1e-06), nn.Linear(1024 + 9, hidden_dim), nn.LeakyReLU())
        self.feature_refiner = self._build_refinement(hidden_dim, dropout, layers=2)
        self.fusion_weight = 0.2
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _build_refinement(self, dim_size, dropout, layers):
        blocks = []
        for i in range(layers - 1):
            blocks += [nn.LayerNorm(dim_size, eps=1e-06), nn.Dropout(dropout), nn.Linear(dim_size, dim_size), nn.LeakyReLU()]
            if i == layers - 2:
                blocks.extend([nn.LayerNorm(dim_size, eps=1e-06)])
        return nn.Sequential(*blocks)

    def padding_ver1(self, x, batch_id, feature_dim):
        unique_ids, counts = torch.unique(batch_id, return_counts=True)
        batch_size = unique_ids.max().item() + 1
        max_len = counts.max().item()
        batch_data = torch.zeros((batch_size, max_len, feature_dim), dtype=x.dtype, device=x.device)
        mask = torch.zeros((batch_size, max_len), dtype=x.dtype, device=x.device)
        split_x = torch.split(x, counts.tolist())
        for i, nodes in zip(unique_ids.tolist(), split_x):
            len_i = nodes.size(0)
            batch_data[i, :len_i, :] = nodes
            mask[i, :len_i] = 1
        return (batch_data, mask)

    def forward(self, coords, node_init, edge_links, batch_idx, base_data):
        base_feat = self.input_processor(base_data.to(self.device))
        base_feat = self.feature_refiner(base_feat)
        geo_feat, edge_init = get_geo_feat(coords, edge_links)
        encoded = self.feature_encoder(node_init.to(self.device), edge_links, edge_init, batch_idx)
        struct_feat, struct_mask = self.padding_ver1(encoded.cpu(), batch_idx.cpu(), encoded.shape[1])
        fused_feat = self.fusion_weight * base_feat + (1 - self.fusion_weight) * struct_feat.to(self.device)
        return (fused_feat, struct_mask.to(self.device))
