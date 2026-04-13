import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn


def check_duplicate_points(sub_X):
    duplicate_points = []
    for i in range(sub_X.shape[0]):
        for j in range(i + 1, sub_X.shape[0]):
            if torch.all(sub_X[i] == sub_X[j]):
                duplicate_points.append((i, j))
    return duplicate_points


class ProteinGraphDataset(data.Dataset):

    def __init__(self, dataset, index, radius=10, device='gpu'):
        super(ProteinGraphDataset, self).__init__()
        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())
        self.label_size = 5106
        self.radius = radius
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs[idx]
        with torch.no_grad():
            X = torch.load(f'./data/Structures/{name}.tensor', weights_only=False)
            seq = torch.tensor([self.letter_to_num.get(aa, -1) for aa in self.dataset[name][0][0]], dtype=torch.long)
            prottrans_feat = torch.load(f'./data/ProtTrans/{name}.tensor', weights_only=False)
            dssp_feat = torch.load(f'./data/DSSP/{name}.tensor', weights_only=False)
            prottrans_feat = torch.tensor(prottrans_feat, dtype=torch.float32)
            dssp_feat = torch.tensor(dssp_feat, dtype=torch.float32)
            pre_computed_node_feat = torch.cat([prottrans_feat, dssp_feat], dim=-1)
            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=200, num_workers=16)
            ec_number = self.dataset[name][0][1]
            graph_data = torch_geometric.data.Data(name=name, seq=seq, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index, y=ec_number)
        return graph_data


def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)
    edge_sbf = _get_edge_sbf(X[:, 1], edge_index)
    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction, edge_sbf], dim=-1)
    return (geo_node_feat, geo_edge_feat)


def _get_edge_sbf(X, edge_index, D_min=0.0, D_max=20.0, D_count=16, num_theta=8):
    src, dst = edge_index
    edge_vectors = X[src] - X[dst]
    distances = torch.norm(edge_vectors, dim=-1)
    rbf = _rbf(distances, D_min, D_max, D_count)
    theta = torch.acos(edge_vectors[:, 2] / (distances + 1e-06))
    phi = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
    angle_feat = torch.cat([torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1), torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1)], dim=-1)
    return torch.cat([rbf, angle_feat], dim=-1)


def _calculate_curvature(X: torch.Tensor, k: int=10) -> torch.Tensor:
    Ca = X[:, 1]
    dist = torch.cdist(Ca, Ca)
    _, indices = torch.topk(dist, k + 1, dim=1, largest=False)
    neighbor_indices = indices[:, 1:]
    neighbors = Ca[neighbor_indices]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    cov = torch.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    curvature = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-06)
    return curvature.unsqueeze(-1)


def _compute_surface_normals(X: torch.Tensor, k: int=10) -> torch.Tensor:
    Ca = X[:, 1]
    dist = torch.cdist(Ca, Ca)
    _, indices = torch.topk(dist, k + 1, dim=1, largest=False)
    neighbor_indices = indices[:, 1:]
    neighbors = Ca[neighbor_indices]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    cov = torch.einsum('nki,nkj->nij', centered, centered)
    _, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    return normals


def _edge_attention_weights(X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    vec = X[src, 1] - X[dst, 1]
    distance = torch.norm(vec, dim=1, keepdim=True)
    direction = vec / (distance + 1e-06)
    src_feat = X[src].flatten(1)
    dst_feat = X[dst].flatten(1)
    cosine = F.cosine_similarity(src_feat, dst_feat, dim=1)
    return torch.cat([distance, cosine.unsqueeze(-1)], dim=1)


def _positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE


def _get_angle(X, eps=1e-07):
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)
    cosD = (u_2 * u_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)
    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _get_distance(X, edge_index):
    atom_N = X[:, 0]
    atom_Ca = X[:, 1]
    atom_C = X[:, 2]
    atom_O = X[:, 3]
    atom_R = X[:, 4]
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', 'R-C', 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1)
    atom_list = ['N', 'Ca', 'C', 'O', 'R']
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)
    return (node_dist, edge_dist)


def _get_direction_orientation(X, edge_index):
    X_N = X[:, 0]
    X_Ca = X[:, 1]
    X_C = X[:, 2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)
    node_j, node_i = edge_index
    t = F.normalize(X[:, [0, 2, 3, 4]] - X_Ca.unsqueeze(1), dim=-1)
    try:
        node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1)
    except Exception as ex:
        print(t.size())
        print(local_frame.size())
        print('except')
        print(ex)
    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1)
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1)
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1)
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1)
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim=-1)
    r = torch.matmul(local_frame[node_i].transpose(-1, -2), local_frame[node_j])
    edge_orientation = _quaternions(r)
    return (node_direction, edge_direction, edge_orientation)


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)))
    _R = lambda i, j: R[:, i, j]
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q
