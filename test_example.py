import os
import torch
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from data import ProteinGraphDataset
from model import GraphEC_model
from encoder import Graph_encoder
from decoder import Graph_Decoder

NN_config = {
    'feature_dim': 1024 + 9,
    'edge_input_dim': 450 + 20,
    'hidden_dim': 256,
    'layer': 3,
    'num_heads': 8,
    'dropout': 0.1,
    'label_size': 5089,
}


def padding_ver1(x, batch_id, feature_dim):
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
    return batch_data, mask


def convert_to_label_vector(EC_numbers, EC_id, num_classes):
    label_vector = np.zeros(num_classes)
    for ec in EC_numbers.split(';'):
        ec_idx = EC_id.get(ec, -1)
        if ec_idx != -1:
            label_vector[ec_idx] = 1
    return label_vector


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder_model = Graph_encoder(
        node_dim=NN_config['feature_dim'],
        edge_dim=NN_config['edge_input_dim'],
        hidden_dim=NN_config['hidden_dim'],
        num_layers=NN_config['layer'],
        dropout=NN_config['dropout'],
        device=device,
    ).to(device)

    decoder_model = Graph_Decoder(
        hidden_dim=NN_config['hidden_dim'],
        dropout=NN_config['dropout'],
        num_heads=NN_config['num_heads'],
        label_size=NN_config['label_size'],
        device=device,
    ).to(device)

    model = GraphEC_model(encoder_model, decoder_model).to(device)

    checkpoint = torch.load('./models/pretrain_model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    with open('./data/example.pkl', 'rb') as f:
        test_data = pickle.load(f)

    with open('./data/Train_set_ec_idx.pkl', 'rb') as f:
        EC_id = pickle.load(f)
    id_to_ec = {v: k for k, v in EC_id.items()}

    test_keys = list(test_data.keys())
    test_dataset = ProteinGraphDataset(test_data, range(len(test_keys)))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for graph_data in test_loader:
            graph_data = graph_data.to(device)

            name = graph_data.name[0]
            node_feat = graph_data.node_feat
            edge_index = graph_data.edge_index
            X = graph_data.X
            seq = graph_data.seq
            batch_index = graph_data.batch

            batch_data, mask_data = padding_ver1(node_feat, batch_index, node_feat.shape[1])
            output = model(graph_data.name, X, node_feat, edge_index, seq, batch_index, batch_data, mask_data)

            if isinstance(output, tuple):
                output = output[0]

            true_ec_str = test_data[name][0][1]
            true_label = convert_to_label_vector(true_ec_str, EC_id, NN_config['label_size'])

            pred_probs = torch.sigmoid(output).cpu().numpy()[0]
            pred_binary = (pred_probs > 0.5).astype(int)

            pred_ec_indices = np.where(pred_binary == 1)[0]
            pred_ec_list = [id_to_ec[idx] for idx in pred_ec_indices if idx in id_to_ec]
            pred_ec_str = ';'.join(sorted(pred_ec_list)) if pred_ec_list else 'None'

            match = np.array_equal(pred_binary, true_label)
            if match:
                correct += 1
            total += 1

            results.append({
                'protein': name,
                'true_ec': true_ec_str,
                'pred_ec': pred_ec_str
            })

    output_path = './results/prediction_results.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{'Protein':<16} {'True EC Numbers':<40} {'Predicted EC Numbers':<40}\n")
        f.write("=" * 96 + "\n")
        for r in results:
            f.write(f"{r['protein']:<16} {r['true_ec']:<40} {r['pred_ec']:<40}\n")
        f.write("=" * 96 + "\n")


if __name__ == '__main__':
    main()
