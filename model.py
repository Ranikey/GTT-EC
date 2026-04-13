import torch
import torch.nn as nn
from encoder import *
from decoder import *

class GraphEC_model(nn.Module):

    def __init__(self, encoder_model, decoder_model):
        super(GraphEC_model, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model

    def forward(self, name, X, h_V, edge_index, seq, batch_id, batch_data, mask_data):
        encoder_output, mask = self.encoder(X, h_V, edge_index, batch_id, batch_data)
        output = self.decoder(name, encoder_output, mask)
        return output
