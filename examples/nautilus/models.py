import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import warnings
# import pdb 


class VesselRouteForecasting(nn.Module):
    def __init__(self, rnn_cell=nn.LSTM, input_size=4, hidden_size=150, num_layers=1, output_size=2,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super(VesselRouteForecasting, self).__init__()
        
        # Input and Recurrent Cell
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.rnn_cell = rnn_cell(
            input_size=input_size, 
            num_layers=self.num_layers, 
            hidden_size=self.hidden_size, 
            batch_first=self.batch_first, 
            bidirectional=self.bidirectional, 
            **kwargs
        )

        self.fc_layer = lambda in_feats, out_feats: nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=out_feats),
            nn.ReLU(),
        )

        # Output layers
        self.output_size = output_size
        fc_layers = [2 * hidden_size if self.bidirectional else hidden_size, *fc_layers, self.output_size]
        self.fc = nn.Sequential(
            *[self.fc_layer(in_feats, out_feats) for in_feats, out_feats in zip(fc_layers, fc_layers[1:-1])],
            nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1])
        )
                        
        self.scale = scale
        if self.scale is not None:
            self.mu, self.sigma = self.scale['mu'], self.scale['sigma']    
        else:
            warnings.warn("Instantiated instance without standardization. This may lead to wrong results...")

    def forward_rnn_cell(self, x, lengths):
        # Sort input sequences by length in descending order
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        sorted_x = x[sorted_idx]

        # Pack the sorted sequences
        packed_x = pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True)

        # Initialize ```hidden state``` and ```cell state``` with zeros
        # h0, c0 = torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device),\
        #          torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate packed sequences through LSTM
        # packed_out, (h_n, c_n) = self.rnn_cell(packed_x, (h0, c0))
        packed_out, (h_n, c_n) = self.rnn_cell(packed_x)
        
        # Reorder the output sequences to match the original input order
        _, reversed_idx = sorted_idx.sort(0)

        return packed_out, (h_n, c_n), sorted_idx, reversed_idx
    

    def forward(self, x, lengths):
        # Initialize ```hidden state``` and ```cell state``` with zeros
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        packed_out, (h_n, c_n), _, ix = self.forward_rnn_cell(x, lengths)

        # Unpack the output sequences
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Decode the hidden state of the last time step
        out = self.fc(
                torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1]
        )     
        
        # return torch.add(torch.mul(out[ix], self.sigma), self.mu)
        return torch.add(torch.mul(out[ix], self.sigma.tile(self.output_size // 2)), self.mu.tile(self.output_size // 2))


class ShipTypeVRF(VesselRouteForecasting):
    def __init__(self, embedding, rnn_cell=nn.LSTM, input_size=4, hidden_size=150, num_layers=1, output_size=2,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super().__init__(
            rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=batch_first, fc_layers=fc_layers, scale=scale, bidirectional=bidirectional, **kwargs
        )

        # Output layers
        self.output_size = output_size
        fc_layers = [(2 * hidden_size if self.bidirectional else hidden_size) + embedding.embedding_dim, *fc_layers, self.output_size]
        self.fc = nn.Sequential(
            *[self.fc_layer(in_feats, out_feats) for in_feats, out_feats in zip(fc_layers, fc_layers[1:-1])],
            nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1])
        )

        # Embedding (convert/cluster shiptypes)
        self.embedding = embedding   
        self.dropout = nn.Dropout(0.25)


    def forward(self, x, lengths, shiptypes):
        # Initialize ```hidden state``` and ```cell state``` with zeros
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        _, (h_n, _), ix_sort, ix_rev = self.forward_rnn_cell(x, lengths)

        shiptypes_embedding = self.embedding(shiptypes[ix_sort])

        # Decode the hidden state of the last time step
        out = self.fc(
            self.dropout(
                torch.cat(
                    (
                        torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1, :, :],
                        shiptypes_embedding
                    ), 
                    dim=-1
                )
            )
        )     
        return torch.add(torch.mul(out[ix_rev], self.sigma.tile(self.output_size // 2)), self.mu.tile(self.output_size // 2))
