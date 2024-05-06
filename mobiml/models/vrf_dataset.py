import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class VRFDataset(Dataset):
    def __init__(self, data, scaler=None, dtype=np.float32, **kwargs):
        # pdb.set_trace()
        self.samples = data['samples'].values
        self.labels = data['labels'].values.ravel()
        self.lengths = [len(l) for l in self.samples]

        self.dtype = dtype        

        if scaler is None:
            self.scaler = StandardScaler().fit(np.concatenate(self.samples))
        else:
            self.scaler = scaler

    def pad_collate(self, batch):
        '''
        xx: Samples (Delta Trajectory), yy: Labels (Next Delta), ll: Lengths
        '''
        (xx, yy, ll) = zip(*batch)
        
        # Right Zero Padding with Zeroes (for delta trajectory)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, torch.stack(yy), torch.tensor(ll)
    
    def __getitem__(self, item):
        return torch.tensor(self.scaler.transform(self.samples[item]).astype(self.dtype)),\
               torch.tensor(self.labels[item].reshape(1, -1).astype(self.dtype)),\
               torch.tensor(self.lengths[item]),\

    def __len__(self):
        return len(self.labels)
