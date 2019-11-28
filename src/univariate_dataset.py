import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset_metadata import univar_non_normalized_datasets, univar_interpolated_values_datasets


class TimeSeriesDatasetUnivariate(Dataset):
    """Univariate time series dataset from the UCR archive."""

    def __init__(self, dataset_path, K, transform=None):
        """
        Args:
        data_path (string): Path to data file containing samples (.tsv for univariate)
        K (int): Number of negative sub-series used to compute loss
        transform (callable, optional): Optional transform to be applied on a sample.

        For more info on format of multivariate data (sktime formatted ts files), see:
        https://github.com/alan-turing-institute/sktime/blob/master/examples/loading_data.ipynb
        """
        df = pd.read_csv(
            dataset_path, sep='\t', header=None)

        # Check if data needs normalizing
        dataset_name = dataset_path.rsplit(sep='/', maxsplit=2)[1]
        if dataset_name in univar_non_normalized_datasets:
            label_col = df.iloc[:, 0]  # Save label column
            all_values = df.iloc[:, 1:].values
            mean = all_values[~np.isnan(all_values)].mean()
            stdev = all_values[~np.isnan(all_values)].std()
            df = (df-mean)/stdev # Normalize
            df.iloc[:, 0] = label_col  # Insert label column

        # Number of values in every sample that aren't NaN. -1 to not count label idx
        self.raw_sample_lenghts = df.count(axis=1) - 1

        # Dataframe where NaN values have been replaced with zeros
        self.samples = df.fillna(0)

        # For compatability with multivariate dataset, attribute checked in encode_and_save()
        self.n_dim = 1

        self.transform = transform
        self.K = K

    def __len__(self):
        return len(self.samples)

    def get_one_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.samples.iloc[idx, 0]
        data = self.samples.iloc[idx, 1:].values
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get triplet sample
    def __getitem__(self, idx):
        """Samples three types of sub-series from the dataset:
           1 reference, 1 positive and K negative.
            This method corresponds to Algorithm 1 in the paper.

        Args:
            idx (int): index of time series containing 'ref' and 'pos' as sub-series

        Returns:
            (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor): reference sample of dim (1,L),
                                                          positive sample of dim (1,L'),
                                                          K negative samples of dim (K, L'')
                                                          (each sample zero-padded to the
                                                          length of the longest i.e L'')
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        y_i = torch.tensor(self.samples.iloc[idx, 1:].values)
        s_i = self.raw_sample_lenghts.iloc[idx]  # y_i might be zero padded
        s_pos = np.random.randint(low=1, high=s_i + 1)  # [1,s_i]
        s_ref = np.random.randint(low=s_pos, high=s_i + 1)  # [s_pos,s_i]

        # pick x_ref u.a.r among subseries of y_i of length s_ref
        x_ref_start = np.random.randint(low=0, high=(s_i - s_ref) + 1)
        x_ref = y_i[x_ref_start:x_ref_start + s_ref]

        # pick x_pos u.a.r among subseries of x_ref of length s_pos
        x_pos_start = np.random.randint(low=0, high=(s_ref - s_pos) + 1)
        x_pos = x_ref[x_pos_start:x_pos_start + s_pos]

        # TODO: discuss assumption: allow multiple neg samples from same time series
        # pick K indices u.a.r with replacement from [1,N]
        neg_idxs = np.random.choice(len(self.samples), self.K, replace=True)
        x_neg_list = []  # list of negative samples (of varying length)

        for k in neg_idxs:
            neg_sample = self.samples.iloc[k, 1:].values
            s_k = self.raw_sample_lenghts.iloc[k]
            s_neg = np.random.randint(low=1, high=s_k + 1)
            x_neg_start = np.random.randint(low=0, high=(s_k - s_neg) + 1)
            x_neg = neg_sample[x_neg_start:x_neg_start + s_neg]
            x_neg_list.append(torch.tensor(x_neg))

        # Use torch API for RNNs to pad negative samples to fixed length, L, and stack them in tensor of dim (L,K).
        # Multiple such tensors are later collated into batch tensor, which is why dimension L must come first (see docs for pad_sequence)
        x_negs_padded = pad_sequence(x_neg_list,
                                     batch_first=True,
                                     padding_value=0).transpose(0, 1)

        label = self.samples.iloc[idx, 0]
        # tell autograd to begin recording operations on tensors and ensure float64 types
        sample = {
            # 'x_ref': x_ref.requires_grad_(True).double(),
            # 'x_pos': x_pos.requires_grad_(True).double(),
            # 'x_negs': x_negs_padded.requires_grad_(True).double(),
            'x_ref': x_ref.double(),
            'x_pos': x_pos.double(),
            'x_negs': x_negs_padded.double(),
            'label': label
        }
        return sample


def collate_fn(sample_list):
    """Custom collate_fn that is called with list of univariate samples to yield a mini-batch

    It preserves the data structure, e.g., if each sample is a dictionary,
    it outputs a dictionary with the same set of keys but batched Tensors as values
    (or lists if the values can not be converted into Tensors).

    Args:
        sample_list (list): list of samples, each containing 1 'ref', 1 'pos' and K 'neg' sub-series

    Returns:
        (dict): {
                    x_ref (torch.FloatTensor): mini-batch tensor of dim (B,1,L)
                    x_pos (torch.FloatTensor): mini-batch tensor of dim (B,1,L')
                    x_negs (torch.FloatTensor): mini-batch tensor of dim (B,K,1,L'')
                    label (torch.FloatTensor): mini-batch tensor of dim (B)
                }
    """
    x_ref_batch = []
    x_pos_batch = []
    x_negs_batch = []
    label_batch = []

    for sample in sample_list:
        x_ref_batch.append(sample["x_ref"])
        x_pos_batch.append(sample["x_pos"])
        x_negs_batch.append(sample["x_negs"])
        label_batch.append(sample["label"])

    # Use torch API for RNNs to pad samples to fixed length, L, and stack them in batch-tensor of dim (B,1,L).
    x_ref_batch = pad_sequence(
        x_ref_batch, batch_first=True, padding_value=0)
    x_ref_batch = x_ref_batch.unsqueeze(1)
    x_pos_batch = pad_sequence(
        x_pos_batch, batch_first=True, padding_value=0)
    x_pos_batch = x_pos_batch.unsqueeze(1)

    # Pad neg tensors with varying length of first dim L, and produce batch (B,K,1,L') where L' denotes longest neg sample
    x_negs_batch = pad_sequence(x_negs_batch,
                                batch_first=True,
                                padding_value=0)
    x_negs_batch = x_negs_batch.transpose(1, 2)  # (B,L',K)->(B,K,L')
    x_negs_batch = x_negs_batch.unsqueeze(2)  # (B,K,L')->(B,K,1,L')

    return {
        'x_ref': x_ref_batch,
        'x_pos': x_pos_batch,
        'x_negs': x_negs_batch,
        'label': torch.tensor(label_batch)
    }
