import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from data_utils import normalize_multivar_dataset
from dataset_metadata import multivariate_datasets, multivar_non_normalized_datasets


class TimeSeriesDatasetMultivariate(Dataset):
    """Multivariate time series dataset from the UEA archive."""

    def __init__(self, dataset_path, K, transform=None):
        """
        Args:
            data_path (string): Path to data file containing samples (tsv/ts for univariate/multivariate respectively).
            K (int): Number of negative sub-series used to compute loss
            transform (callable, optional): Optional transform to be applied on a sample.

        For more info on format of multivariate data (sktime formatted ts files), see:
        https://github.com/alan-turing-institute/sktime/blob/master/examples/loading_data.ipynb
        """
        dataset_name = dataset_path.rsplit(sep='/', maxsplit=2)[1]

        samples, labels = load_from_tsfile_to_dataframe(dataset_path)
        self.samples = samples
        self.labels = labels

        # Number of dimensions of each sample
        self.n_dim = len(samples.columns)

        # Save length of each (non-padded) sample (length can vary across dimensions)
        self.sample_lengths = np.zeros((len(samples), self.n_dim), dtype=int)
        for d, col in enumerate(samples.columns):
            self.sample_lengths[:, d] = samples[col].apply(
                lambda x: len(x)).values.astype(int)

       # Check if data needs normalizing
        if dataset_name in multivar_non_normalized_datasets:
            self.samples = normalize_multivar_dataset(self.samples)

        # Dataframe where NaN values have been replaced with zeros
        for i in range(len(samples)):
            for dim in samples.columns:
                self.samples[dim][i] = self.samples[dim][i].fillna(0)

        self.transform = transform
        self.K = K

    def __len__(self):
        return len(self.samples)

    def get_one_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        sample = {'data': self.create_tensor(idx), 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def create_tensor(self, idx):
        """
        Helper method to create torch.DoubleTensor from pandas DataFrame.
        Upon creating the tensor, the data is copied to a new memory location.
        Hence, modifying the tensor won't affect the pandas DataFrame.

        Args:
            idx (int): Index of sample in DataFram to generate tensor from

        Returns:
            sample_tensor (torch.DoubleTensor): sample of dim (n_dims, sample_len)
        """
        sample_tensor = torch.zeros(
            (self.n_dim, self.sample_lengths[idx].max()),
            dtype=torch.double)

        for i, col in enumerate(self.samples.columns):
            # In rare cases, a sample has different length across it's dimensions
            dim_length = len(self.samples[col][idx])

            # Create zero-padded torch.Tensor using data from pandas DataFrame
            sample_tensor[i, 0:dim_length] = torch.tensor(
                self.samples[col][idx],
                dtype=torch.double)

        return sample_tensor

    def __getitem__(self, idx):
        """
        Samples three types of sub-series from the dataset:
        1 reference, 1 positive and K negative.
        This method corresponds to Algorithm 1 in the paper.

        Args:
            idx (int): index of time series containing 'ref' and 'pos' as sub-series

        Returns:
            (torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor): reference sample of dim (n_dim,L),
                                                          positive sample of dim (n_dim,L'),
                                                          K negative samples of dim (n_dim, L'')
                                                          (each negative sample zero-padded to the
                                                          length of the longest i.e L'')
        """
        y_i = self.create_tensor(idx)
        s_i = self.sample_lengths[idx].min()

        # Same size of sub-series within each dim of sample
        s_pos = np.random.randint(low=1, high=s_i + 1)  # [1,s_i]
        s_ref = np.random.randint(low=s_pos, high=s_i + 1)  # [s_pos,s_i]

        # Pick x_ref u.a.r among subseries of y_i of length s_ref
        # Extract same range of indices across all dims.
        x_ref_start = np.random.randint(low=0, high=(s_i - s_ref) + 1)
        x_ref = y_i[:, x_ref_start:x_ref_start + s_ref]

        # Pick x_pos u.a.r among subseries of x_ref of length s_pos
        # Extract same range of indices across all dims.
        x_pos_start = np.random.randint(low=0, high=(s_ref - s_pos) + 1)
        x_pos = x_ref[:, x_pos_start:x_pos_start + s_pos]

        # TODO: Discuss assumption: allow multiple neg samples from same time series
        # Pick K indices u.a.r with replacement from [1,N]
        neg_idxs = np.random.choice(len(self.samples), self.K, replace=True)
        # List of negative samples. Each of dim (n_dim,sample_len)
        x_neg_list = []

        for k in neg_idxs:
            neg_sample = self.create_tensor(k)
            s_k = self.sample_lengths[k].min()
            s_neg = np.random.randint(low=1, high=s_k + 1)
            x_neg_start = np.random.randint(low=0, high=(s_k - s_neg) + 1)
            x_neg = neg_sample[:, x_neg_start:x_neg_start + s_neg]
            x_neg_list.append(x_neg.transpose(0, 1))  # (L, n_dim)

        # Use torch API for RNNs to pad negative samples to fixed length, L', and stack them in tensor of dim (L',K,n_dim).
        # Multiple such tensors are later collated into batch tensor, which is why dimension L must come first (see docs for pad_sequence)
        x_negs_padded = pad_sequence(x_neg_list,
                                     batch_first=True,
                                     padding_value=0)  # (K, L', n_dim)

        # Tell autograd to begin recording operations on tensors
        # Transpose before collate_fn which requires dimension to be padded to lie first
        sample = {
            'x_ref': x_ref.transpose(0, 1),  # (L''',n_dim)
            'x_pos': x_pos.transpose(0, 1),  # (L'',n_dim)
            'x_negs': x_negs_padded.transpose(0, 1),  # (L', K, n_dim)
            'label': self.labels[idx]
        }
        return sample


def collate_fn(sample_list):
    """Custom collate_fn that is called with list of multivariate samples to yield a mini-batch

    It preserves the data structure, e.g., if each sample is a dictionary,
    it outputs a dictionary with the same set of keys but batched Tensors as values
    (or lists if the values can not be converted into Tensors).

    Args:
        sample_list (list): list of samples, each containing 1 'ref', 1 'pos' and K 'neg' sub-series

    Returns:
        (dict): {
                    x_ref (torch.DoubleTensor): mini-batch tensor of dim (B,n_dim,L)
                    x_pos (torch.DoubleTensor): mini-batch tensor of dim (B,n_dim,L')
                    x_negs (torch.DoubleTensor): mini-batch tensor of dim (B,K,n_dim,L'')
                    label (list): mini-batch tensor of dim (B)
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

    # Use torch API for RNNs to pad samples to fixed length, L, and stack them in batch-tensor of dim (B,n_dim,L).
    x_ref_batch = pad_sequence(
        x_ref_batch,
        batch_first=True,
        padding_value=0)  # (B,L,n_dim)
    x_ref_batch = x_ref_batch.transpose(1, 2)  # (B,n_dim,L)

    x_pos_batch = pad_sequence(
        x_pos_batch,
        batch_first=True,
        padding_value=0)  # (B,L,n_dim)
    x_pos_batch = x_pos_batch.transpose(1, 2)  # (B,n_dim,L)

    # Pad neg tensors with varying length of first dim L, and produce batch (B,K,n_dim,L') where L' is padded length
    x_negs_batch = pad_sequence(x_negs_batch,
                                batch_first=True,
                                padding_value=0)  # (B, L', K, n_dim)
    x_negs_batch = x_negs_batch.transpose(1, 2)  # (B, K, L', n_dim)
    x_negs_batch = x_negs_batch.transpose(2, 3)  # (B, K, n_dim, L')

    return {
        'x_ref': x_ref_batch,
        'x_pos': x_pos_batch,
        'x_negs': x_negs_batch,
        'label': label_batch
    }


if __name__ == "__main__":
    dataset_path = '../Multivariate_ts/ERing/ERing_TRAIN.ts'
    ds = TimeSeriesDatasetMultivariate(dataset_path, K=3)

    sample = ds.get_one_sample(0)
    print('test get_one_sample')
    print(sample['data'].shape)
    print()

    sample = ds.__getitem__(0)
    print('test __getitem__')
    print('x_ref shape:', sample['x_ref'].shape)
    print('x_pos shape:', sample['x_pos'].shape)
    print('x_neg shape:', sample['x_negs'].shape)
    print()

    mini_batch = collate_fn([ds.__getitem__(0), ds.__getitem__(1)])
    print('test collate_fn')
    print('x_ref_batch shape:', mini_batch['x_ref'].shape)
    print('x_pos_batch shape:', mini_batch['x_pos'].shape)
    print('x_neg_batch shape:', mini_batch['x_negs'].shape)
