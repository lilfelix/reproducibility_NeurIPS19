import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TimeSeriesDatasetIHEPC(Dataset):
    """IHEPC time series dataset from the UCI Machine Learning Repository
       dataset source: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption#
       data parsing inspired by: https://github.com/amirrezaeian/Individual-household-electric-power-consumption-Data-Set-/blob/master/data_e_power.ipynb
    """

    def __init__(self, dataset_path, window_size, K=10, transform=None, n_dim=7, generate_labels=False):
        """
        Args:
        data_path (string): Path to data file containing samples (.tsv for univariate)
        K (int): Number of negative sub-series used to compute loss
        window_size (int): Must be either 1440, 12*7*1440 or 5*10^5
        transform (callable, optional): Optional transform to be applied on a sample.
        n_dim (int): Number if features to use. Either all 7 channels or only 1
        """
        self.n_dim = n_dim
        self.transform = transform
        self.K = K

        if self.n_dim == 1:  # Use only 'Global_active_power' feature if n_dim is 1
            cols = ['Global_active_power', 'Date', 'Time']
        else:
            cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Date', 'Time']

        # Missing values are replaced with np.nan
        # The two columns 'Date' and 'Time' are merged into one: 'dt'.
        df = pd.read_csv(dataset_path, sep=';',
                         parse_dates={'dt': ['Date', 'Time']},
                         infer_datetime_format=True,
                         low_memory=False,
                         na_values=['nan', '?'],
                         index_col='dt',
                         usecols=cols)

        # Linear interpolation of missing values followed by z-normalization
        df_interp = df.interpolate()
        self.samples = (df_interp - df_interp.mean()) / df_interp.std()

        if window_size not in [1440, 12*7*1440, 5*(10**5)]:
            raise ValueError(
                "Window size must be 'day','quarter' or 'all values'")
        else:
            self.window_size = window_size

        # During training, only a single long time series is used
        if window_size == 5*(10**5):
            self.n_windows = 1
            self.win_start = [0]
            self.win_end = [window_size - 1]
            self.labels = [0]
            self.long_tensor = torch.tensor(self.get_one_sample(0))
        else:
            # Determine indices for windows
            self.win_end = np.array([(self.window_size + i)
                                     for i in range(1, len(self.samples) - self.window_size - 1)], dtype=int)
            self.win_start = self.win_end - self.window_size
            # len(self.samples) - self.window_size - 2
            self.n_windows = len(self.win_start)
            print(f'len(self.samples) = {len(self.samples)}')
            print(
                f'win_start[0] = {self.win_start[0]}, win_start[-1] = {self.win_start[-1]}')
            print(
                f'win_end[0] = {self.win_end[0]}, win_end[-1] = {self.win_end[-1]}, n_windows = {self.n_windows}')

            # Generate labels
            if generate_labels:
                labels_path = os.path.splitext(dataset_path)[
                    0] + '_labels' + '_ndim_' + str(self.n_dim) + '_wsize_' + str(window_size) + '.npy'
                try:
                    print(
                        f'Loading labels from path {labels_path} (same folder as dataset)..')
                    self.labels = np.load(labels_path)
                    print('Done loading labels')
                    print(
                        f'In [TimeSeriesDatasetIHEPC] constructor: Done generating labels of shape: {self.labels.shape}')
                except:
                    print(
                        f'Generating labels and saving to path {labels_path} (same folder as dataset)..')
                    self.labels = self.generate_labels()
                    np.save(labels_path, self.labels)

            else:
                # Training the encoder doesn't require labels
                self.labels = [0 for i in range(self.n_windows)]

        print(
            f'In [TimeSeriesDatasetIHEPC] constructor: n_windows in dataloader: {self.n_windows}')

    def __len__(self):
        return self.n_windows  # len(self.samples)

    def get_one_sample(self, idx):
        # Extract rows corresponding to window[idx]
        start = self.win_start[idx]
        end = self.win_end[idx]  # [start:499998]
        return self.samples.iloc[start:end, :].values

    # Get triplet sample
    def __getitem__(self, idx):
        """Samples three types of sub-series from the dataset:
           1 reference, 1 positive and K negative.
            This method corresponds to Algorithm 1 in the paper.

        Args:
            idx (int): index of time series containing 'ref' and 'pos' as sub-series

        Returns:
            (torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor): reference sample of dim (1,L),
                                                          positive sample of dim (1,L'),
                                                          K negative samples of dim (K, L'')
                                                          (each sample zero-padded to the
                                                          length of the longest i.e L'')
        """
        with torch.no_grad():
            if self.window_size != 5 * (10 ** 5):
                # Extract rows corresponding to window[idx]
                # values = self.get_window_values(idx)
                values = self.get_one_sample(idx)
                y_i = torch.tensor(values)
                # print('In [__getitem__]: returning item:', y_i)
                return y_i

            # the code below is used for the case of window_size = 500K
            # Keep a single tensor for the long time series, when training
            y_i = self.long_tensor

            """if self.window_size == 5 * (10 ** 5):
                # Keep a single tensor for the long time series, when training
                y_i = self.long_tensor
            else:
                # Extract rows corresponding to window[idx]
                values = self.get_window_values(idx)
                y_i = torch.tensor(values)"""

            s_i = self.win_end[idx] - self.win_start[idx]
            s_pos = np.random.randint(low=1, high=s_i + 1)  # [1,s_i]
            s_ref = np.random.randint(low=s_pos, high=s_i + 1)  # [s_pos,s_i]

            # pick x_ref u.a.r among subseries of y_i of length s_ref
            x_ref_start = np.random.randint(low=0, high=(s_i - s_ref) + 1)
            x_ref = y_i[x_ref_start:x_ref_start + s_ref]

            # pick x_pos u.a.r among subseries of x_ref of length s_pos
            x_pos_start = np.random.randint(low=0, high=(s_ref - s_pos) + 1)
            x_pos = x_ref[x_pos_start:x_pos_start + s_pos]

            # TODO: discuss assumption: allow multiple neg samples from same time series
            # pick K indices u.a.r with replacement from [1,n_windows]
            neg_idxs = np.random.choice(self.n_windows, self.K, replace=True)
            x_neg_list = []  # list of negative samples (of varying length)

            for k in neg_idxs:
                if self.window_size == 5 * (10 ** 5):
                    # Keep a single tensor for the long time series, when training
                    neg_sample = self.long_tensor
                else:
                    # Extract rows corresponding to window[k]
                    neg_sample = self.get_window_values(k)

                s_k = self.win_end[k] - self.win_start[k]
                s_neg = np.random.randint(low=1, high=s_k + 1)
                x_neg_start = np.random.randint(low=0, high=(s_k - s_neg) + 1)
                x_neg = neg_sample[x_neg_start:x_neg_start + s_neg]
                x_neg_list.append(x_neg.clone().detach().requires_grad_(True))

            # Use torch API for RNNs to pad negative samples to fixed length, L, and stack them in tensor of dim (K,C,L).
            x_negs_padded = pad_sequence(x_neg_list,
                                         batch_first=True,
                                         padding_value=0)

            # tell autograd to begin recording operations on tensors and ensure float64 types
            sample = {
                'x_ref': x_ref.transpose(0, 1).double(),
                'x_pos': x_pos.transpose(0, 1).double(),
                'x_negs': x_negs_padded.transpose(1, 2).double(),
                'label': self.labels[idx]
            }
            return sample

    def generate_labels(self):
        """
        Creates labels from the time series according to the instructions of the paper.
         for each time step:
             predict the discrepancy between the mean value of the series for the next time period (either a day or quarter)
             and the one for the previous period.
        """
        # First and last windows don't have labels
        labels = np.zeros((self.n_windows, self.n_dim))

        for i, win_i in enumerate(self.win_start):
            if win_i % 100000 == 0:
                print(f'In [generate_labels]: generating label for win={win_i}')

            prev_win = self.samples.iloc[win_i-1:(win_i-1) + self.window_size]
            next_win = self.samples.iloc[win_i+1:(win_i+1) + self.window_size]
            labels[i] = (next_win.mean() - prev_win.mean()).values

        return labels


if __name__ == "__main__":
    dataset_path = '/Users/felix/Skola/5/Deep Learning Extended DD2412/Reprod_NIPS19/household_power_consumption/household_power_consumption_TRAIN.txt'
    ds = TimeSeriesDatasetIHEPC(dataset_path, window_size=5*(10**5), K=10)

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
