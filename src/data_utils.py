import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from univariate_dataset import TimeSeriesDatasetUnivariate, collate_fn
from dataset_metadata import univar_interpolated_values_datasets, multivariate_datasets


def get_dataset_path(datasets_path, split='train'):
    # Right split on the last occurence of '/'
    root, name = datasets_path.rsplit(sep='/', maxsplit=1)

    # Check if data has interpolated values. These are loaded from another folder
    if name in univar_interpolated_values_datasets:
        interpolated_datasets_dir = 'Missing_value_and_variable_length_datasets_adjusted'
        datasets_path = os.path.join(
            root, interpolated_datasets_dir, name)

    # Determine file format
    if 'household_power_consumption' in datasets_path:
        file_format = '.txt'
    elif name in multivariate_datasets:
        file_format = '.ts'
    else:
        file_format = '.tsv'

    if split == 'train':
        dataset_path = datasets_path + '/' + name + '_TRAIN' + file_format
    elif split == 'test':
        dataset_path = datasets_path + '/' + name + '_TEST' + file_format
    return dataset_path


def avg_batch_norm(tensor):
    """Computes L2 norm of each row in 2D-tensor and averages the result"""
    return torch.sum(torch.norm(tensor.detach().squeeze(), dim=1)) / tensor.shape[0]


def encode_and_save(dataset, model, path, device, IHEPC=False):
    """
    Iterate over dataset, pass samples through encoder and save
    their representations.

    Args:
        dataset (Dataset object): the dataset to input to the encoder.
        model (): the encoder.
        path (str): the path where we save the features,
                    f. ex: 'features/[dataset_name]/[train/test]/k_[K]/'
        IHEPC (bool): samples are saved differently for IHEPC
    """
    # Set model to inference mode
    model.eval()

    nb_samples = len(dataset)

    # When saving window encodings for IHEPC first and last encodings are discarded
    # These encodings don't have a label (computed w.r.t prev and next window)
    if IHEPC:
        sample_range = range(1, dataset.n_windows - 1)
    else:
        sample_range = range(nb_samples)

    print(
        f'Generate encodings for n_windows: {dataset.n_windows}.  last window idx: {dataset.n_windows - 2}')

    features_list = []
    labels_list = []
    for sample_idx in sample_range:
        if sample_idx % 1000 == 0:
            print(f'{sample_idx} / {dataset.n_windows - 2}')
        sample = dataset.get_one_sample(sample_idx)
        series = sample['data']
        label = sample['label']
        labels_list.append(label)

        if IHEPC:
            # Transpose and unsqueeze once to get sample.shape = [1,C,T]
            series = series.T
            series = np.expand_dims(series, 0)
        elif dataset.n_dim == 1:
            # Unsqueeze twice to get sample.shape = [1,1,T]
            series = np.expand_dims(series, 0)
            series = np.expand_dims(series, 0)
        else:
            # Unsqueeze once to get sample.shape = [1,C,T]
            series = np.expand_dims(series, 0)

        # Make torch.DoubleTensor and encode sample
        feature_tensor = model(torch.from_numpy(series).double().to(device))
        # Detach from grads and convert to numpy
        features = feature_tensor.cpu().detach().numpy()
        features_list.append(features)

    # This array will have shape [N,]
    labels_array = np.asarray(labels_list)
    # This array is [N, 1, R, 1]
    features_array = np.asarray(features_list)
    # Squeeze to [N, R]
    features_array = np.squeeze(features_array, axis=(1, 3))
    # Save as a dictionary with keys features and labels
    np.savez(path, features=features_array, labels=labels_array)


def make_save_path(dataset_name, features_path, k_neg, split='train'):
    """
    Args:
        dataset_name (str): Name of dataset.
        features_path (str): Path to folder for all features.
        k_neg (int): value of K during training
        split (str): 'train' | 'test'
    Returns:
        path (str): Path for features of one sample to be saved
    """
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    partial_path = features_path + dataset_name + '/'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    partial_path += split + '/'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    path = partial_path + 'k_' + str(k_neg) + '.npz'
    return path


def normalize_multivar_dataset(df):
    n_samples = len(df)
    n_dim = len(df.columns)

    # Each sample has a series of values for each dimension (column).
    # 1. Concatenate series along column (across all samples for current dim)
    # 2. Compute mean and sample standard deviation for concatenated series
    for dim in df.columns:
        mean_and_std = pd.concat(df[dim].values).agg(['mean', 'std']).values
        dim_mean = mean_and_std[0]
        dim_std = mean_and_std[1]

        # Z-normalize each sample, one dimension at the time
        for i in range(n_samples):
            df[dim][i] = (df[dim][i] - dim_mean) / dim_std

    return df


if __name__ == "__main__":
    dataset_path = '/dataset/Coffee/Coffee_TRAIN.tsv'
    train_ds = TimeSeriesDatasetUnivariate(dataset_path=dataset_path, K=5)

    train_dataloader = DataLoader(train_ds,
                                  batch_size=10,
                                  shuffle=False,
                                  num_workers=1,
                                  collate_fn=collate_fn)

    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['label'], sample_batched['x_ref'].size(),
              sample_batched['x_pos'].size(), sample_batched['label'].size())
