from torch.utils.data import Dataset, DataLoader
import os
import ast
import torch
import argparse
from encoder import Encoder
from loss import triplet_loss
from univariate_dataset import TimeSeriesDatasetUnivariate
from univariate_dataset import collate_fn as univar_collate
from multivariate_dataset import TimeSeriesDatasetMultivariate
from multivariate_dataset import collate_fn as multivar_collate
from data_utils import get_dataset_path, make_save_path, encode_and_save, avg_batch_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time series experiments.')

    # Training args
    parser.add_argument('--epochs_high_k', type=int, default=20)
    parser.add_argument('--epochs_low_k', type=int, default=15)
    parser.add_argument('--cutoff_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--k_neg', type=str, default='[5]')

    # Architectural args
    parser.add_argument('--nb_layers', type=int, default=10)
    parser.add_argument('--nb_channels', type=int, default=40)
    parser.add_argument('--output_channels', type=int, default=320)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--leaky_relu_slope', type=float, default=0.01)
    parser.add_argument('--representation_dim', type=int, default=160)

    # Other args. 'parse_bool' fixes for bool-parsing isse: https://stackoverflow.com/a/46951029
    def parse_bool(x): return (str(x).lower() in ['true', '1', 'yes'])

    # Whether time series data is multivariate
    parser.add_argument(
        '--multivariate', type=parse_bool)
    # Path to folder containing all datasets
    parser.add_argument('--datasets_path', type=str, default='/dataset/')
    # Path to folder where to save encoded features
    parser.add_argument('--features_path', type=str, default='../features/')
    # Path to specific dataset, for instance 'Coffee'
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--use_cuda', type=parse_bool)

    args = parser.parse_args()

    k_neg_list = ast.literal_eval(args.k_neg)
    multivariate_data = args.multivariate

    # Get all dataset paths except for interpolated datasets which are handled in get_dataset_path
    if args.dataset is not None:
        dataset_paths = [args.datasets_path + args.dataset]
    else:
        interpolated_datasets_dir = 'Missing_value_and_variable_length_datasets_adjusted'
        dataset_paths = [
            f.path for f in os.scandir(args.datasets_path) if f.is_dir() and
            interpolated_datasets_dir not in f.path
        ]

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {
        'num_workers': 0, 'pin_memory': False}

    for dsp in dataset_paths:
        print('Start training on data in {}... \n'.format(dsp))

        for k_neg in k_neg_list:
            print('FordA was trained on K (nb negative samples) = {}'.format(k_neg))

            # Load data
            dataset_path = get_dataset_path(dsp, split='train')
            if multivariate_data:
                train_ds = TimeSeriesDatasetMultivariate(dataset_path, K=k_neg)
                collate_fn = multivar_collate
                in_channels = train_ds.n_dim
            else:
                train_ds = TimeSeriesDatasetUnivariate(dataset_path, K=k_neg)
                collate_fn = univar_collate
                in_channels = 1

            if multivariate_data:
                test_ds = TimeSeriesDatasetMultivariate(get_dataset_path(dsp, split='test'),
                                                        K=k_neg)
            else:
                test_ds = TimeSeriesDatasetUnivariate(get_dataset_path(dsp, split='test'),
                                                      K=k_neg)

            dataset_name = dsp.rsplit(sep='/', maxsplit=1)[1]
            train_path = make_save_path(
                dataset_name, args.features_path, k_neg=k_neg, split='train')
            test_path = make_save_path(
                dataset_name, args.features_path, k_neg=k_neg, split='test')
            
            # Instantiate encoder architecture
            encoder = Encoder(n_layers=args.nb_layers,
                              hidden_out_channels=args.nb_channels,
                              kernel_size=args.kernel_size,
                              last_out_channels=args.output_channels,
                              rep_length=args.representation_dim,
                              in_channels=in_channels).double()

            encoder.load_state_dict(torch.load('fordA_k5.pth'))
            encoder.to(device)

            print('Encoding and saving each series from dataset...\n')
            print('(Saving encoded training set to {})\n'.format(train_path))
            encode_and_save(train_ds, encoder, train_path, device)
            print('(Saving encoded test set to {})\n'.format(test_path))
            encode_and_save(test_ds, encoder, test_path, device)
