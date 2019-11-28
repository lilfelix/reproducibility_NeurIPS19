import os
import torch
import numpy as np
import argparse
from encoder import Encoder
from ihepc_dataset import TimeSeriesDatasetIHEPC
from data_utils import get_dataset_path
from torch.utils.data import DataLoader


def encode_IHEPC(dataset,
                 encoder,
                 loader_kwargs,
                 day_or_quarter,
                 train_or_test,
                 device,
                 features_path):
    """
    Iterate over dataset, pass batches of samples through encoder and save
    their representations.

    :param dataset:
    :param encoder:
    :param loader_kwargs:
    :param day_or_quarter
    :param train_or_test:
    :param device:
    :param features_path:
    :return:
    """
    # Set encoder to inference mode
    encoder.eval()

    data_loader = DataLoader(dataset, **loader_kwargs)

    # Directory for saving the encodings and labels
    encodings_dir = os.path.join(features_path,
                                 day_or_quarter, train_or_test)
    if not os.path.exists(encodings_dir):
        os.makedirs(encodings_dir)

    labels_file = encodings_dir + '/labels.npy'
    np.save(labels_file, dataset.labels)

    for i_batch, batch in enumerate(data_loader):
        # changing from (10, 1440, 1) to (10, 1, 1440) for the encoder to understand
        batch = batch.permute(0, 2, 1)
        encoding_batch = encoder(batch.to(device))

        # save the encoding_batch for the current batch so space could be freed for the next batch
        np.save(f'{encodings_dir}/i_batch_{i_batch}.npy',
                encoding_batch.detach().cpu().numpy())

        if i_batch % 100 == 0:
            print(f'Saved batch {i_batch}')
        del encoding_batch

    print(f'Encodings saved to {encodings_dir}')


def parse_args():
    """ Parse command line arguments. Args with default=None are mandatory"""
    def parse_bool(x): return (str(x).lower() in ['true', '1', 'yes'])
    parser = argparse.ArgumentParser()

    # Architectural args
    parser.add_argument('--nb_layers', type=int, default=10)
    parser.add_argument('--nb_channels', type=int, default=30)
    parser.add_argument('--output_channels', type=int, default=160)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--leaky_relu_slope', type=float, default=0.01)
    parser.add_argument('--representation_dim', type=int, default=80)
    parser.add_argument('--in_channels', type=int, default=1)

    # Path to IHEPC dataset, encoder and features
    parser.add_argument('--datasets_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--encoder_path', type=str, default=None)
    parser.add_argument('--features_path', type=str, default=None)

    # Hyperparameters for generating encodings
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--use_cuda', type=parse_bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    """ Encode time series from IHEPC dataset. Samples have either 'day' or 'quarter' size"""
    args = parse_args()

    # Determine whether to run on CPU or GPU
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': False}

    # Instantiate encoder architecture
    encoder = Encoder(n_layers=args.nb_layers,
                      hidden_out_channels=args.nb_channels,
                      kernel_size=args.kernel_size,
                      last_out_channels=args.output_channels,
                      rep_length=args.representation_dim,
                      in_channels=args.in_channels).double()

    encoder.load_state_dict(torch.load(args.encoder_path,
                                       map_location=device)['model_state_dict'])
    encoder.to(device)

    dataset = TimeSeriesDatasetIHEPC(get_dataset_path(args.datasets_path,
                                                      split=args.split),
                                     args.window_size,
                                     n_dim=args.in_channels,
                                     generate_labels=True)
    if args.window_size == 1440:
        day_or_quarter = 'day'
    elif args.window_size == 12*7*1440:
        day_or_quarter = 'quarter'
    else:
        raise ValueError("Invalid window size. Must be day or quarter")

    print('Encoding and saving samples...\n')
    encode_IHEPC(dataset,
                 encoder,
                 dataloader_kwargs,
                 day_or_quarter,
                 args.split,
                 device,
                 features_path=args.features_path)
