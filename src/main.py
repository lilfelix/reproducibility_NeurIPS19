from torch.utils.data import Dataset, DataLoader
import os
import ast
import torch
import argparse
from encoder import Encoder
from loss import triplet_loss_pos, triplet_loss_neg
from univariate_dataset import TimeSeriesDatasetUnivariate, collate_fn as univar_collate
from multivariate_dataset import TimeSeriesDatasetMultivariate, collate_fn as multivar_collate
from ihepc_dataset import TimeSeriesDatasetIHEPC
from data_utils import get_dataset_path, make_save_path, encode_and_save, avg_batch_norm

import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
    Keys are device ids as integers.
    Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map.values()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time series experiments.')

    # Training args
    parser.add_argument('--steps_high_k', type=int, default=2000)
    parser.add_argument('--steps_low_k', type=int, default=1500)
    parser.add_argument('--cutoff_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--k_neg', type=str, default='[1,2,5,10]')

    # Architectural args
    parser.add_argument('--nb_layers', type=int, default=10)
    parser.add_argument('--nb_channels', type=int, default=30)
    parser.add_argument('--output_channels', type=int, default=160)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--leaky_relu_slope', type=float, default=0.01)
    parser.add_argument('--representation_dim', type=int, default=80)

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

    # Specifically for IHEPC. Must be specified if dataset is IHEPC (None will raise Exception)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--in_channels', type=int, default=1)

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
            print('K (nb negative samples) = {}'.format(k_neg))

            # Load data
            dataset_path = get_dataset_path(dsp, split='train')

            if 'household_power_consumption' in dataset_path:
                # IHEPC encoder is trained on a single long time series
                collate_fn = None
                in_channels = args.in_channels
                args.batch_size = 1
                train_ds = TimeSeriesDatasetIHEPC(dataset_path,
                                                  window_size=args.window_size,
                                                  K=k_neg,
                                                  n_dim=in_channels)
            elif multivariate_data:
                train_ds = TimeSeriesDatasetMultivariate(dataset_path, K=k_neg)
                collate_fn = multivar_collate
                in_channels = train_ds.n_dim
            else:
                train_ds = TimeSeriesDatasetUnivariate(dataset_path, K=k_neg)
                collate_fn = univar_collate
                in_channels = 1

            train_dataloader = DataLoader(train_ds, batch_size=args.batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=kwargs['num_workers'])

            # Instantiate encoder architecture
            encoder = Encoder(n_layers=args.nb_layers,
                              hidden_out_channels=args.nb_channels,
                              kernel_size=args.kernel_size,
                              last_out_channels=args.output_channels,
                              rep_length=args.representation_dim,
                              in_channels=in_channels).double()

            encoder.to(device)

            optimizer = torch.optim.Adam(encoder.parameters(),
                                         lr=args.learning_rate,
                                         betas=(args.beta_1, args.beta_2))

            nb_steps = args.steps_high_k if k_neg >= args.cutoff_k else args.steps_low_k
            nb_samples = len(train_ds)
            epoch_steps = int(nb_samples/args.batch_size)

            epoch = 0
            step = 0

            while step < nb_steps:

                for i_batch, triplet_batch in enumerate(train_dataloader):

                    if step == nb_steps:
                        break

                    if step % epoch_steps == 0:
                        print('Training epoch {}'.format(epoch))
                        epoch += 1

                    # Pass x_ref, x_pos through encoder and compute positive loss term.
                    # The input to the encoder should be [batch_size, 1, nb_timesteps] which yields output as [batch_size, 160, 1]
                    out_ref_batch = encoder(triplet_batch['x_ref'].to(device))
                    out_pos_batch = encoder(triplet_batch['x_pos'].to(device))

                    optimizer.zero_grad()
                    loss_pos = triplet_loss_pos(out_ref_batch, out_pos_batch)
                    # Retrain_graph=True ensures cached values aren't freed
                    loss_pos.backward(retain_graph=True)
                    optimizer.step()

                    # Save scalar loss term and free memory that won't be used again
                    loss_pos_scalar = loss_pos.item()
                    del out_pos_batch
                    del loss_pos
                    out_ref_batch = out_ref_batch.detach()
                    torch.cuda.empty_cache()

                    # Save shape before flattening: M = B x K s.t a batch of (M,C,L) neg samples is passed through encoder
                    # [N,K,C,L]
                    neg_tensor_shape = triplet_batch['x_negs'].shape
                    M = neg_tensor_shape[0] * neg_tensor_shape[1]
                    K = neg_tensor_shape[1]
                    C = neg_tensor_shape[2]
                    L = neg_tensor_shape[3]
                    triplet_batch['x_negs'] = triplet_batch['x_negs'].reshape(
                        M, C, L)

                    K_slice = 2
                    loss_neg_scalar = 0
                    for i in range(0, K, K_slice):

                        x_negs_slice = triplet_batch['x_negs'][i:i +
                                                               K_slice].to(device)
                        out_neg = encoder(x_negs_slice)
                        del x_negs_slice

                        # Update weights w.r.t to a single negative sample (1 out of K)
                        optimizer.zero_grad()
                        loss_neg = triplet_loss_neg(out_ref_batch,
                                                    out_neg)
                        loss_neg.backward(retain_graph=False)
                        optimizer.step()

                        # Save scalar loss term and free memory that won't be used again
                        print(
                            f'loss for neg samples {i}-{i+K_slice}: {loss_neg.item()}')
                        loss_neg_scalar += loss_neg.item()
                        del out_neg
                        del loss_neg
                        torch.cuda.empty_cache()

                    del out_ref_batch

                    # print statistics
                    print(
                        f'total neg loss: {loss_neg_scalar}, pos loss: {loss_pos_scalar}')
                    loss = loss_pos_scalar + loss_neg_scalar
                    if i_batch % 1 == 0:    # print every mini-batch
                        print('[%d, %5d] loss: %.3f' %
                              (epoch, i_batch, loss))

                    step += 1

                    # Save model checkpoint every 50 epochs
                    if step % 50 == 0:
                        print(f'Saving encoder checkpoint at epoch {epoch}...')
                        encoder_path = './IHEPC_epoch_' + str(epoch) + \
                            '_dim_' + str(args.in_channels) + \
                            '_wsize_' + str(args.window_size) + '.pth'

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': encoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                        }, encoder_path)
                        print('Encoder saved!')

            if args.window_size is not None:  # Encoder has been trained on IHEPC
                encoder_path = './IHEPC_epoch_' + str(epoch) + \
                    '_dim_' + str(args.in_channels) + \
                    '_wsize_' + str(args.window_size) + '.pth'

                torch.save(encoder.state_dict(), encoder_path)

            elif multivariate_data:
                test_ds = TimeSeriesDatasetMultivariate(get_dataset_path(dsp, split='test'),
                                                        K=k_neg)
            else:
                test_ds = TimeSeriesDatasetUnivariate(get_dataset_path(dsp, split='test'),
                                                      K=k_neg)

            if args.window_size is None:  # Generate encodings for IHEPC separately after training
                dataset_name = dsp.rsplit(sep='/', maxsplit=1)[1]
                train_path = make_save_path(
                    dataset_name, args.features_path, k_neg=k_neg, split='train')
                test_path = make_save_path(
                    dataset_name, args.features_path, k_neg=k_neg, split='test')

                print('Encoding and saving each series from dataset...\n')
                print('(Saving encoded training set to {})\n'.format(train_path))
                encode_and_save(train_ds, encoder, train_path, device)
                print('(Saving encoded test set to {})\n'.format(test_path))
                encode_and_save(test_ds, encoder, test_path, device)
