import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from datetime import datetime
import time
import glob
from data_utils import make_save_path


class LinearRegressor(torch.nn.Module):
    def __init__(self, dim_in=80, dim_out=1):
        super().__init__()

        self.linear = torch.nn.Linear(
            in_features=dim_in, out_features=dim_out).double()

    def forward(self, encoding):
        return self.linear(encoding)


class IHEPC_Ecodings(Dataset):
    def __init__(self,
                 encodings_path,
                 day_or_quarter,
                 train_or_test,
                 max_i_batch):

        # read and concatenate all the batches to create the whole set
        self.labels = np.load(encodings_path + '/labels.npy')
        self.all_encodings = np.zeros((len(self.labels),80,1), dtype=np.double)
        for batch_index in range(max_i_batch + 1):  # including max_i_batch
            if batch_index % 10000 == 0:
                print(f'Concatenating batch {batch_index}')

            encodings_batch = np.load(encodings_path + f'/i_batch_{batch_index}.npy')
            self.all_encodings[batch_index:batch_index+len(encodings_batch)] = encodings_batch
            del encodings_batch

        self.all_encodings = torch.tensor(self.all_encodings)

    def __len__(self):
        return self.all_encodings.shape[0]

    def __getitem__(self, idx):
        return {'sample': self.all_encodings[idx], 'label': self.labels[idx]}


def train_regressor(regressor,
                    dataloader,
                    day_or_quarter,
                    device,
                    max_optim_step,
                    args):

    optimizer = torch.optim.Adam(regressor.parameters(),
                                 lr=args.learning_rate,
                                 betas=(args.beta_1, args.beta_2))
    loss_MSE = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        for i_batch, batch in enumerate(dataloader):

            enc_batch = batch['sample'].to(device)
            lbl_batch = batch['label'].to(device)

            # Pass mini-batch through regressor
            # (80, 1)  => (80)  because linear regressor expects that
            out_batch = regressor(enc_batch.squeeze(2))

            # Compute loss and update weights
            optimizer.zero_grad()
            loss = loss_MSE(out_batch, lbl_batch)
            loss.backward()
            optimizer.step()

            # Print loss
            if i_batch % 100 == 0:
                print(f'[{epoch},{i_batch}] loss: {loss.item()}')

            # Primitive early stopping. TODO: revise when running on real data
            if i_batch > max_optim_step:
                break

    # Create a timestamp to identify the trained linear regressor
    ts = datetime.timestamp(datetime.now())
    ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    # Construct path to where model will be saved (create dir if non-existent)
    if not os.path.exists(args.regressors_path):
        os.makedirs(args.regressors_path)
        print(f'Models path{args.regressors_path} created...')
    else:
        print(f'Models path{args.regressors_path} already exists...')

    # Save model
    model_name = f'optim_{max_optim_step}.pth'
    torch.save(regressor.state_dict(), os.path.join(
        args.regressors_path, model_name))
    print(
        f'Saved regressor to path: {os.path.join(args.regressors_path,model_name)}')


def test_regressor(regressor,
                   dataloader,
                   day_or_quarter,
                   regressors_path,
                   device):

    MSELoss = torch.nn.MSELoss()
    total_loss = 0
    print('Computing test loss')
    for i_b, btch in enumerate(dataloader):
        test_encoding = btch['sample'].to(device)
        true_label = btch['label'].to(device)
        # (80, 1)  => (80)  because linear regressor expects that
        regress_val = regressor(test_encoding.squeeze(2))
        loss = MSELoss(regress_val, true_label).item()
        if i_b % 100 == 0:
            print(f'Loss at batch {i_b}: {loss}')
        total_loss += loss

    print('average loss:', total_loss/len(dataloader))


def parse_args():
    """ Parse command line arguments. Args with default=None are mandatory"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--encodings_path', type=str,
                        default=None)
    parser.add_argument('--regressors_path', type=str,
                        default=None)
    parser.add_argument('--day_or_quarter', type=str,
                        default=None)
    parser.add_argument('--train_or_test', type=str,
                        default=None)
    parser.add_argument('--max_optim_step', type=int,
                        default=None)
    parser.add_argument('--max_i_batch', type=int,
                        default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)

    args = parser.parse_args()
    if args.encodings_path is None or \
            args.regressors_path is None or \
            args.day_or_quarter is None or \
            args.train_or_test is None or \
            args.max_optim_step is None or \
            args.max_optim_step is None:
        raise ValueError('Not enough arguments specified!')

    return args


if __name__ == "__main__":
    args = parse_args()

    # Determine whether to run on CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_kwargs = {'batch_size': 10,
                     'shuffle': True,
                     'num_workers': 4 if device == 'gpu' else 0,
                     'pin_memory':  device == 'gpu'
                     }

    print(f'Instantiating DataLoader...')
    data_loader = DataLoader(IHEPC_Ecodings(args.encodings_path,
                                            args.day_or_quarter,
                                            args.train_or_test,
                                            args.max_i_batch),
                             **loader_kwargs)

    print(f'Finished instantiating DataLoader')
    regressor = LinearRegressor().double()
    start = time.time()
    if args.train_or_test == 'train':
        regressor.to(device)
        train_regressor(regressor,
                        data_loader,
                        args.day_or_quarter,
                        device,
                        args.max_optim_step,
                        args)
    elif args.train_or_test == 'test':
        regressor.load_state_dict(torch.load(args.regressors_path))
        regressor.eval()  # Set to inference mode
        regressor.to(device)
        test_regressor(regressor,
                       data_loader,
                       args.day_or_quarter,
                       args.regressors_path,
                       device)
    else:
        raise ValueError(
            "--train_or_test must be specified as 'train' or 'test'")
    elapsed = time.time() - start
    print(f'took {elapsed} seconds')
