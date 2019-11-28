import torch
import numpy as np
from data_utils import avg_batch_norm


def triplet_loss(out_ref, out_pos, out_negs):
    """Calculates batch triplet loss for a time series

    Args:
        out_ref (torch.DoubleTensor): [batch_size, output_channels, nb_timesteps]
                                     output from forward pass on reference sample
        out_pos (torch.DoubleTensor): [batch_size, output_channels, nb_timesteps]
                                     output from forward pass on positive sample
        out_negs (torch.DoubleTensor): [batch_size, K, output_channels, nb_timesteps]
                             tensor of outputs from forward passes on each negative sample

        Note that nb_timesteps is 1 because of the temporal squeeze of the encoder.

    Returns:
        double: average triplet loss for the whole batch.
    """
    batch_size = out_ref.shape[0]
    batch_loss = 0
    for b in range(batch_size):
        # [1,160] * [160,K] -> [1,K]
        neg_loss_term = torch.matmul(
            out_ref[b].transpose(0, 1), out_negs[b].squeeze(2).transpose(0, 1))
        neg_loss_term = torch.sum(
            torch.log(torch.sigmoid(-1 * neg_loss_term) + 1e-89))

        # [1,160] * [160,1] -> [1,1]
        pos_loss_term = torch.matmul(
            out_ref[b].transpose(0, 1), out_pos[b])
        pos_loss_term = -1 * torch.log(torch.sigmoid(pos_loss_term) + 1e-89)

        sample_loss = pos_loss_term - neg_loss_term
        batch_loss += sample_loss

    # Use the average loss across the batch. This seems to be most common
    # but can be discussed, see https://stats.stackexchange.com/questions/201452/...
    # ...is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-...
    # ...instead-of-the
    batch_loss /= batch_size

    return batch_loss


def triplet_loss_pos(out_ref, out_pos):
    """Calculates positive term of batch triplet loss for a time series

    Args:
        out_ref (torch.DoubleTensor): [batch_size, output_channels, nb_timesteps]
                                     output from forward pass on reference sample
        out_pos (torch.DoubleTensor): [batch_size, output_channels, nb_timesteps]
                                     output from forward pass on positive sample
    Returns:
        double: the positive term of the triplet loss for the whole batch.
    """
    batch_size = out_ref.shape[0]
    pos_batch_loss = 0
    for b in range(batch_size):
        # [1,160] * [160,1] -> [1,1]
        pos_loss_term = torch.matmul(
            out_ref[b].transpose(0, 1), out_pos[b])
        pos_batch_loss -= torch.log(torch.sigmoid(pos_loss_term) + 1e-89)

    return pos_batch_loss / batch_size


def triplet_loss_neg(out_ref, out_negs):
    """Calculates negative term of batch triplet loss for a time series

    Args:
        out_ref (torch.DoubleTensor): [batch_size, output_channels, nb_timesteps]
                                     output from forward pass on reference sample
        out_negs (torch.DoubleTensor): [batch_size, K, output_channels, nb_timesteps]
                             tensor of outputs from forward passes on each negative sample
    Returns:
        double: average triplet loss for the whole batch.
    """
    neg_batch_loss = 0
    K = out_negs.shape[0]  # out_negs: [K,80,C] out_ref: [1,80,C]
    # print(f'out_ref shape: {out_ref.shape}')
    # print(f'out_negs shape: {out_negs.shape}')
    # input()
    # [1,80] * [80,K] -> [1,K]
    neg_loss_term = torch.matmul(out_ref.squeeze(2),
                                 out_negs.squeeze(2).transpose(0, 1))
    neg_batch_loss -= torch.sum(
        torch.log(torch.sigmoid(-1 * neg_loss_term) + 1e-89))

    return neg_batch_loss / K


if __name__ == "__main__":
    X = torch.rand(4, 1, 4)
    out_ref = X[0, 0, :]
    out_pos = X[1, 0, :]
    out_negs = [X[2, 0, :], X[2, 0, :]]
    print(triplet_loss(out_ref, out_pos, out_negs))
