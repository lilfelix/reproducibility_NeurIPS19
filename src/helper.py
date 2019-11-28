"""
    Helper modules used in other classes.
"""
import torch


class RightTruncate(torch.nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, causal_out):
        """
        Truncates the right padding_size units from the causal CNN output. This is used because using the padding
        option of the Conv1D module also adds the padding to the right side, which needs to be removed then.
        :param causal_out: the output of the causal CNN
        :return: the right truncated output
        """
        return causal_out[:, :, :-self.padding_size]  # shape (N, C, L)
