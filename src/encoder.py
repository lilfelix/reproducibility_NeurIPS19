import torch
from helper import RightTruncate


class CausalCNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, layer, up_or_down_sample=False):
        """
        Each of the causal convolution layers as depicted in Fig. 2 (b) of the paper.
        :param in_channels: channel size of the input to the causal layer
        :param out_channels: channel size of the output of the causal layer
        :param kernel_size: -
        :param dilation: -
        :param layer: the position of the layer ('first', 'hidden', or 'last), which matters for determining the
        'in_channels' and 'out_channels' parameters of the convolutions in the causal layer.
        :param up_or_down_sample: if up- or down-sampling should be done in the residual bock. It is necessary for the
        very first and last layers where 'in_channels' is different from 'out_channels'. In other cases the input should
        be directly added to the output of the layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up_or_down_sample = up_or_down_sample
        self.layer = layer

        # for (kernel_size -1) times, we need to jump back with length equal to dilation
        padding_length = (self.kernel_size - 1) * self.dilation
        # print("In [CausalCNNLayer]: padding_length is:", padding_length)

        """
        determining the input and output channels of the convolutions in the causal layer.
        Example:
            - first causal layer in_channels = 1, out_channels = 40: 1 -> conv1 -> 40, 40 -> conv2 -> 40
            - hidden causal layer in_channels = 40, out_channels = 40: 40 -> conv1 -> 40, 40 -> conv2 -> 40
            - last causal layer in_channels = 40, out_channels = 320: 40 -> conv1 -> 40, 40 -> conv2 -> 320
        """
        if self.layer is 'first' or self.layer is 'hidden':
            conv1_out_channels = self.out_channels
        else:  # self.layer is 'last'
            conv1_out_channels = self.in_channels

        # name and dim in weight norm????
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(in_channels=self.in_channels,
                                                           out_channels=conv1_out_channels,
                                                           kernel_size=self.kernel_size,
                                                           padding=padding_length,
                                                           dilation=self.dilation))
        trunc1 = RightTruncate(padding_size=padding_length)
        lrelu1 = torch.nn.LeakyReLU(negative_slope=0.01)

        # in_channel for this convolution should be equal to out_channel (output channel of the previous convolution)
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(in_channels=conv1_out_channels,
                                                           out_channels=self.out_channels,
                                                           kernel_size=self.kernel_size,
                                                           padding=padding_length,
                                                           dilation=self.dilation))
        trunc2 = RightTruncate(padding_size=padding_length)
        lrelu2 = torch.nn.LeakyReLU(negative_slope=0.01)

        # The sequential block consists of two weight normalized causal convolution with truncated output and leaky ReLU
        self.causal_block = torch.nn.Sequential(
            conv1, trunc1, lrelu1, conv2, trunc2, lrelu2)

        # The residual part in Fig. 2.b
        if self.up_or_down_sample:
            self.residual = torch.nn.Conv1d(
                in_channels=self.in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, series):
        """
        The forward operation of the the causal layer.
        :param series: the input to the causal layer.
        :return: the output of the causal layer.
        """
        causal_out = self.causal_block(series)
        if self.up_or_down_sample:
            # up- or down-sampled input to be added
            causal_out = causal_out + self.residual(series)
        else:
            causal_out += series  # the actual input to be added
        return causal_out


class Encoder(torch.nn.Module):
    def __init__(self,
                 n_layers,
                 hidden_out_channels,
                 kernel_size,
                 last_out_channels,
                 rep_length,
                 in_channels=1):
        """
        The encoder architecture as proposed in Fig. 2 of the paper.
        :param n_layers: number of causal convolutional layers
        :param hidden_out_channels: output/input channels of the hidden causal layers
        :param kernel_size: kernel size of each convolution
        :param last_out_channels: output channels size of the very last causal layer
        :param rep_length: the length of the representation (after max pooling and linear transformation)
        :param in_channels: number of input channels (dimensionality of a time series sample)

        The very first and last causal layer are constructed out of the for loop because their input and output channels are different from that the hidden layers. This difference is:
            - for the first one: the input channel is D (in case of D-dimensional time series)
            - for the last one, the output channel is determined by the user (through the 'last_out_channels' param)
        """
        super().__init__()
        layers = []  # layers in a Python list
        dilation = 1

        # the first convolution takes a 1-D input and outputs with the desired number of output channels
        first_causal = CausalCNNLayer(in_channels, out_channels=hidden_out_channels,
                                      kernel_size=kernel_size, dilation=dilation,
                                      layer='first', up_or_down_sample=True)
        layers += [first_causal]

        # hidden layers (all layers expect the very first and last layer)
        for i in range(n_layers - 2):
            dilation *= 2
            # the inner convolutions keep the channel dimension the same (as mentioned in the paper Appendix)
            layers += [CausalCNNLayer(in_channels=hidden_out_channels, out_channels=hidden_out_channels,
                                      kernel_size=kernel_size, dilation=dilation, layer='hidden')]

        # the very last casual layer
        dilation *= 2
        last_causal = CausalCNNLayer(in_channels=hidden_out_channels, out_channels=last_out_channels,
                                     kernel_size=kernel_size, dilation=dilation, layer='last', up_or_down_sample=True)
        layers += [last_causal]

        # all the causal layers stacked on top of each other
        self.causal_layers = torch.nn.ModuleList(
            layers)  # converting the list to a Module

        # after max_pooling, the linear transformation is to convert last_out_channels to rep_length (e.g. 320 -> 160)
        self.linear = torch.nn.Linear(
            in_features=last_out_channels, out_features=rep_length)

    def forward(self, series):
        """
        Performs the forward operation of the encoder.
        :param series: shape (N, C, L) - the (sub-)series to perform the operation on. Could be positive, negative, or reference samples. N is the batch size, C is the channels of the sequence, and L is the length.
        :return out: shape (N, rep_length, 1) - the output of the encoder (representation of the input) and N is the batch size

        Note on the linear transformation:
        to use nn.Linear() we need to bring the dimension that is changed (320 -> 160) to the last index by .permute()
        why to use .contiguous() is explained here: https://stackoverflow.com/questions/48915810/pytorch-contiguous
        """
        out = None
        # causal layers
        for i in range(len(self.causal_layers)):
            causal = self.causal_layers[i]
            out = causal(series) if i == 0 else causal(
                out)  # the first or hidden layers

        # global max pooling layer: shape [N, 320, seq_length] -> [N, 320, 1]
        # torch.max returns a tuple of max and argmax
        out = torch.max(out, dim=2, keepdim=True)[0]

        # linear transformation: shape [N, 320, 1] -> [N, 160, 1]
        # more info in the comment of the forward function (above)
        out = out.permute(0, 2, 1).contiguous()
        out = self.linear(out).permute(0, 2, 1).contiguous()
        return out
