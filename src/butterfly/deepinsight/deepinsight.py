import torch.nn
import collections
import numpy as np


ConvLayer = collections.namedtuple("ConvLayer", ["pad", "conv", "batch", "relu", "pool"])


def init_args(args, **kwargs):
    if args is None:
        return kwargs
    else:
        return {**kwargs, **args}


def init_convlayer(conv_args=None, batch_args=None, relu_args=None, pool_args=None):
    s = conv_args["kernel_size"]
    if s % 2 == 0:
        padding = (s // 2 - 1, s // 2, s // 2 - 1, s // 2)
    else:
        padding = (s // 2, s // 2, s // 2, s // 2)

    layer = ConvLayer(
        torch.nn.ZeroPad2d(padding=padding),
        torch.nn.Conv2d(**init_args(
            conv_args)),
        torch.nn.BatchNorm2d(**init_args(
            batch_args, num_features=1)),
        torch.nn.ReLU(**init_args(
            relu_args)),
        torch.nn.MaxPool2d(**init_args(
            pool_args, kernel_size=2, stride=2, padding=0)))
    return layer


def forward_convlayer(x, conv_layer, pool=True):
    x = conv_layer.pad(x)
    x = conv_layer.conv(x)
    x = conv_layer.batch(x)
    x = conv_layer.relu(x)
    if pool:
        x = conv_layer.pool(x)
    return x


class DeepInsight(torch.nn.Module):

    def __init__(self, input_dim, kernel_size1=1, kernel_size2=2, n_initial_filters=1, n_classes=2):
        super(DeepInsight, self).__init__()

        self.conv1_1 = init_convlayer(
            conv_args=dict(
                in_channels=1, out_channels=1*n_initial_filters,
                kernel_size=kernel_size1, stride=1),
            batch_args=dict(num_features=1*n_initial_filters)
        )
        self.conv1_2 = init_convlayer(
            conv_args=dict(
                in_channels=1*n_initial_filters, out_channels=2*n_initial_filters,
                kernel_size=kernel_size1, stride=1),
            batch_args=dict(num_features=2*n_initial_filters)
        )
        self.conv1_3 = init_convlayer(
            conv_args=dict(
                in_channels=2*n_initial_filters, out_channels=4*n_initial_filters,
                kernel_size=kernel_size1, stride=1),
            batch_args=dict(num_features=4*n_initial_filters)
        )
        self.conv1_4 = init_convlayer(
            conv_args=dict(
                in_channels=4*n_initial_filters, out_channels=8*n_initial_filters,
                kernel_size=kernel_size1, stride=1),
            batch_args=dict(num_features=8*n_initial_filters)
        )

        self.conv2_1 = init_convlayer(
            conv_args=dict(
                in_channels=1, out_channels=1 * n_initial_filters,
                kernel_size=kernel_size2, stride=1),
            batch_args=dict(num_features=1*n_initial_filters)
        )
        self.conv2_2 = init_convlayer(
            conv_args=dict(
                in_channels=1*n_initial_filters, out_channels=2 * n_initial_filters,
                kernel_size=kernel_size2, stride=1),
            batch_args=dict(num_features=2*n_initial_filters)
        )
        self.conv2_3 = init_convlayer(
            conv_args=dict(
                in_channels=2*n_initial_filters, out_channels=4 * n_initial_filters,
                kernel_size=kernel_size2, stride=1),
            batch_args=dict(num_features=4*n_initial_filters)
        )
        self.conv2_4 = init_convlayer(
            conv_args=dict(
                in_channels=4*n_initial_filters, out_channels=8 * n_initial_filters,
                kernel_size=kernel_size2, stride=1),
            batch_args=dict(num_features=8*n_initial_filters)
        )

        self.avg = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(
            int(8 * ((input_dim[0] / 8) // 2) * ((input_dim[1] / 8) // 2)),
            n_classes)
        self.softmax = torch.nn.Softmax(dim=n_classes)

    def forward(self, x):

        x1 = forward_convlayer(x, self.conv1_1)
        x1 = forward_convlayer(x1, self.conv1_2)
        x1 = forward_convlayer(x1, self.conv1_3)
        x1 = forward_convlayer(x1, self.conv1_4, pool=False)

        x2 = forward_convlayer(x, self.conv2_1)
        x2 = forward_convlayer(x2, self.conv2_2)
        x2 = forward_convlayer(x2, self.conv2_3)
        x2 = forward_convlayer(x2, self.conv2_4, pool=False)

        x3 = x1 + x2
        x3 = self.avg(x3)  # TODO: not padded!!! Might loose information here
        x3 = self.fc(x3.reshape(x.shape[0], x.shape[1], -1))
        x3 = self.softmax(x3)

        return x3


if __name__ == '__main__':
    X = torch.tensor(np.random.random((40, 40))).reshape(1, 1, 40, 40).float()

    m = DeepInsight(input_dim=X.shape[2:])
    m.forward(X)

    # lp = torch.nn.ZeroPad2d(padding=(1,0,0,0))
    # lc = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    # lb = torch.nn.BatchNorm2d(1)
    # x = lb.forward(lc.forward(lp.forward(x)))
    # print(x.shape)
