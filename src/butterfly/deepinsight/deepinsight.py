import torch.nn
import collections


ConvLayer = collections.namedtuple("ConvLayer", ["conv", "batch", "relu", "pool"])


def init_args(args, **kwargs):
    if args is None:
        return kwargs
    else:
        return {**kwargs, **args}


def init_convlayer(conv_args=None, batch_args=None, relu_args=None, pool_args=None):
    layer = ConvLayer(
        torch.nn.Conv2d(**init_args(
            conv_args, in_channels=3, out_channels=18, kernel_size=3, stride=1, padding=1)),
        torch.nn.BatchNorm1d(**init_args(
            batch_args, in_channels=3, out_channels=18)),
        torch.nn.ReLU(**init_args(
            relu_args)),
        torch.nn.MaxPool2d(**init_args(
            pool_args, kernel_size=2, stride=2, padding=0)))
    return layer


def forward_convlayer(x, conv_layer):
    x = conv_layer.conv(x)
    x = conv_layer.batch(x)
    x = conv_layer.relu(x)
    x = conv_layer.pool(x)
    return x


class DeepInsight(torch.nn.Module):

    def __init__(self):
        super(DeepInsight, self).__init__()

        self.conv1_1 = init_convlayer()
        self.conv1_2 = init_convlayer()
        self.conv1_3 = init_convlayer()
        self.conv1_4 = init_convlayer()

        self.conv2_1 = init_convlayer()
        self.conv2_2 = init_convlayer()
        self.conv2_3 = init_convlayer()
        self.conv2_4 = init_convlayer()

        self.avg = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(30, 20)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):

        x1 = forward_convlayer(x, self.conv1_1)
        x1 = forward_convlayer(x1, self.conv1_2)
        x1 = forward_convlayer(x1, self.conv1_3)
        x1 = forward_convlayer(x1, self.conv1_4)

        x2 = forward_convlayer(x, self.conv1_1)
        x2 = forward_convlayer(x2, self.conv1_2)
        x2 = forward_convlayer(x2, self.conv1_3)
        x2 = forward_convlayer(x2, self.conv1_4)

        x3 = x1 + x2
        x3 = self.avg(x3)
        x3 = self.fc(x3)
        x3 = self.softmax(x3)

        return x3
