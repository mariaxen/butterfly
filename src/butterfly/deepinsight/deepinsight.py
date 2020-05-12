import torch.nn
import collections
import numpy as np


def init_args(args, **kwargs):
    if args is None:
        return kwargs
    else:
        return {**kwargs, **args}


def init_convlayer(conv_args=None, batch_args=None, relu_args=None, pool_args=None, pool=True):
    s = conv_args["kernel_size"]
    if s % 2 == 0:
        padding = (s // 2 - 1, s // 2, s // 2 - 1, s // 2)
    else:
        padding = (s // 2, s // 2, s // 2, s // 2)

    layer = torch.nn.Sequential()
    layer.add_module("pad", torch.nn.ZeroPad2d(padding=padding))
    layer.add_module("conv", torch.nn.Conv2d(**init_args(conv_args)))
    layer.add_module("batch", torch.nn.BatchNorm2d(**init_args(batch_args, num_features=1)))
    layer.add_module("relu", torch.nn.ReLU(**init_args(relu_args)))
    if pool:
        layer.add_module("pool", torch.nn.MaxPool2d(**init_args(
                pool_args, kernel_size=2, stride=2, padding=0)))
    return layer


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
            batch_args=dict(num_features=8*n_initial_filters),
            pool=False
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
            batch_args=dict(num_features=8*n_initial_filters),
            pool=False
        )

        self.avg = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(
            int(8 * ((input_dim[0] / 8) // 2) * ((input_dim[1] / 8) // 2)),
            n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)

        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)

        x3 = x1 + x2
        x3 = self.avg(x3)  # TODO: not padded!!! Might loose information here
        x3 = self.fc(x3.reshape(x.shape[0], -1))
        x3 = self.softmax(x3)

        return x3


if __name__ == '__main__':

    X = torch.tensor(np.random.random((10, 40, 40))).reshape(-1, 1, 40, 40).float()
    y = np.random.choice([False, True], 10).reshape(-1, 1)
    y = torch.tensor(np.concatenate([y], axis=1)).long().flatten()
    print(y)

    m = DeepInsight(input_dim=X.shape[2:])

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(X)
    ss.transform(X2)


    import h5py
    file = h5py.File('../../../external/DeepInsight/Data/dataset2.mat', "r")
    print(file["dset"]["Xtest"][...].shape)

    import torch.optim as optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(zip(X, y)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = X, y
            print(labels.dtype)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = m(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # lp = torch.nn.ZeroPad2d(padding=(1,0,0,0))
    # lc = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    # lb = torch.nn.BatchNorm2d(1)
    # x = lb.forward(lc.forward(lp.forward(X)))
    # print(x.shape)
