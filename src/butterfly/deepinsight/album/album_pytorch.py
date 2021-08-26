import torch
import torch.utils.data


class AlbumDataset(torch.utils.data.Dataset):

    def __init__(self, album, labels):
        self.album = torch.tensor(album).float()
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, item):
        return self.album[item], self.labels[item]

    def __len__(self):
        return self.album.shape[0]
