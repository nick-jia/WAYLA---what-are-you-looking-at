import torch.utils.data as data
from torch import from_numpy

class SaliencyDataset(data.Dataset):

    def __init__(self, x, y, transform):
        self.X = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.transform(self.X[index])
        if self.y is not None:
            sample = (image, self.y[index])
        else:
            sample = image
        return sample