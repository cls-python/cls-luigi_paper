from torch.utils.data import Dataset
from torch import FloatTensor


class TorchDataset(Dataset):
    def __init__(self, X, y, sols, objs):
        self.X = X
        self.y = y
        self.sols = sols
        self.objs = objs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            FloatTensor(self.X[index]),
            FloatTensor(self.y[index]),
            FloatTensor(self.sols[index]),
            FloatTensor(self.objs[index]),
        )
