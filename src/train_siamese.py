import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset

import numpy as np
from time import time

from models.siamese import SiameseNet
from datetime import datetime
from tools import read


class RandomPairDataset(Dataset):
    def __init__(self, length, wins, losses):
        self.length = length
        self.wins = wins
        self.losses = losses

    def __getitem__(self, _):
        rand_win = self.wins[np.random.randint(0, self.wins.shape[0])]
        rand_loss = self.losses[np.random.randint(0, self.losses.shape[0])]

        # rand_win = self.wins[0]
        # rand_loss = self.losses[1234]

        # label = self.to_torch_tensor(np.array([1, 0]))
        # rand_win = self.to_torch_tensor(rand_win)
        # rand_loss = self.to_torch_tensor(rand_loss)
        # return ((rand_win, rand_loss), label)

        order = np.random.randint(0, 2)
        if order == 0:
            label = self.to_torch_tensor(np.array([1, 0]))
            rand_win = self.to_torch_tensor(rand_win)
            rand_loss = self.to_torch_tensor(rand_loss)
            return ((rand_win, rand_loss), label)
        else:
            label = self.to_torch_tensor(np.array([0, 1]))
            rand_win = self.to_torch_tensor(rand_win)
            rand_loss = self.to_torch_tensor(rand_loss)
            return ((rand_loss, rand_win), label)

    def __len__(self):
        return self.length

    def to_torch_tensor(self, array):
        return torch.from_numpy(array).type(torch.FloatTensor)


def train(dataloader, model, loss_function, optimizer, device):
    size = len(dataloader.dataset)
    correct = 0
    train_loss = 0

    for X, y in dataloader:
        x1, x2 = X
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        pred = model(x1, x2)
        loss = loss_function(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = F.softmax(pred, dim=1)
        pred = (pred > 0.5).float()
        correct += (pred == y).sum() / 2

    train_loss /= size
    return train_loss, correct


def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    correct = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            x1, x2 = X
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            pred = model(x1, x2)
            test_loss += loss_function(pred, y).item()

            pred = F.softmax(pred, dim=1)
            pred = (pred > 0.5).float()
            correct += (pred == y).sum() / 2

    test_loss /= size
    return test_loss, correct


def load_data(*filenames):
    features = None
    labels = None

    for name in filenames:
        data = read(name)

        file_features = np.array([np.append(np.unpackbits(state), i % 2)
                                  for game in data for i, state in enumerate(game[0])])

        file_labels = np.array([game[1] for game in data for _ in game[0]])

        if features is None:
            features = file_features
            labels = file_labels
        else:
            features = np.append(features, file_features, axis=0)
            labels = np.append(labels, file_labels, axis=0)

    return features, labels


if __name__ == '__main__':
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.set_default_dtype(torch.float32)

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}')

    # hyperparameters
    learning_rate = 0.01
    lr_decay_gamma = 0.99
    batch_size = 256
    epochs = 100
    dataset_size = 100000

    model = SiameseNet().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adagrad(model.parameters())
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer, gamma=lr_decay_gamma)

    train_loss_history = []
    test_loss_history = []

    wins_features, wins_labels = load_data(
        "./dataset-white-658-2022-01-22--15-00-37", "dataset-1-white-656-2022-01-23--15-45-03", "dataset-2-white-672-2022-01-23--17-05-27", "dataset-3-white-664-2022-01-23--18-27-25")
    losses_features, losses_labels = load_data(
        "./dataset-black-342-2022-01-22--15-00-37", "dataset-1-black-344-2022-01-23--15-45-03", "dataset-2-black-328-2022-01-23--17-05-27", "dataset-3-black-336-2022-01-23--18-27-25")

    total_wins = wins_features.shape[0]
    total_losses = losses_features.shape[0]

    train_dataloader = DataLoader(
        RandomPairDataset(dataset_size, wins_features[:-20000], losses_features[:-20000]), batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(RandomPairDataset(
        dataset_size, wins_features[-20000:], losses_features[-20000:]), batch_size=batch_size, shuffle=True)

    start = time()

    for t in range(epochs):

        train_loss, train_correct = train(train_dataloader, model, loss_function,
                                          optimizer, device)
        train_loss_history.append(train_loss)

        test_loss, test_correct = test(test_dataloader, model, loss_function)
        test_loss_history.append(test_loss)

        print(f"Epoch {t+1}")
        print(f'   Train: {train_loss}')
        print(f'   Test:  {test_loss}')
        print(
            f'   Train Acc: [{train_correct}/{dataset_size}] ({train_correct * 100 / dataset_size}%)')
        print(
            f'   Test Acc:  [{test_correct}/{dataset_size}] ({test_correct * 100 / dataset_size}%)')

        # scheduler.step()

    # print(count_dict)

    # torch.save(model.state_dict(), './model' +
    #            datetime.today().strftime('%Y-%m-%d--%H-%M-%S'))
    plt.plot(train_loss_history)
    plt.plot(test_loss_history)
    plt.show()
    # 332 sec
    # 133 sec
    print("Done!")
    print(f"Elapsed: {time() - start}")
