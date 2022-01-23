import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

import numpy as np
from time import time

from models.autoencoder import Autoencoder
from datetime import datetime
from tools import get_date, read


def train(dataloader, model, loss_function, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_function(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= size
    print(f"Avg train loss: {train_loss:>8f} \n")

    # if batch % 100 == 0:
    #     loss, current = loss.item(), batch * len(X)
    #     print(f"loss: {loss:>20f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_function(pred, y).item()

    test_loss /= size
    print(f"Avg test loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    torch.manual_seed(0)

    data = read("./dataset-1")

    features = np.array([np.append(np.unpackbits(state), i % 2)
                        for game in data for i, state in enumerate(game[0])])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}')

    model = Autoencoder().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    dataset_size = features.shape[0]
    train_size = int(dataset_size * 0.95)
    test_size = dataset_size - train_size
    batch_size = 256

    train_tensor = torch.Tensor(features[:train_size])
    test_tensor = torch.Tensor(features[train_size:])

    train_dataloader = DataLoader(TensorDataset(
        train_tensor, train_tensor), batch_size=batch_size)
    test_dataloader = DataLoader(TensorDataset(
        test_tensor, test_tensor), batch_size=64)

    epochs = 200
    start = time()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer, device)
        scheduler.step()
        test(test_dataloader, model, loss_function)

    torch.save(model.state_dict(), './model' + get_date())

    print("Done!")
    print(f"Elapsed: {time() - start}")
