from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import SEMEION
from torchvision.transforms import transforms


def train_model(model: nn.Module, optimizer, crit, data_loader: DataLoader, device: str = "cuda:0") -> List[float]:
    losses = []
    if not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        y_hat = y_hat.squeeze()
        loss = crit(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def val_model(model: nn.Module, data_loader: DataLoader, device: str = "cuda:0") -> float:
    if not torch.cuda.is_available():
        device = "cpu"
    with torch.no_grad():
        total = 0
        correct = 0
        model.to(device)
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            y_hat = y_hat.squeeze()
            y_hat = y_hat.softmax(dim=1)
            probs, idxes = y_hat.max(dim=1)
            total += y.size(0)
            correct += (idxes == y).sum().item()
    return correct / total


def get_semeion_data(root: str = "./data", image_size: int = 28, batch_size: int = 32, channel: int = 1,
                     train_percent: float = 0.8) -> Tuple[
    DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=2),
        # 构造一个通道的数据
        transforms.Lambda(lambda x: x.repeat(channel, 1, 1))
    ])
    data = SEMEION(root=root, download=False, transform=transform)

    # 构建dataset 和 dataloader
    train_size = int(len(data) * train_percent)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def enhance_data(data: SEMEION, random_times: int = 5, image_size: int = 28, batch_size: int = 32,
                 train_percent: float = 0.8) -> [DataLoader,
                                                 DataLoader]:
    crop_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=2),
    ])

    x = []
    y = []
    for image, label in data:
        for i in range(random_times):
            x.append(crop_transform(image))
            y.append(label)
    x = torch.stack(x)
    y = torch.tensor(y)
    data = torch.utils.data.TensorDataset(x, y)
    train_size = int(len(data) * train_percent)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
