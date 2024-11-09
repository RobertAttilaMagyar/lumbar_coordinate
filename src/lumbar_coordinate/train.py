from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import torch

import numpy as np

from typing import Callable


class Trainer:
    def __init__(self, train_ds: Dataset, batch_size: int = 16, val_ds: Dataset = None):
        self._dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self._val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True) if val_ds is not None else None

    @staticmethod
    def _loss_function(outputs: torch.Tensor, coords: torch.Tensor):
        return ((coords - outputs) ** 2).sum(dim=2).sum(dim=1).mean()

    def _validate(self, model: nn.Module, loss_fn: Callable):
        model.eval()
        losses = []
        with torch.no_grad():
            for vimg, vcoords in tqdm(self._val_loader):
                vimg = vimg.to(self._device)
                vcoords = vcoords.to(self._device)
                vouts = model(vimg)
                vloss = loss_fn(vouts, vcoords)
                losses.append(vloss.item())
        print(f"Average validation loss: {np.mean(losses)}")

    def _one_epoch(self, model: nn.Module, optimizer, validate=False):
        model.to(self._device)
        model.train()
        losses = []
        for img, coords in tqdm(self._dataloader):
            img = img.to(self._device)
            coords = coords.to(self._device)

            optimizer.zero_grad()

            outputs = model(img)
            loss = self._loss_function(outputs, coords)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Average training loss in epoch: {np.mean(losses)}")
        if validate:
            self._validate(model, self._loss_function)

            pass

    def _adjust_learning_rate(self, optimizer, epoch):
        if epoch > 1 and epoch%5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /=2

    def train(self, model: nn.Module, num_epochs=10, validate=True, lr=0.001):
        model.to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self._adjust_learning_rate(optimizer, epoch)
            print(f"Epoch #{epoch}")
            self._one_epoch(model, optimizer=optimizer, validate=validate)
        
        if self._val_loader is not None:
            self._validate(model, self._loss_function)



