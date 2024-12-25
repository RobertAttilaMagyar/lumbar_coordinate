from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import torch

import numpy as np

from typing import Callable


class Trainer:
    def __init__(self, train_ds: Dataset, batch_size: int = 16, val_ds: Dataset = None):
        self._train_epoch_size: int = len(train_ds)
        self._val_epoch_size: int = len(val_ds)
        self._dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self._val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True) if val_ds is not None else None

    @staticmethod
    def _loss_function(outputs: torch.Tensor, coords: torch.Tensor):
        return ((coords - outputs) ** 2).sum(dim=2).sum(dim=1).mean()

    def _validate(self, model: nn.Module, loss_fn: Callable)->float:
        model.eval()
        losses = []
        with torch.no_grad():
            for vimg, vcoords in tqdm(self._val_loader):
                vimg = vimg.to(self._device)
                vcoords = vcoords.to(self._device)
                vouts = model(vimg)
                vloss = loss_fn(vouts, vcoords)
                losses.append(vloss.item())
        print(f"Average validation loss: {(mean_loss := np.mean(losses))}")

        return mean_loss
    

    def _one_epoch(self, model: nn.Module, optimizer, validate=False)->tuple[float, float]:
        model.to(self._device)
        model.train()
        losses = []
        for img, coords in tqdm(self._dataloader, leave=False):
            img = img.to(self._device)
            coords = coords.to(self._device)

            optimizer.zero_grad()

            outputs = model(img)
            loss = self._loss_function(outputs, coords)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Average training loss in epoch: {(mean_train_loss := np.mean(losses))}")
        if validate:
            mean_val_loss = self._validate(model, self._loss_function)
        
        return mean_train_loss, mean_val_loss
    
    def _adjust_learning_rate(self, optimizer, epoch):
        if epoch > 1 and epoch%5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /=2

    def train(self, model: nn.Module, num_epochs=30, validate=True, lr=0.001):
        model.to(self._device)
        train_history: list[float] = []
        val_history: list[float] = []
        best_validation_loss = torch.inf

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = 10)
        for epoch in (pbar:=tqdm(range(num_epochs))):
            pbar.set_description(f"Currently at {epoch}, lr: {scheduler.get_last_lr()[0]}")
            # self._adjust_learning_rate(optimizer, epoch)
            mtl, mvl = self._one_epoch(model, optimizer=optimizer, validate=validate)
            
            train_history.append(mtl)
            val_history.append(mvl)

            if mvl < best_validation_loss:
                best_validation_loss = mvl
                torch.save(model.state_dict(), "best-model.pt")

            scheduler.step()
        
        if self._val_loader is not None:
            self._validate(model, self._loss_function)

        return train_history, val_history



