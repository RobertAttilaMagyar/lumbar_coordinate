import argparse
from lumbar_coordinate.models.unet import UNet
from lumbar_coordinate.train import Trainer
from lumbar_coordinate.dataset import getData
from torch.utils.data import random_split
import torch
from dataclasses import dataclass

import matplotlib.pyplot as plt

torch.cuda.empty_cache()

def main(data_path, epochs):
    whole_ds = getData(data_path)
    train_ds, val_ds, test_ds = random_split(whole_ds, [0.8, 0.1, 0.1])
    trainer = Trainer(train_ds, batch_size=8, val_ds = val_ds)
    model = UNet()
    train_history, val_history = trainer.train(model, num_epochs=epochs)
    plt.figure()
    plt.plot(train_history, label = "Train losses")
    plt.plot(val_history, label = "Validation losses")
    plt.title("Loss history during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig("train_history_unet_cos_annealing_Tmax10.png")

    torch.save(model.state_dict(), "unet_cos_annealing_Tmax10.pt")



main("/home/ad.adasworks.com/attila.magyar/Desktop/lumbar_coordinate/data", epochs=50)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(add_help=True)
#     parser.add_argument("--data-path", type=str, required=True, help = "Path to lumbar coordinate dataset")
#     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
#     args = parser.parse_args()
