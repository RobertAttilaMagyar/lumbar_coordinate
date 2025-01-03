from torch.utils.data import Dataset
import torchvision.transforms.v2 as tt
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

preprocess = tt.Compose([
    tt.Grayscale(),
    tt.CenterCrop(256),
    tt.ToTensor(),
    tt.Normalize(mean=[0.], std = [1.])
])

train_preprocess = tt.Compose(
    [
        tt.Grayscale(),
        tt.ToTensor(),
        tt.Normalize(mean=[0.], std = [1.]),
        tt.GaussianBlur(3, sigma=(0.1, 2.0))
    ]
)

class getData(Dataset):
    def __init__(self, base_path: str|Path):
        super().__init__()
        if isinstance(base_path, str):
            base_path = Path(base_path)
        base_path = base_path.resolve()

        df = pd.read_csv(base_path/"coords_pretrain.csv")

        df["img_paths"] = [(base_path/"data"/f"processed_{row['source']}_jpgs"/row['filename']) for _, row in df.iterrows()]
        df = df.groupby("img_paths").agg(lambda x: list(x))
        self._filenames = list(df.index.values)
        self._xs = list(df.relative_x.values)
        self._ys = list(df.relative_y.values)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        img = train_preprocess(Image.open(self._filenames[index]))
        xs = self._xs[index]
        ys = self._ys[index]
        target = torch.Tensor([xs, ys]).T
        return img, target
    
    def __len__(self):
        return len(self._filenames)
    
    def visualize(self, index):
        img, coords = self[index]
        plt.imshow(img.permute(1,2,0), cmap = "Greys")
        plt.scatter(256*coords[:,0], 256*coords[:, 1], color = "lime")

        

