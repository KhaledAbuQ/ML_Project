import kagglehub

# Download latest version
path = kagglehub.dataset_download("defileroff/comic-faces-paired-synthetic")

print("Path to dataset files:", path)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.plugins.environments import ClusterEnvironment
from PIL import Image
import torch.optim as optim


# ----------------- MODEL -----------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ----------------- DATASET -----------------
class PairedImageDataset(Dataset):
    def __init__(self, face_paths, comic_paths, transform=None):
        self.face_paths = face_paths
        self.comic_paths = comic_paths
        self.transform = transform

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_image = Image.open(self.face_paths[idx]).convert("RGB")
        comic_image = Image.open(self.comic_paths[idx]).convert("RGB")
        if self.transform:
            face_image = self.transform(face_image)
            comic_image = self.transform(comic_image)
        return face_image, comic_image


# ----------------- LIGHTNING MODULE -----------------
class LitUNet(pl.LightningModule):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.model = UNet()
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        face, comic = batch
        output = self.model(face)
        loss = self.loss_fn(output, comic)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)




# class MyClusterEnvironment(ClusterEnvironment):
#     @property
#     def creates_processes_externally(self) -> bool:
#         """Return True if the cluster is managed (you don't launch processes yourself)"""
#         return True

#     def world_size(self) -> int:
#         return int("2")

#     def global_rank(self) -> int:
#         return int("2")

#     def local_rank(self) -> int:
#         return int("0")

#     def node_rank(self) -> int:
#         return int("2")

#     def main_address(self) -> str:
#         return "192.168.1.10"

#     def main_port(self) -> int:
#         return int("12355")


# ----------------- DATA & TRAINER -----------------
if __name__ == "__main__":
    os.environ["NCCL_DEBUG"]="INFO"
    os.environ["NCCL_SOCKET_IFNAME"]="eno1"
    path = 'face2comics_v1.0.0_by_Sxela/'
    comics_path = path + 'comics/'
    face_path = path + 'face/'

    comics = [os.path.join(comics_path, f) for f in os.listdir(comics_path)]
    face = [os.path.join(face_path, f) for f in os.listdir(face_path)]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = PairedImageDataset(face[:8000], comics[:8000], transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = LitUNet(lr=5e-5)

    trainer = pl.Trainer(
    accelerator="gpu",
    num_nodes=2,
    strategy="ddp",
    devices="auto",
    max_epochs=50,
    # plugins=[MyClusterEnvironment()]
)

    trainer.fit(model, dataloader)