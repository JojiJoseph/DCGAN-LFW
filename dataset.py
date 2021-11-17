import torch
import torch.nn as nn
import torchvision
import os
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms.transforms import Normalize, ToPILImage


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        dataset_folder = "lfw-deepfunneled"
        self.filelist = []
        for root, folders, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith(".jpg"):
                    path = os.path.join(root, file)
                    self.filelist.append(path)
    def __len__(self):
        # return 2000
        return len(self.filelist)
    def __getitem__(self, idx):
        path = self.filelist[idx]
        img = Image.open(path).resize((64,64))
        img = ToTensor()(img)
        img = Normalize(0.5,0.5)(img)
        # print(img.min(), img.max())
        return img
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy
    dataset = LFWDataset()
    print(len(dataset))
    print(dataset[1].shape)
    print(dataset[2].max(), dataset[2].min())
    img = ToPILImage()(dataset[1]/2+0.5)
    plt.imshow(img)
    plt.show()