from dataset import LFWDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Normalize, PILToTensor
from torchvision.transforms.transforms import ConvertImageDtype, ToPILImage
import os

from models import Discriminator, Generator
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = LFWDataset()

batch_size = 128
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=16)

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0, 0.002)
    if type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.normal_(m.weight.data, 0, 0.002)
    if type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    os.makedirs("./figs", exist_ok=True)
    gen = Generator().to(device)
    # with open("gen_dcgan.p")
    # gen.load_state_dict(torch.load("gen_dcgan.pt"))
    disc = Discriminator().to(device)
    # disc.load_state_dict(torch.load("disc_dcgan.pt"))

    gen.apply(init_weights)
    disc.apply(init_weights)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5,0.999))

    bce = torch.nn.MSELoss()#torch.nn.BCELoss()
    for epoch in range(10_000):
        print(f"Epoch {epoch+1} ...")
        for batch_id, real_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_size = real_batch.shape[0]
            real_batch = real_batch.to(device)
            
            # Train generator
            z = np.random.normal(size=(batch_size, 128))
            z = torch.from_numpy(z).float().to(device)
            y_g = gen(z)
            y_d = disc(y_g)
            loss = bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_gen.zero_grad()
            loss.backward()
            # print(loss.detach().cpu().numpy().item())
            opt_gen.step()

            # Train discriminator
            y_g = y_g.detach()
            y_d = disc(y_g)
            loss = 0.5*bce(y_d, torch.zeros((batch_size, 1)).to(device))
            y_d = disc(
                real_batch)
            loss += 0.5*bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_disc.zero_grad()
            loss.backward()
            opt_disc.step() 
        torch.save(disc.state_dict(), "./disc_dcgan.pt")
        torch.save(gen.state_dict(), "./gen_dcgan.pt")

        for i in range(10):
            plt.subplot(2, 5, i+1)
            z = np.random.normal(size=(1, 128))
            z = torch.from_numpy(z).float().to(device)
            cat = np.array([i])
            cat = torch.from_numpy(cat).long().to(device)
            gen.eval()
            with torch.no_grad():
                y = gen(z)[0]
            y = y.reshape(3, 64, 64)
            y = (y+1)/2
            y = y.permute(1,2,0)
            gen.train()
            plt.imshow(y.cpu().detach().numpy())
        plt.savefig(f"./figs/dcgan_fig{epoch}.png")
    plt.show()