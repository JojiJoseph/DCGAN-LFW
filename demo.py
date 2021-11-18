import matplotlib.pyplot as plt
import numpy as np
import torch
import streamlit as st

from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
gen.load_state_dict(torch.load("./gen_dcgan.pt"))
gen.eval()

st.title("DCGAN Demo")

fig = plt.figure()

for i in range(9):
	plt.subplot(3, 3, i+1)
	z = np.random.normal(size=(1, 128))
	z = torch.from_numpy(z).float().to(device)
	y = (gen(z)[0]+1)/2
	y = y.permute(1,2,0)
	y = torch.reshape(y, (64, 64, 3))
	y = y.detach().cpu().numpy()
	plt.axis("off")
	plt.imshow(y, cmap="gray")

st.pyplot(fig)
