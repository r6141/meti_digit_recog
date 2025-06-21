import os
import requests
import streamlit as st
import torch
from torch import nn
from PIL import Image
import numpy as np

# Constants
RAW_URL = "https://github.com/r6141/hosts/raw/main/generator.pth"
LOCAL_PATH = "generator.pth"

# Generator architecture
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        return self.net(x).view(-1, 1, 28, 28)

def one_hot(labels, num_classes):
    y = torch.zeros(labels.size(0), num_classes)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y

@st.cache_resource
def download_model(url=RAW_URL, local_path=LOCAL_PATH):
    if not os.path.exists(local_path):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

@st.cache_resource
def load_generator(model_path, latent_dim=100, label_dim=10):
    gen = Generator(latent_dim, label_dim)
    state = torch.load(model_path, map_location="cpu")
    gen.load_state_dict(state)
    gen.eval()
    return gen

def generate_images(model, digit, num_images=5, latent_dim=100):
    z = torch.randn(num_images, latent_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    labels_onehot = one_hot(labels, 10)
    with torch.no_grad():
        imgs_tensor = model(z, labels_onehot)
    imgs = ((imgs_tensor + 1.0) / 2.0 * 255).clamp(0, 255).byte().cpu().numpy()
    return [Image.fromarray(imgs[i,0], mode='L') for i in range(imgs.shape[0])]

def main():
    st.title("🎈 My updated GAN app")
    st.write("Generate MNIST-style digits using a pre-trained GAN.")

    model_path = download_model()
    generator = load_generator(model_path)

    digit = st.selectbox("Select digit", range(10), index=0)
    num_images = st.number_input("Number of images", 1, 20, 5)

    if st.button("Generate"):
        imgs = generate_images(generator, digit, num_images)
        cols = st.columns(min(num_images, 5))
        for idx, img in enumerate(imgs):
            cols[idx % len(cols)].image(img, caption=f"Digit {digit}", use_container_width=True)

if __name__ == "__main__":
    main()
