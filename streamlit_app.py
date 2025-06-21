# import streamlit as st

# st.title("🎈 My new changed app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

import streamlit as st
import torch
from torch import nn
from PIL import Image
import numpy as np

# Generator definition must match the one used during training
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)

# One-hot helper
def one_hot(labels, num_classes):
    y = torch.zeros(labels.size(0), num_classes)
    # scatter_ in-place; ensure same device (CPU here)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y

# Cache-loading the model so it’s loaded only once per session
@st.cache_resource
def load_generator(model_path: str, latent_dim: int = 100, label_dim: int = 10):
    gen = Generator(latent_dim=latent_dim, label_dim=label_dim)
    # Assumes "generator.pth" is in the same directory or provide full path
    state = torch.load(model_path, map_location="cpu")
    gen.load_state_dict(state)
    gen.eval()
    return gen

# Function to generate images given the model and chosen digit
def generate_images(model: nn.Module, digit: int, num_images: int = 5, latent_dim: int = 100):
    # sample latent vectors
    z = torch.randn(num_images, latent_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    labels_onehot = one_hot(labels, model.net[0].in_features - latent_dim if hasattr(model.net[0], 'in_features') else 10)
    # Note: model.label_dim is known (10); above is just a check—since you know label_dim=10, you can also pass 10 directly
    with torch.no_grad():
        imgs_tensor = model(z, labels_onehot)  # shape [num_images,1,28,28]
    # from [-1,1] to [0,255], uint8
    imgs = ((imgs_tensor + 1.0) / 2.0 * 255).clamp(0, 255).byte().cpu().numpy()
    pil_images = []
    for i in range(imgs.shape[0]):
        arr = imgs[i, 0]  # shape (28,28)
        pil = Image.fromarray(arr, mode='L')  # grayscale
        # Optionally resize for visibility:
        # pil = pil.resize((140, 140), resample=Image.BILINEAR)
        pil_images.append(pil)
    return pil_images

def main():
    st.title("🎈 My new changed app")
    st.write(
        "Generate MNIST-style digit images with a pre-trained GAN."
    )

    # Load model once
    generator = load_generator("/workspaces/meti_digit_recog/generator.pth", latent_dim=100, label_dim=10)

    # Dropdown for digit selection
    digit = st.selectbox("Select digit to generate", list(range(10)), index=0)

    # Optionally let user choose how many images; default 5
    num_images = st.number_input("Number of images", min_value=1, max_value=20, value=5, step=1)

    if st.button("Generate"):
        # Generate and display
        imgs = generate_images(generator, digit, num_images=num_images, latent_dim=100)
        # Display in a grid-like fashion
        cols = st.columns(min(num_images, 5))
        for idx, img in enumerate(imgs):
            col = cols[idx % len(cols)]
            col.image(img, caption=f"Digit {digit}", use_column_width=True)

if __name__ == "__main__":
    main()
