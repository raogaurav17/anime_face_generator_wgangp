import torch
import torchvision.utils as vutils
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np

# === Load Generator ===
z_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Your Generator model class here
from generator_model import Generator

gen = Generator(z_dim=z_dim).to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
gen.eval()
torch.save(gen.state_dict(), "GGenerator.pth")


# === Generate Fake Images ===
def generate_images(num_images=32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn(num_images, z_dim, 1, 1).to(device)
    with torch.no_grad():
        fake = gen(noise).detach().cpu()

    fake = torch.nan_to_num(fake)

    # Normalize correctly (if your generator outputs in [-1, 1])
    grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))

    # Convert to image
    npimg = grid.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # [H, W, C]
    npimg = np.clip(npimg * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(npimg)


# Gradio Interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Slider(1, 64, step=1, value=32, label="Number of Faces"),
        gr.Number(label="Random Seed (optional)", value=None),
    ],
    outputs=gr.Image(type="pil", label="Generated Faces"),
    title="Anime Face Generator (WGAN-GP)",
    description="Click 'Submit' to generate anime-style faces using a WGAN-GP model trained on anime face dataset.",
    allow_flagging="never",
)

iface.launch()
