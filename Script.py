import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob


class Critic(nn.Module):
    def __init__(self, channels, features_d):
        super().__init__()
        self.net = nn.Sequential(
            self.block(channels, features_d, 4, 2, 1),  # 64x64 → 32x32
            self.block(features_d, features_d * 2, 4, 2, 1),  # 32x32 → 16x16
            self.block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 → 8x8
            self.block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 → 4x4
            nn.Conv2d(
                features_d * 8, 1, kernel_size=4, stride=2, padding=0
            ),  # 4x4 → 1x1
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, feature_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.block(z_dim, feature_g * 16, 4, 1, 0),  # 1x1 → 4x4
            self.block(feature_g * 16, feature_g * 8, 4, 2, 1),  # 4x4 → 8x8
            self.block(feature_g * 8, feature_g * 4, 4, 2, 1),  # 8x8 → 16x16
            self.block(feature_g * 4, feature_g * 2, 4, 2, 1),  # 16x16 → 32x32
            nn.ConvTranspose2d(
                feature_g * 2,
                channel_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 32x32 → 64x64
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():

    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    torch.manual_seed(0)

    # Test Discriminator
    x = torch.randn((N, in_channels, H, W))
    critic = Critic(in_channels, 8)
    initialize_weights(critic)
    print("Discriminator output shape:", critic(x).shape)
    assert critic(x).shape == (N, 1, 1, 1), "Critic test failed"

    # Test Generator
    gen = Generator(z_dim, in_channels, 64)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    out = gen(z)
    print("Generator output shape:", out.shape)
    assert out.shape == (N, in_channels, H, W), "Generator test failed"

    print("Success: All tests passed!")


test()

# hyperparameter
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-5
batch_size = 64
image_size = 64
channel_img = 3
z_dim = 100
num_epochs = 15
features_disc = 64
features_gen = 64
critic_iter = 5
lambda_gp = 10

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channel_img)], [0.5 for _ in range(channel_img)]
        ),
    ]
)


class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = glob.glob(root_dir + "/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


dataset = AnimeFaceDataset("/content/animefacedataset/images", transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channel_img, features_gen).to(device)
crt = Critic(channel_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(crt)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
opt_disc = optim.Adam(crt.parameters(), lr=lr, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/real")
writer_fake = SummaryWriter(f"runs/fake")
step = 0
best_lossG = float("inf")


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # calculate critic score
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# Training
gen.train()
crt.train()

for epoch in range(num_epochs):
    loop = tqdm(loader, leave=True)

    for batch_id, real in enumerate(loop):
        real = real.to(device)
        batch_size = real.size(0)

        # Train Critic multiple times
        for _ in range(critic_iter):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            crt_real = crt(real).reshape(-1)
            crt_fake = crt(fake.detach()).reshape(-1)
            gp = gradient_penalty(crt, real, fake, device=device)

            lossD = -(torch.mean(crt_real) - torch.mean(crt_fake)) + lambda_gp * gp

            opt_disc.zero_grad()
            lossD.backward()
            opt_disc.step()

        # Train Generator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)
        output = crt(fake).reshape(-1)
        lossG = -torch.mean(output)

        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Accuracy-like metrics
        crt_acc_real = torch.mean(crt_real).item()
        crt_acc_fake = torch.mean(crt_fake).item()

        # TQDM Logging
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(
            {
                "Loss_D": f"{lossD.item():.4f}",
                "Loss_G": f"{lossG.item():.4f}",
                "D(x)": f"{crt_acc_real:.2f}",
                "D(G(z))": f"{crt_acc_fake:.2f}",
            }
        )

        # TensorBoard Logging
        if step % 100 == 0:
            with torch.no_grad():
                gen.eval()
                fake = gen(fixed_noise).reshape(-1, 3, 64, 64)
                real_imgs = real[:32].reshape(-1, 3, 64, 64)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_imgs, normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)

                writer_fake.add_scalar("Generator Loss", lossG.item(), global_step=step)
                writer_real.add_scalar(
                    "Discriminator Loss", lossD.item(), global_step=step
                )
                writer_real.add_scalar(
                    "Critic Real Score", crt_acc_real, global_step=step
                )
                writer_fake.add_scalar(
                    "Critic Fake Score", crt_acc_fake, global_step=step
                )

                gen.train()

        step += 1

torch.save(gen.state_dict(), "best_generator.pth")
torch.save(crt.state_dict(), "best_critic.pth")
