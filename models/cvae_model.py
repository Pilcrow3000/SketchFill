"""
cvae_model.py — Class-Conditional Variational Autoencoder (CVAE)
Architecture for SketchFill: 128x128 grayscale sketch generation
with class-label conditioning at both encoder and decoder.

Encoder: spatial class-tiling + 4x Conv2D → mu, logvar
Decoder: FC projection + 4x ConvTranspose2D → sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Convolutional encoder with spatial class conditioning.
    The one-hot class vector is tiled into a spatial map and concatenated
    with the input image so every conv layer sees the class context.
    """

    def __init__(self, latent_dim: int = 64, num_classes: int = 6, image_size: int = 128):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.image_size  = image_size

        in_channels = 1 + num_classes  # image + spatial class map

        self.conv = nn.Sequential(
            # 128 → 64
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 → 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 → 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 → 4 (extra layer)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flattened spatial size after 5x stride-2 downsampling: 4x4
        flat_dim = 512 * (image_size // 32) * (image_size // 32)
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, c_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        [B, 1, H, W]  input image
            c_onehot: [B, num_classes] one-hot class label
        Returns:
            mu, logvar: each [B, latent_dim]
        """
        # Tile class vector into spatial map [B, num_classes, H, W]
        B, _, H, W = x.shape
        c_spatial = c_onehot.view(B, self.num_classes, 1, 1).expand(B, self.num_classes, H, W)

        # Concatenate along channel dimension → [B, 1+num_classes, H, W]
        x_cond = torch.cat([x, c_spatial], dim=1)

        h = self.conv(x_cond)
        h = h.flatten(start_dim=1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Convolutional decoder with class conditioning via latent vector concatenation.
    """

    def __init__(self, latent_dim: int = 64, num_classes: int = 6, image_size: int = 128):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.image_size  = image_size

        # Projected feature map starts at 4x4 with 512 channels
        self.base_size   = image_size // 32   # = 4 for 128x128
        flat_dim         = 512 * self.base_size * self.base_size

        self.fc = nn.Linear(latent_dim + num_classes, flat_dim)

        self.deconv = nn.Sequential(
            # 4 → 8 (extra layer)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 8 → 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 16 → 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32 → 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 → 128
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, c_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:        [B, latent_dim]
            c_onehot: [B, num_classes]
        Returns:
            recon:    [B, 1, H, W]
        """
        z_cond = torch.cat([z, c_onehot], dim=1)
        h = self.fc(z_cond)
        h = h.view(-1, 512, self.base_size, self.base_size)
        return self.deconv(h)


# ── CVAE ──────────────────────────────────────────────────────────────────────

class CVAE(nn.Module):
    """
    Class-Conditional Variational Autoencoder for sketch generation.

    Args:
        latent_dim:  dimensionality of the latent space (default 64)
        num_classes: number of sketch categories (default 6)
        image_size:  input/output image size in pixels (default 128)
    """

    def __init__(self, latent_dim: int = 64, num_classes: int = 6, image_size: int = 128):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.image_size  = image_size

        self.encoder = Encoder(latent_dim, num_classes, image_size)
        self.decoder = Decoder(latent_dim, num_classes, image_size)

    def encode(self, x: torch.Tensor, c_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image + class to (mu, logvar)."""
        return self.encoder(x, c_onehot)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std
        During inference (eval mode) returns mu directly for deterministic output.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, c_onehot: torch.Tensor) -> torch.Tensor:
        """Decode latent vector + class to reconstructed image."""
        return self.decoder(z, c_onehot)

    def forward(
        self, x: torch.Tensor, c_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        Returns:
            recon_x: [B, 1, H, W] — reconstructed image
            mu:      [B, latent_dim]
            logvar:  [B, latent_dim]
        """
        mu, logvar = self.encode(x, c_onehot)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z, c_onehot)
        return recon_x, mu, logvar

    def sample(
        self, c_onehot: torch.Tensor, num_samples: int = 1, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate samples by sampling z ~ N(0, I) and decoding.
        Args:
            c_onehot: [num_classes] or [B, num_classes] — class condition
            num_samples: how many samples to generate
        Returns:
            images: [num_samples, 1, H, W]
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            if c_onehot.dim() == 1:
                c_onehot = c_onehot.unsqueeze(0).expand(num_samples, -1)
            elif c_onehot.shape[0] == 1:
                c_onehot = c_onehot.expand(num_samples, -1)
            return self.decode(z, c_onehot)


# ── Loss Function ─────────────────────────────────────────────────────────────

def cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CVAE loss = Reconstruction loss + beta * KL divergence

    Args:
        recon_x: [B, 1, H, W] reconstructed image (sigmoid output)
        x:       [B, 1, H, W] original image
        mu:      [B, latent_dim]
        logvar:  [B, latent_dim]
        beta:    KL weight (beta-VAE, default 1.0)
        free_bits: minimum KL per dimension (prevents posterior collapse)

    Returns:
        (total_loss, recon_loss, kl_loss) — all are scalar tensors
    """
    # BCE reconstruction loss, averaged over batch
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)

    # KL divergence per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Free bits: prevent KL from going below threshold per dimension
    if free_bits > 0:
        kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))
    
    # Sum over latent dimensions, mean over batch
    kl_loss = kl_per_dim.sum(dim=1).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  CVAE Architecture Smoke Test")
    print("=" * 50)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH      = 4
    NUM_CLS    = 6
    LATENT_DIM = 64
    IMG_SIZE   = 128

    print(f"\nDevice: {device}")
    model = CVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLS, image_size=IMG_SIZE).to(device)

    # Random inputs
    x       = torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE).to(device)
    c_idx   = torch.randint(0, NUM_CLS, (BATCH,))
    c_one   = torch.zeros(BATCH, NUM_CLS).to(device)
    c_one.scatter_(1, c_idx.unsqueeze(1).to(device), 1.0)

    # Forward pass
    recon, mu, logvar = model(x, c_one)
    total, recon_l, kl_l = cvae_loss(recon, x, mu, logvar, beta=1.0)

    print(f"\nInput  shape : {list(x.shape)}")
    print(f"Recon  shape : {list(recon.shape)}")
    print(f"mu     shape : {list(mu.shape)}")
    print(f"logvar shape : {list(logvar.shape)}")
    print(f"Total  loss  : {total.item():.4f}")
    print(f"Recon  loss  : {recon_l.item():.4f}")
    print(f"KL     loss  : {kl_l.item():.4f}")

    assert recon.shape == (BATCH, 1, IMG_SIZE, IMG_SIZE), "Recon shape mismatch!"
    assert mu.shape    == (BATCH, LATENT_DIM),             "mu shape mismatch!"
    assert logvar.shape == (BATCH, LATENT_DIM),            "logvar shape mismatch!"
    assert torch.isfinite(total),                          "Loss is not finite!"
    assert recon.min() >= 0.0 and recon.max() <= 1.0,     "Recon out of [0,1]!"

    # Sample test
    model.eval()
    c_sample = torch.zeros(1, NUM_CLS).to(device)
    c_sample[0, 0] = 1.0
    samples = model.sample(c_sample, num_samples=3, device=device)
    assert samples.shape == (3, 1, IMG_SIZE, IMG_SIZE), "Sample shape mismatch!"

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    print("\n✓ CVAE architecture OK")
    sys.exit(0)