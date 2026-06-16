import torch
import torch.nn as nn

class MultiOmicsAutoencoder(nn.Module):
    def __init__(self, expr_dim, mut_dim, latent_dim=32, n_classes=5):
        super().__init__()
        self.expr_encoder = nn.Sequential(
            nn.Linear(expr_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
        self.expr_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, expr_dim)
        )
        self.mut_encoder = nn.Sequential(
            nn.Linear(mut_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )
        self.mut_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, mut_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, expr, mut):
        latent_expr = self.expr_encoder(expr)
        latent_mut = self.mut_encoder(mut)
        recon_expr = self.expr_decoder(latent_expr)
        recon_mut = self.mut_decoder(latent_mut)
        fused = torch.cat([latent_expr, latent_mut], dim=1)
        logits = self.classifier(fused)
        return logits, recon_expr, recon_mut
