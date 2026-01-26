# model/dcl_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, embed_dim, diffusion_steps, beta_start, beta_end, device):
        super(DiffusionModel, self).__init__()
        self.embed_dim = embed_dim
        self.diffusion_steps = diffusion_steps
        self.device = device

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, diffusion_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim * 2),  # +1 for time step embedding
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process: q(x_t | x_0)
        Args:
            x_0: Initial node embeddings [batch_size, embed_dim]
            t: Time steps [batch_size]
        Returns:
            x_t: Noisy embeddings at time t
            noise: Added noise
        """
        noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        # Forward diffusion
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def reverse_diffusion(self, x_t, t):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Args:
            x_t: Noisy embeddings at time t [batch_size, embed_dim]
            t: Time steps [batch_size]
        Returns:
            Predicted x_{t-1}
        """
        # Time embedding
        t_emb = t.float().unsqueeze(-1) / self.diffusion_steps
        x_time = torch.cat([x_t, t_emb], dim=-1)

        # Predict noise
        predicted_noise = self.denoiser(x_time)

        # Extract coefficients
        alpha_t = self.alphas[t].view(-1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1)
        beta_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        # Compute x_{t-1} mean
        x_t_minus_1_mean = sqrt_recip_alpha_t * (
                x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # Add noise for x_{t-1} sample (only if t > 0)
        posterior_variance_t = self.posterior_variance[t].view(-1, 1)
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().unsqueeze(-1)
        x_t_minus_1 = x_t_minus_1_mean + mask * torch.sqrt(posterior_variance_t) * noise

        return x_t_minus_1

    def denoise(self, x_t, t):
        """
        Predict denoised x_0 from x_t
        Args:
            x_t: Noisy embeddings at time t [batch_size, embed_dim]
            t: Time steps [batch_size]
        Returns:
            Predicted x_0
        """
        # Time embedding
        t_emb = t.float().unsqueeze(-1) / self.diffusion_steps
        x_time = torch.cat([x_t, t_emb], dim=-1)

        # Predict noise
        predicted_noise = self.denoiser(x_time)

        # Extract coefficient
        sqrt_recip_alphas_cumprod_t = 1 / self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        # Predict x_0
        predicted_x0 = sqrt_recip_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise

        return predicted_x0

    def sample(self, x_T):
        """
        Sample from the reverse diffusion process
        Args:
            x_T: Noise sample [batch_size, embed_dim]
        Returns:
            Generated sample x_0
        """
        batch_size = x_T.shape[0]
        device = x_T.device

        # Start from pure noise
        x_t = x_T

        # Iteratively denoise
        for t in range(self.diffusion_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.reverse_diffusion(x_t, t_batch)

        return x_t


class DiffusionContrastiveLearning(nn.Module):
    def __init__(self, embed_dim, diffusion_steps, beta_start, beta_end, temperature, device):
        super(DiffusionContrastiveLearning, self).__init__()
        self.diffusion_model = DiffusionModel(embed_dim, diffusion_steps, beta_start, beta_end, device)
        self.temperature = temperature
        self.device = device

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, embeddings, adjacency_matrix=None):
        """
        Apply diffusion contrastive learning
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            adjacency_matrix: Adjacency matrix for defining positive pairs [num_nodes, num_nodes]
        Returns:
            Enhanced embeddings and contrastive loss
        """
        batch_size, embed_dim = embeddings.shape
        device = embeddings.device

        # Sample timesteps
        t = torch.randint(0, self.diffusion_model.diffusion_steps, (batch_size,), device=device)

        # Apply forward diffusion
        noisy_embeddings, _ = self.diffusion_model.forward_diffusion(embeddings, t)

        # Denoise to get enhanced embeddings
        denoised_embeddings = self.diffusion_model.denoise(noisy_embeddings, t)

        # Project embeddings for contrastive learning
        z_i = self.projection(embeddings)
        z_j = self.projection(denoised_embeddings)

        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # Define positive pairs
        if adjacency_matrix is None:
            # Use self-similarity as positive pairs (diagonal)
            positive_mask = torch.eye(batch_size, device=device).bool()
        else:
            # Use adjacency matrix to define positive pairs
            positive_mask = adjacency_matrix.bool()

        # Contrastive loss
        contrastive_loss = 0
        for i in range(batch_size):
            if positive_mask[i].sum() > 0:
                positive_indices = torch.where(positive_mask[i])[0]
                positive_similarities = similarity_matrix[i, positive_indices]

                # Negative similarities (all except positives)
                negative_mask = ~positive_mask[i]
                negative_similarities = similarity_matrix[i, negative_mask]

                # InfoNCE loss
                numerator = torch.exp(positive_similarities).sum()
                denominator = numerator + torch.exp(negative_similarities).sum()
                contrastive_loss -= torch.log(numerator / denominator)

        contrastive_loss = contrastive_loss / batch_size

        return denoised_embeddings, contrastive_loss

    def enhance_embeddings(self, embeddings):
        """
        Enhance embeddings using the trained diffusion model
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
        Returns:
            Enhanced embeddings
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Add noise
        t = torch.ones(batch_size, device=device).long() * (self.diffusion_model.diffusion_steps // 2)
        noisy_embeddings, _ = self.diffusion_model.forward_diffusion(embeddings, t)

        # Denoise
        enhanced_embeddings = self.diffusion_model.denoise(noisy_embeddings, t)

        return enhanced_embeddings