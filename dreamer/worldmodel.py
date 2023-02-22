import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, obs_shape, act_shape, z_dim, h_dim):
        super().__init__()
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim * 2)
        )

        # Prior network
        self.prior = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, h_dim + z_dim)
        )

        # Posterior network
        self.posterior = nn.Sequential(
            nn.Linear(h_dim + obs_shape[0] + act_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, h_dim + z_dim * 2)
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, obs_shape[0])
        )

        # Transition network
        self.transition = nn.GRUCell(z_dim + act_shape[0], h_dim)

    def encode(self, obs):
        # Compute mean and variance of the latent variable z
        q_z = self.encoder(obs)
        mean, logvar = q_z[:, :self.z_dim], q_z[:, self.z_dim:]
        std = torch.exp(0.5 * logvar)
        # Sample z from the reparameterized Gaussian distribution
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

    def transition_model(self, z, h, a):
        # Concatenate the latent variable z and action a as input to the transition model
        inp = torch.cat([z, a], dim=1)
        # Compute the new hidden state h_t
        h_t = self.transition(inp, h)
        return h_t

    def prior_model(self, z, h):
        # Concatenate the latent variable z and the previous hidden state h as input to the prior model
        inp = torch.cat([h, z], dim=1)
        # Compute the mean and variance of the prior distribution over z
        p_z = self.prior(inp)
        mean, logvar = p_z[:, :self.z_dim], p_z[:, self.z_dim:]
        return mean, logvar

    def posterior_model(self, obs, h, a):
        # Concatenate the previous hidden state h, the current observation obs, and action a as input to the posterior model
        inp = torch.cat([h, obs, a], dim=1)
        # Compute the mean and variance of the posterior distribution over z
        q_z = self.posterior(inp)
        mean, logvar = q_z[:, :self.z_dim], q_z[:, self.z_dim:]
        return mean, logvar

    def decode(self, z, h):
        # Concatenate the latent variable z and the current hidden state h as input to the decoder
        inp = torch.cat([h, z], dim=1)
        # Compute the reconstruction of the input observation x_t
        x_t = self.decoder(inp)
        return x_t


    def forward(self, obs, act, hidden, cell):
        # Encode observation
        obs_enc = self.encode(obs)

        # Prior
        prior_inputs = torch.cat([hidden, obs_enc[:, :self.z_dim]], dim=-1)
        prior_mean, prior_logvar = torch.chunk(self.prior(prior_inputs), chunks=2, dim=-1)

        # Posterior
        post_inputs = torch.cat([hidden, obs, act], dim=-1)
        posterior_mean, posterior_logvar = torch.chunk(self.posterior(post_inputs), chunks=2, dim=-1)

        # Sample latent variable
        z = self.reparameterize(posterior_mean, posterior_logvar)

        # Transition
        transition_inputs = torch.cat([z, act], dim=-1)
        next_hidden = self.transition(transition_inputs, hidden)

        # Decode observation
        obs_dec = self.decode(torch.cat([next_hidden, z], dim=-1))

        return prior_mean, prior_logvar, posterior_mean, posterior_logvar, z, next_hidden, obs_dec

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def reset_hidden(self, batch_size):
        return torch.zeros(batch_size, self.h_dim)

    def reset_cell(self, batch_size):
        return torch.zeros(batch_size, self.z_dim)

    def predict(self, obs, act, hidden, cell):
        prior_mean, prior_logvar, posterior_mean, posterior_logvar, z, next_hidden, obs_dec = self(obs, act, hidden, cell)
        return prior_mean, prior_logvar, posterior_mean, posterior_logvar, z, next_hidden

    def update(self, obs, act, hidden, cell):
        _, _, posterior_mean, posterior_logvar, z, next_hidden, obs_dec = self(obs, act, hidden, cell)
        return posterior_mean, posterior_logvar, z, next_hidden, obs_dec