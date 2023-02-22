import torch
import torch.nn as nn
import torch.functional as F

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


class WorldModel(nn.Module):
    def __init__(self, obs_shape, action_shape, latent_size, classes=None):
        super().__init__()
        self.latent_size = latent_size
        self.action_shape = action_shape
        self._classes = classes

        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=6, stride=2)
        )

        if self._classes:
            self.z_logits = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, self._classes)
            )
            self.z_stoch = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, self._classes * self.latent_size)
            )
        else:
            self.z_mean = nn.Linear(latent_size, latent_size)
            self.z_std = nn.Linear(latent_size, latent_size)
            self.z_stoch = nn.Linear(latent_size, latent_size)

        self.rnn = nn.GRU(input_size=latent_size + action_shape[0], hidden_size=latent_size)

        self.dyn = nn.Sequential(
            nn.Linear(latent_size + action_shape[0], latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )

        self.reward = nn.Sequential(
            nn.Linear(latent_size + action_shape[0], latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1)
        )

        self.cont = nn.Sequential(
            nn.Linear(latent_size + action_shape[0], latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1),
            nn.Sigmoid()
        )

        self.latent_dist = torch.distributions.OneHotCategorical if self._classes else torch.distributions.Normal

    def encode(self, obs):
        obs = torch.Tensor(obs).to(self.device)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        return self.fc1(x)

    def decode(self, z, action):
        z = z.to(self.device)
        action = torch.Tensor(action).to(self.device)
        x = F.relu(self.fc2(torch.cat([z, action], dim=1)))
        x = x.view(-1, 256, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

    def dynamics_model(self, z, action):
        # Implementation of dynamics_model method.
        pass

    def reward_model(self, z, action):
        # Implementation of reward_model method.
        pass

    def continuation_model(self, z):
        # Implementation of continuation_model method.
        pass

    def loss_function(self, obs, action, reward, next_obs):
        # Implementation of loss_function method.
        pass

    def forward(self, obs, action, reward):
        # Implementation of forward method.
        pass