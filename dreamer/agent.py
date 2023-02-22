import embodied
import nets
import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box

from collections import deque
import numpy as np
import random

from nets import RSSM

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 3
IMAGE_SIZE = 64
NUM_ACTIONS = 4
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
GAMMA = 0.99
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 4
GRAD_NORM_CLIP = 1.0

# Define transformation to resize and normalize images
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Define replay buffer for storing transitions
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        state, action, next_state, reward, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), torch.LongTensor(action), torch.cat(next_state), torch.FloatTensor(reward), torch.FloatTensor(done)
        
    def __len__(self):
        return len(self.buffer)

# Define function to get environment observations as tensors
def get_observation(observation):
    if isinstance(observation, dict):
        return {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in observation.items()}
    else:
        return torch.from_numpy(observation).float().unsqueeze(0)

# Define function to get environment actions as tensors
def get_action(action):
    return torch.LongTensor([action])

# Define function to train the world model
def train_world_model(train_env, val_env, n_epochs=100, batch_size=32, learning_rate=1e-4, beta_pred=1, beta_dyn=0.5, beta_rep=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModel(obs_shape=train_env.observation_space.shape, action_shape=train_env.action_space.shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        epoch_train_losses = []
        epoch_val_losses = []

        # Train
        model.train()
        for i in range(train_env.n // batch_size):
            obs, actions, rewards, next_obs, dones = train_env.sample(batch_size)
            obs = torch.from_numpy(obs).float().to(device)
            actions = torch.from_numpy(actions).float().to(device)
            rewards = torch.from_numpy(rewards).float().to(device)
            next_obs = torch.from_numpy(next_obs).float().to(device)
            dones = torch.from_numpy(dones).float().to(device)

            # Forward pass
            z, pred, stoch = model(obs, actions)
            dyn_loss = F.mse_loss(z[1:], stoch[:-1])
            pred_loss = F.mse_loss(pred, next_obs)
            rep_loss = F.mse_loss(z[0], obs)
            loss = beta_dyn * dyn_loss + beta_pred * pred_loss + beta_rep * rep_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            for i in range(val_env.n // batch_size):
                obs, actions, rewards, next_obs, dones = val_env.sample(batch_size)
                obs = torch.from_numpy(obs).float().to(device)
                actions = torch.from_numpy(actions).float().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_obs = torch.from_numpy(next_obs).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)

                # Forward pass
                z, pred, stoch = model(obs, actions)
                dyn_loss = F.mse_loss(z[1:], stoch[:-1])
                pred_loss = F.mse_loss(pred, next_obs)
                rep_loss = F.mse_loss(z[0], obs)
                loss = beta_dyn * dyn_loss + beta_pred * pred_loss + beta_rep * rep_loss

                epoch_val_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))

        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")

    return model, train_losses, val_losses




class Agent(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        self.config = config
        self.model = embodied.Agent(obs_space, act_space, step, config)
        self.wm = WorldModel(obs_space, act_space, config, name='wm')

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, config):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
        self.rssm = nets.RSSM(obs_space.shape, act_space.shape, config.z_dim, config.h_dim)
        self.heads = {
            'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
            'reward': nets.MLP((), **config.reward_head, name='rew'),
            'cont': nets.MLP((), **config.cont_head, name='cont')
        }
        self.opt = torch.optim.Adam(self.parameters(), lr=config.lr)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop('image'), scales.pop('vector')
        scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
        scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
        self.scales = scales
