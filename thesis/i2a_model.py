import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from thesis.preprocessor import MyObssPreprocessor
from thesis.wrappers import instantiate_environment
from thesis.acmodel import MyACModel
import numpy as np


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class I2AModel(nn.Module, torch_rl.ACModel):
    def __init__(self, environment_class, environment_model, imagination_steps, rollout_hidden_size=256):
        super().__init__()

        example_environment = instantiate_environment(environment_class)
        obs_space = MyObssPreprocessor(example_environment.observation_space).obs_space

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        z = obs_space["image"][2]

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(z, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.imagination_steps = imagination_steps
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.n_actions = example_environment.action_space.n

        self.embedding_size = self.image_embedding_size + rollout_hidden_size * self.n_actions

        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.encoder = RolloutEncoder((z, n, m), rollout_hidden_size)

        self.apply(initialize_parameters)

        imagination_policy = MyACModel(environment_class, use_memory=False)
        object.__setattr__(self, "environment_model", environment_model)
        object.__setattr__(self, "imagination_policy", imagination_policy)

    def forward(self, obs):
        convolutions = self.image_conv(torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3))
        flat_convolutions = convolutions.reshape(convolutions.shape[0], -1)

        encoded_rollouts = self.rollouts_batch(obs.image)

        both_pathways = torch.cat((flat_convolutions, encoded_rollouts), dim=1)

        logits = self.actor(both_pathways)
        distributions = Categorical(logits=F.log_softmax(logits, dim=1))

        values = self.critic(both_pathways).squeeze(1)

        return distributions, values

    def rollouts_batch(self, observations):
        batch_size = observations.size()[0]
        observations_shape = observations.size()[1:]

        if batch_size == 1:
            old_observations = observations.expand(batch_size * self.n_actions, *observations_shape)
        else:
            old_observations = observations.unsqueeze(1)
            old_observations = old_observations.expand(batch_size, self.n_actions, *observations_shape)
            old_observations = old_observations.contiguous().view(-1, *observations_shape)

        actions = torch.tensor(np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size))
        predicted_observations, predicted_rewards = [], []

        for step_idx in range(self.imagination_steps):
            new_observations, new_rewards = self.environment_model(old_observations, actions)
            predicted_observations.append(new_observations.detach())
            predicted_rewards.append(new_rewards.detach())

            # don't need actions for the last step
            if step_idx == self.imagination_steps - 1:
                break

            # combine the delta from EM into new observation
            old_observations = torch.transpose(torch.transpose(new_observations, 2, 3), 1, 3)

            # select actions
            distributions, _, _ = self.imagination_policy(old_observations, None)
            actions = distributions.sample()

        predicted_observations = torch.stack(predicted_observations)
        predicted_rewards = torch.stack(predicted_rewards)
        encoded = self.encoder(predicted_observations, predicted_rewards)
        return encoded.view(batch_size, -1)


class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(RolloutEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.rnn = nn.LSTM(input_size=conv_out_size + 1, hidden_size=hidden_size, batch_first=False)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs_v, reward_v):
        """
        Input is in (time, batch, *) order
        """
        n_time = obs_v.size()[0]
        n_batch = obs_v.size()[1]
        n_items = n_time * n_batch
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:])
        conv_out = self.conv(obs_flat_v)
        conv_out = conv_out.view(n_time, n_batch, -1)
        rnn_in = torch.cat((conv_out, reward_v), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)
