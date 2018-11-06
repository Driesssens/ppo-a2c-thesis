import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from thesis.preprocessor import MyObssPreprocessor
from thesis.wrappers import instantiate_environment


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MyACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, environment_class, use_instr=False, use_memory=True):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory

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
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.use_carrying = False
        if "carrying" in obs_space:
            self.use_carrying = True

        # Define memory
        if self.use_memory:
            if self.use_carrying:
                self.image_embedding_size += obs_space['carrying']
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define instruction embedding
        if self.use_instr:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["instr"], self.word_embedding_size)
            self.instr_embedding_size = 128
            self.instr_rnn = nn.GRU(self.word_embedding_size, self.instr_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr:
            self.embedding_size += self.instr_embedding_size

        if self.use_carrying and not self.use_memory:
            self.embedding_size += obs_space['carrying']

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, example_environment.action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            if self.use_carrying:
                x = torch.cat((x, obs.carrying), dim=1)

            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])

            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            if self.use_carrying:
                embedding = torch.cat((embedding, obs.carrying), dim=1)

        if self.use_instr:
            embed_instr = self._get_embed_instr(obs.instr)
            embedding = torch.cat((embedding, embed_instr), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_instr(self, instr):
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]
