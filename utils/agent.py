import torch
from gym_minigrid.minigrid import Grid
import utils
import time

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, model_dir, observation_space, argmax=False, num_envs=1):
        self.preprocess_obss = utils.ObssPreprocessor(model_dir, observation_space)
        self.model = utils.load_model(model_dir)
        self.argmax = argmax
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model.recurrent:
            self.memories = torch.zeros(self.num_envs, self.model.memory_size)

        self.number = 0

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss)

        with torch.no_grad():
            if self.model.recurrent:
                dist, _, self.memories = self.model(preprocessed_obss, self.memories)
            else:
                dist, _, x, y, z = self.model(preprocessed_obss, introspect=True)

                if self.number == 7:
                    Grid.decode(y[0][0].round().numpy()).render_human()
                    print(len(x), len(y), len(z))
                    print(y[0].shape, z[0].shape)
                    print(x, y, z)
                self.number += 1


        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        if torch.cuda.is_available():
            actions = actions.cpu().numpy()

        return actions

    def get_action(self, obs):
        return self.get_actions([obs]).item()

    def analyze_feedbacks(self, rewards, dones):
        if self.model.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])