import torch
from torch import nn
from thesis.wrappers import instantiate_environment
from thesis.preprocessor import MyObssPreprocessor
import numpy as np
import utils
from torch_rl.utils import DictList, ParallelEnv
import datetime
from thesis.wrappers import environment_name
import time


def train_environment_model(environment_class, agent_name, n_environments=16, seed=0, learning_rate=5e-4, batch_per_environment=4, observation_weight=10, reward_weight=1, note=None, tensorboard=True, train_for_n_frames=None, log_interval=1, store_interval=10):
    saved_arguments = locals()

    date_suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    note = note + "_" if note else ""

    model_name = "EM_{}{}_s{}_{}".format(note, environment_name(environment_class), seed, date_suffix)
    model_directory = utils.get_model_dir(model_name)
    logger = utils.get_logger(model_directory)
    csv_file, csv_writer = utils.get_csv_writer(model_directory)
    logger.info("{}\n".format(saved_arguments))

    if tensorboard:
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(model_directory)

    total_start_time = time.time()

    utils.seed(seed)
    agent_model = utils.load_model(utils.get_model_dir(agent_name))
    environment_model = EnvironmentModel(environment_class)
    optimizer = torch.optim.Adam(environment_model.parameters(), lr=learning_rate)

    logger.info("Using pre-trained agent model: {}\n".format(agent_name))
    logger.info("{}\n".format(agent_model))

    logger.info("Environment model architecture: {}\n".format(agent_name))
    logger.info("{}\n".format(environment_model))

    environments = []
    for i in range(n_environments):
        environment = instantiate_environment(environment_class)
        environment.seed(seed + 10000 * i)
        environments.append(environment)

    observation_preprocessor = MyObssPreprocessor(environments[0].observation_space)
    environments = ParallelEnv(environments)

    n_updates = 0
    n_frames = 0

    last_observations = environments.reset()

    while train_for_n_frames is None or n_frames < train_for_n_frames:
        batch_start_time = time.time()

        old_observation_batch = torch.Tensor()
        action_batch = torch.Tensor()
        new_observation_batch = torch.IntTensor()
        reward_batch = torch.Tensor()

        for batch in range(batch_per_environment):
            distributions, _ = agent_model(observation_preprocessor(last_observations))
            actions = distributions.sample()
            new_observations, rewards, _ = environments.step(actions.numpy())

            old_observation_batch = torch.cat(old_observation_batch, torch.tensor(last_observations))
            action_batch = torch.cat(action_batch, actions)
            new_observation_batch = torch.cat(new_observation_batch, torch.tensor(new_observations))
            reward_batch = torch.cat(reward_batch, torch.tensor(rewards))

        optimizer.zero_grad()
        predicted_observations, predicted_rewards = environment_model(old_observation_batch, action_batch)
        observation_loss = nn.functional.mse_loss(predicted_observations, new_observation_batch)
        reward_loss = nn.functional.mse_loss(predicted_rewards, reward_batch)
        total_loss = observation_loss * observation_weight + reward_loss * reward_weight
        total_loss.backward()
        optimizer.step()

        additional_frames = n_environments * batch_per_environment
        n_frames += additional_frames
        n_updates += 1

        if n_updates % log_interval == 0:
            batch_end_time = time.time()
            fps = additional_frames / batch_end_time - batch_start_time
            duration = int(time.time() - total_start_time)

            header = ["update", "frames", "FPS", "duration"]
            data = [n_updates, n_frames, fps, duration]

            header += ["observation_loss", "reward_loss", "total_loss"]
            data += [observation_loss.item(), reward_loss.item(), total_loss.item()]

            if n_frames == additional_frames:
                csv_writer.writerow(header)
            csv_writer.writerow(data)
            csv_file.flush()

            if tensorboard:
                for field, value in zip(header, data):
                    tensorboard_writer.add_scalar(field, value, n_frames)

        if n_updates % store_interval == 0:
            utils.save_model(environment_model, model_directory)
            logger.info("Model successfully saved")


class EnvironmentModel(nn.Module):
    def __init__(self, environment_class):
        super().__init__()

        example_environment = instantiate_environment(environment_class)
        observation_shape = MyObssPreprocessor(example_environment.observation_space).obs_space["image"]
        self.n_actions = example_environment.action_space.n
        self.observation_width = observation_shape[0]
        self.observation_height = observation_shape[1]
        self.observation_layers = observation_shape[2]

        input_depth = self.observation_layers + self.n_actions

        self.convolution1 = nn.Sequential(
            nn.Conv2d(input_depth, 64, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.convolution2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.deconvolution = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=4, padding=0)

        self.reward_convolution = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        reward_convolution_out_shape = self.reward_convolution(
            self.convolution1(torch.zeros(1, input_depth, observation_shape[0], observation_shape[1]))
        )
        reward_convolution_out_size = int(np.prod(reward_convolution_out_shape.size()))

        self.reward_predictor = nn.Sequential(
            nn.Linear(reward_convolution_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observations, actions):
        batch_size = actions.size()[0]

        action_encoding_planes = torch.FloatTensor(batch_size, self.n_actions, self.observation_width, self.observation_height).zero_()
        action_encoding_planes[range(batch_size), actions] = 1.0
        input = torch.cat((observations, action_encoding_planes), dim=1)

        convolution1 = self.convolution1(input)
        convolution2 = self.convolution2(convolution1)
        sum = convolution1 + convolution2

        predicted_observation = self.deconvolution(sum)

        reward_convolution = self.reward_convolution(sum)
        predicted_reward = self.reward_predictor(reward_convolution.view(batch_size, -1))

        return predicted_observation, predicted_reward
