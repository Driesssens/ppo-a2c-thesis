import gym
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from enum import IntEnum


def environment_name(environment_class):
    if isinstance(environment_class, tuple):
        return environment_class[1].name + '-' + environment_class[0].__name__
    elif isinstance(environment_class, list):
        name = environment_class[0].__name__
        for i in range(1, len(environment_class)):
            name = environment_class[i].name + '-' + name
        return name
    else:
        return environment_class.__name__


def instantiate_environment(environment_class):
    if isinstance(environment_class, tuple):
        return environment_class[1](environment_class[0]())
    elif isinstance(environment_class, list):
        obj = environment_class[0]()
        for i in range(1, len(environment_class)):
            obj = environment_class[i](obj)
        return obj
    else:
        return environment_class()


class MyFullyObservableWrapper(gym.core.ObservationWrapper):
    name = "Fully"

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.env.grid_size, self.env.grid_size, 3),  # number of cells
                dtype='uint8'
            ),
            'carrying': spaces.Box(
                low=0,
                high=255,
                shape=[2],
                dtype='uint8'
            )
        })

    def observation(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([255, self.env.agent_dir, 0])
        obs["image"] = full_grid

        if self.env.carrying:
            obs["carrying"] = np.array([OBJECT_TO_IDX[self.env.carrying.type], COLOR_TO_IDX[self.env.carrying.color]])
        else:
            obs["carrying"] = np.array([0, 6])  # empty type (0) and non-existing color (6)
        return obs


class MyFullyObservableWrapperBroadcast(gym.core.ObservationWrapper):
    name = "FullyBroad"

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.env.grid_size, self.env.grid_size, 4),  # number of cells
                dtype='uint8'
            ),
        })

    def observation(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([255, self.env.agent_dir, 0])
        carrying = OBJECT_TO_IDX[self.env.carrying.type] if self.env.carrying else 0
        extra_layer = np.full((full_grid.shape[0], full_grid.shape[1], 1), carrying)

        obs["image"] = np.concatenate((full_grid, extra_layer), axis=2)

        print(np.transpose(obs["image"]))

        return obs


class MyFullyObservableWrapperEgo(gym.core.ObservationWrapper):
    name = "Ego"

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.env.grid_size, self.env.grid_size, 3),  # number of cells
                dtype='uint8'
            ),
        })

    def observation(self, obs):
        full_grid = self.env.grid.encode()
        carrying = OBJECT_TO_IDX[self.env.carrying.type] if self.env.carrying else 0
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([9, self.env.agent_dir, carrying])

        obs["image"] = full_grid

        # print(np.transpose(obs["image"]))

        return obs


class ReducedActionWrapper(gym.core.Wrapper):
    name = "ActRed"

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        interact = 3

    def reset(self):
        return self.env.reset()

    def __init__(self, env):
        super().__init__(env)
        # self.__dict__.update(vars(env))
        self.actions = ReducedActionWrapper.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def step(self, action):
        if action in [self.actions.left, self.actions.right, self.actions.forward]:
            # print("from {} to {}".format(action, action))
            return self.env.step(action)
        else:
            # Get the position in front of the agent
            fwd_pos = self.unwrapped.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)

            if fwd_cell and fwd_cell.can_pickup():
                # print("from {} to {}".format(action, self.unwrapped.actions.pickup))
                return self.env.step(self.unwrapped.actions.pickup)
            else:
                # print("from {} to {}".format(action, self.unwrapped.actions.toggle))
                return self.env.step(self.unwrapped.actions.toggle)


class UndiscountedRewards(gym.core.RewardWrapper):
    name = "Undis"

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            reward = 1
        else:
            reward = 0
            # reward = -(1 / self.unwrapped.max_steps)

        return observation, reward, done, info


class HastyRewards(gym.core.RewardWrapper):
    name = "Haste"

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            reward = 1
        else:
            reward = -(1 / self.unwrapped.max_steps)

        return observation, reward, done, info
