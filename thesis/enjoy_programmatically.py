#!/usr/bin/env python3

import argparse
import gym
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils
from thesis.environments import SmallUnlock8x8
from thesis.wrappers import MyFullyObservableWrapperBroadcast, MyFullyObservableWrapper, MyFullyObservableWrapperEgo, ReducedActionWrapper, UndiscountedRewards, HastyRewards
from gym_minigrid.envs import EmptyEnv8x8


def enjoy(environment, model, seed=0, argmax=False, pause=0.1):
    utils.seed(seed)

    # Generate environment

    environment.seed(seed)

    # Define agent

    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(model_dir, environment.observation_space, argmax)

    # Run the agent

    done = True

    while True:
        if done:
            obs = environment.reset()
            print("Instr:", obs["mission"])

        time.sleep(pause)
        renderer = environment.render("human")
        renderer.window.update_imagination_display([[1, 2, 3], 2, 3], None, None)

        action = agent.get_action(obs)
        obs, reward, done, _ = environment.step(action)
        agent.analyze_feedback(reward, done)

        if renderer.window is None:
            break


def enjoy_FullyObsTest():
    env = gym.make("MiniGrid-Unlock-v0")
    env = FullyObsWrapper2(env)
    enjoy(env, "MiniGrid-Unlock-v0_a2c_seed1_18-11-03-11-13-46")


def enjoy_unlock():
    environment_class = gym_minigrid.envs.Unlock()
    enjoy(environment_class, "Unlock_A2CAlgo_seed1_18-11-05-14-38-23")


def enjoy_wrapped():
    environment_class = HastyRewards(MyFullyObservableWrapperEgo(SmallUnlock()))
    enjoy(environment_class, "A2C_NoMem8x8_Haste-Ego-SmallUnlock_s1_18-11-07-21-56-42")


def enjoy_i2a_empty():
    environment_class = HastyRewards(MyFullyObservableWrapperEgo(EmptyEnv8x8()))
    enjoy(environment_class, "I2A-3_Haste-Ego-EmptyEnv8x8_s1_18-11-08-07-45-08")


def enjoy_i2a_unlocked():
    environment_class = HastyRewards(MyFullyObservableWrapperEgo(SmallUnlock8x8()))
    enjoy(environment_class, "I2A-3_Haste-Ego-SmallUnlock8x8_s1_18-11-08-08-04-06")


enjoy_i2a_unlocked()
