#!/usr/bin/env python3

import argparse
import gym
import time
from gym_minigrid.minigrid import Grid

try:
    import gym_minigrid
except ImportError:
    pass

import utils
from thesis.environments import SmallUnlock8x8
from thesis.wrappers import MyFullyObservableWrapperBroadcast, MyFullyObservableWrapper, MyFullyObservableWrapperEgo, ReducedActionWrapper, UndiscountedRewards, HastyRewards
from gym_minigrid.envs import EmptyEnv8x8
import torch


def introspect_i2a(environment, model, seed=0, argmax=False, pause=0.1):
    utils.seed(seed)

    # Generate environment

    environment.seed(seed)

    # Define agent

    model_dir = utils.get_model_dir(model)
    preprocess_obss = utils.ObssPreprocessor(model_dir, environment.observation_space)
    model = utils.load_model(model_dir)

    # Run the agent

    done = True

    while True:
        if done:
            obs = environment.reset()
            print("Instr:", obs["mission"])

        time.sleep(pause)
        renderer = environment.render("human")

        preprocessed_obss = preprocess_obss([obs])

        with torch.no_grad():
            dist, _, pred_actions, pred_observations, pred_rewards = model(preprocessed_obss, introspect=True)

        renderer.window.update_imagination_display(pred_observations, pred_actions, pred_rewards)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        if torch.cuda.is_available():
            actions = actions.cpu().numpy()

        obs, reward, done, _ = environment.step(actions.item())

        if renderer.window is None:
            break


def enjoy_i2a_empty():
    environment_class = HastyRewards(MyFullyObservableWrapperEgo(EmptyEnv8x8()))
    introspect_i2a(environment_class, "I2A-3_Haste-Ego-EmptyEnv8x8_s1_18-11-08-07-45-08", pause=1)


def enjoy_i2a_unlocked():
    environment_class = HastyRewards(MyFullyObservableWrapperEgo(SmallUnlock8x8()))
    introspect_i2a(environment_class, "I2A-3_Haste-Ego-SmallUnlock8x8_s1_18-11-08-08-04-06", pause=1)


enjoy_i2a_unlocked()
