#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys
from thesis.preprocessor import MyObssPreprocessor
from thesis.acmodel import MyACModel
from thesis.wrappers import environment_name

try:
    import gym_minigrid
except ImportError:
    pass

import utils
from model import ACModel


def train(environment,  # name of the environment to train on (REQUIRED)
          algorithm,  # class
          seed=1,  # random seed (default: 1)
          procs=16,  # number of processes (default: 16)
          frames=10 ** 7,  # number of frames of training (default: 10e7)
          log_interval=1,  # number of updates between two logs (default: 1)
          save_interval=10,  # number of updates between two saves (default: 0, 0 means no saving)
          frames_per_proc=None,  # number of frames per process before update (default: 5 for A2C and 128 for PPO)
          discount=0.99,  # discount factor (default: 0.99)
          lr=7e-4,  # learning rate for optimizers (default: 7e-4)
          gae_lambda=0.95,  # lambda coefficient in GAE formula (default: 0.95, 1 means no gae)
          entropy_coef=0.01,  # entropy term coefficient (default: 0.01)
          value_loss_coef=0.5,  # value loss term coefficient (default: 0.5)
          max_grad_norm=0.5,  # maximum norm of gradient (default: 0.5)
          recurrence=1,  # number of steps the gradient is propagated back in time (default: 1)
          optim_eps=1e-5,  # Adam and RMSprop optimizer epsilon (default: 1e-5)
          optim_alpha=0.99,  # RMSprop optimizer apha (default: 0.99)
          clip_eps=0.2,  # clipping epsilon for PPO (default: 0.2)
          epochs=4,  # number of epochs for PPO (default: 4)
          batch_size=256,  # batch size for PPO (default: 256)
          no_instr=False,  # don't use instructions in the model
          no_mem=False,  # don't use memory in the model
          note=None, # name suffix
          tensorboard=True):
    saved_arguments = locals()

    date_suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    note = note + "_" if note else ""

    model_name = "A2C_{}{}_s{}_{}".format(note, environment_name(environment), seed, date_suffix)
    model_dir = utils.get_model_dir(model_name)

    # Define logger, CSV writer and Tensorboard writer
    logger = utils.get_logger(model_dir)
    csv_file, csv_writer = utils.get_csv_writer(model_dir)

    if tensorboard:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(model_dir)

    # Log command and all script arguments
    logger.info("{}\n".format(saved_arguments))

    # Set seed for all randomness sources
    utils.seed(seed)

    # Load training status
    try:
        status = utils.load_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}

    # Define actor-critic model

    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]

    acmodel = MyACModel(environment)
    algorithm.load_acmodel(acmodel)
    logger.info("{}\n".format(acmodel))
    logger.info("Model uses carrying: {}\n".format(acmodel.use_carrying))

    while num_frames < frames:
        # Update model parameters

        update_start_time = time.time()
        logs = algorithm.update_parameters()
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - total_start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)
            csv_file.flush()

            if tensorboard:
                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            status = {"num_frames": num_frames, "update": update}
            utils.save_status(status, model_dir)

        # Save vocabulary and model

        if save_interval > 0 and update % save_interval == 0:
            utils.save_model(algorithm.acmodel, model_dir)
            logger.info("Model successfully saved")
