from torch_rl import A2CAlgo
from gym_minigrid.envs import EmptyEnv6x6, EmptyEnv8x8, EmptyEnv16x16, Unlock, UnlockPickup
from thesis.train_programmatically import train
from thesis.wrappers import MyFullyObservableWrapper, MyFullyObservableWrapperBroadcast, MyFullyObservableWrapperEgo, ReducedActionWrapper, UndiscountedRewards, HastyRewards
from thesis.environments import SmallUnlock8x8
from thesis.environment_model import EnvironmentModel, train_environment_model
from thesis.i2a_algorithm import I2Algorithm
from thesis.i2a_train import train_i2a_model


def first_test_run():
    environment_class = EmptyEnv16x16
    algorithm = A2CAlgo(environment_class)
    train(environment_class, algorithm)


def unlock_run():
    environment_class = UnlockPickup
    algorithm = A2CAlgo(environment_class)
    train(environment_class, algorithm)


def fully_observable():
    environment_class = (Unlock, MyFullyObservableWrapper)
    algorithm = A2CAlgo(environment_class)
    train(environment_class, algorithm)


def fully_observable_broadcast():
    environment_class = (ShapedUnlock, MyFullyObservableWrapperEgo)
    algorithm = A2CAlgo(environment_class)
    train(environment_class, algorithm, tensorboard=True)


def reduced_action():
    environment_class = [Unlock, MyFullyObservableWrapperEgo]
    algorithm = A2CAlgo(environment_class, n_processes=16)
    train(environment_class, algorithm, tensorboard=True)


def rewards():
    environment_class = [EmptyEnv, MyFullyObservableWrapperEgo, UndiscountedRewards]
    algorithm = A2CAlgo(environment_class, n_processes=16)
    train(environment_class, algorithm, tensorboard=True, note="NoNegativeRewards")


def create_some_pretrained_agents():
    environment_class = [SmallUnlock8x8, MyFullyObservableWrapperEgo]
    algorithm = A2CAlgo(environment_class, n_processes=16)
    train(environment_class, algorithm, tensorboard=True, no_mem=True)


def test_environment_model_empty():
    environment_class = [EmptyEnv8x8, MyFullyObservableWrapperEgo]
    agent_name = "A2C_Ego-EmptyEnv8x8_s1_18-11-07-23-15-04"
    train_environment_model(environment_class, agent_name, tensorboard=True)


def test_environment_model_unlock():
    environment_class = [SmallUnlock8x8, MyFullyObservableWrapperEgo]
    agent_name = "A2C_Ego-SmallUnlock8x8_s1_18-11-07-23-18-18"
    train_environment_model(environment_class, agent_name, tensorboard=True)


def test_i2a_empty():
    environment_class = [EmptyEnv8x8, MyFullyObservableWrapperEgo, HastyRewards]
    environment_model_name = "EM_Haste-Ego-EmptyEnv8x8_s0_18-11-07-23-10-39"
    algorithm = I2Algorithm(environment_class)
    train_i2a_model(environment_class, environment_model_name, algorithm, 3, tensorboard=True)


def test_i2a_unlock():
    environment_class = [SmallUnlock8x8, MyFullyObservableWrapperEgo]
    environment_model_name = "EM_Ego-SmallUnlock8x8_s0_18-11-07-23-35-55"
    algorithm = I2Algorithm(environment_class)
    train_i2a_model(environment_class, environment_model_name, algorithm, 1)


if __name__ == '__main__':
    test_i2a_unlock()
