from torch_rl import A2CAlgo
from gym_minigrid.envs import EmptyEnv6x6, EmptyEnv10x10, EmptyEnv16x16, Unlock, UnlockPickup
from thesis.train_programmatically import train
from thesis.wrappers import MyFullyObservableWrapper, MyFullyObservableWrapperBroadcast, MyFullyObservableWrapperEgo, ReducedActionWrapper, UndiscountedRewards, HastyRewards
from thesis.environments import SmallUnlock
from thesis.environment_model import EnvironmentModel, train_environment_model


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
    environment_class = [SmallUnlock, MyFullyObservableWrapperEgo]
    # environment_class = [EmptyEnv10x10, MyFullyObservableWrapperEgo]
    algorithm = A2CAlgo(environment_class, n_processes=16)
    train(environment_class, algorithm, tensorboard=True, no_mem=True)


if __name__ == '__main__':
    create_some_pretrained_agents()
