from torch_rl import A2CAlgo
from gym_minigrid.envs import EmptyEnv, EmptyEnv16x16, Unlock, UnlockPickup
from thesis.train_programmatically import train
from thesis.wrappers import MyFullyObservableWrapper, MyFullyObservableWrapperBroadcast, MyFullyObservableWrapperEgo, ReducedActionWrapper
from thesis.environments import ShapedUnlock


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


if __name__ == '__main__':
    reduced_action()
