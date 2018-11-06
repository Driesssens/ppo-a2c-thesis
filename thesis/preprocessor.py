import torch_rl
import numpy
import torch


class MyObssPreprocessor:
    """A preprocessor of observations returned by the environment.
    It converts MiniGrid observation space and MiniGrid observations
    into the format that the model can handle."""

    def __init__(self, obs_space):
        self.obs_space = {
            "image": obs_space.spaces['image'].shape,
        }

        if 'carrying' in obs_space.spaces:
            self.obs_space['carrying'] = 2

    def __call__(self, obss, device=None):
        """Converts a list of MiniGrid observations, i.e. a list of
        (image, instruction) tuples into two PyTorch tensors.

        The images are concatenated. The instructions are tokenified, then
        tokens are converted into lists of ids using a Vocabulary object, and
        finally, the lists of ids are concatenated.

        Returns
        -------
        preprocessed_obss : DictList
            Contains preprocessed images and preprocessed instructions.
        """

        preprocessed_obss = torch_rl.DictList()

        if "image" in self.obs_space.keys():
            images = numpy.array([obs["image"] for obs in obss])
            images = torch.tensor(images, device=device, dtype=torch.float)

            preprocessed_obss.image = images

        if "carrying" in self.obs_space:
            carryings = numpy.array([obs["carrying"] for obs in obss])
            carryings = torch.tensor(carryings, device=device, dtype=torch.float)

            preprocessed_obss.carrying = carryings

        return preprocessed_obss
