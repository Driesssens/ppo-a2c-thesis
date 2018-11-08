import numpy
import torch

from torch_rl.algos.base import BaseAlgo


class I2Algorithm(BaseAlgo):
    def __init__(self, environment_class, n_processes=16, seed=1, acmodel=None, num_frames_per_proc=None, discount=0.99,
                 lr=7e-4, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(environment_class, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, n_processes, seed)

        # self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.agent_optimizer = None
        self.imagination_policy_optimizer = None

    def load_acmodel(self, acmodel):
        super().load_acmodel(acmodel)

        self.agent_optimizer = torch.optim.RMSprop(self.acmodel.parameters(), self.lr, alpha=0.99, eps=1e-5)
        self.imagination_policy_optimizer = torch.optim.Adam(self.acmodel.imagination_policy.parameters(), lr=self.lr)

    def update_parameters(self):
        # Collect experiences
        exps, logs = self.collect_experiences()

        # Initialize update values
        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Compute loss
        dist, value = self.acmodel(exps.obs)

        entropy = dist.entropy().mean()
        policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()
        value_loss = (value - exps.returnn).pow(2).mean()
        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

        # Update batch values
        update_entropy += entropy.item()
        update_value += value.mean().item()
        update_policy_loss += policy_loss.item()
        update_value_loss += value_loss.item()
        update_loss += loss

        # Update update values
        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic
        self.agent_optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.agent_optimizer.step()

        self.imagination_policy_optimizer.zero_grad()
        distilled_distributions, _, _ = self.acmodel.imagination_policy(exps.obs, None)
        distillation_loss = (-1 * distilled_distributions.logits * dist.probs.detach()).sum(dim=1).mean()
        distillation_loss.backward()
        self.imagination_policy_optimizer.step()

        # Log some values
        logs["entropy"] = update_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm
        logs["distillation_loss"] = distillation_loss.item()

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
