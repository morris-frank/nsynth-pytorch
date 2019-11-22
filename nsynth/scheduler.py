from bisect import bisect_right
from typing import List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class ManualMultiStepLR(_LRScheduler):
    """

    :param optimizer (Optimizer): Wrapped optimizer
    :param milestones (list): List of epoch indices. Must be increasing.
    :param gammas (list): List of learning rates at each milestone.
        Must be same length as milestones.
    :param last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, milestones: List[int],
                 gammas: List[float], last_epoch: int = -1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if len(milestones) != len(gammas):
            raise ValueError('Lists Milestones and gammas should'
                             ' have same length. Got {} and {}',
                             len(milestones), len(gammas))
        self.milestones = milestones
        self.gammas = gammas
        super(ManualMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gammas[bisect_right(self.milestones,
                                                   self.last_epoch)]
                for base_lr in self.base_lrs]
