from statistics import mean
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn import metrics


class ConfusionMatrix(object):
    """
    Little Helper class for a cached computation of a confusion matrix
    """

    def __init__(self, size: int = 256, i_max: int = 100):
        """

        :param size: Number of classes
        :param i_max: number of rounds to cache before computation
        """
        self.mat = np.zeros((size, size))
        self.i_max, self.size = i_max, size
        self.i = 0
        self.t, self.y = np.array([]), np.array([])

    def add(self, y: torch.Tensor, t: torch.Tensor):
        """
        Add new results to the confusion matrix
        :param y: the predicted labels
        :param t: the ground truth labels
        :return:
        """
        self.i += 1
        np_y = y.detach().cpu().argmax(dim=1).numpy().flatten()
        np_t = t.cpu().numpy().flatten()
        self.t = np.append(self.t, np_t)
        self.y = np.append(self.y, np_y)

        if self.i == self.i_max:
            self.update()

    def update(self):
        if self.i == 0:
            return
        self.mat += metrics.confusion_matrix(self.t, self.y,
                                             labels=list(range(self.size)))
        self.t, self.y, self.i = np.array([]), np.array([]), 0

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots()
        cmap = 'viridis'
        sns.heatmap(self.mat, annot=True, ax=ax, cmap=cmap, robust=True)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


class MonkeyWriter(object):
    """
    Monkey-patched empty version of SummaryWriter of Tensorboard
    """

    def add_scalar(self, tag, val, it):
        pass

    def add_figure(self, tag, val, it):
        pass

    def add_histogram(self, tag, val, it):
        pass


def log(writer: MonkeyWriter, it: int, values: Dict):
    """
    Logs the given key value pairs. Writes to CLI and to a given writer.
    :param writer: The writer (SummaryWriter of TensorBoard)
    :param it: current global iteration
    :param values: Dict of tag⇒values to log
    """
    mess = f'it={it:>10}\t'

    for tag, val in values.items():
        if isinstance(val, float):
            mess += f'{tag}:{val:.3e}'
            writer.add_scalar(tag, val, it)

        if isinstance(val, plt.Figure):
            writer.add_figure(tag, val, it)

        if isinstance(val, list):
            writer.add_histogram(tag, np.array(val), it)
            mean_tag, mean_val = f'Mean {tag}', mean(val)
            mess += f'{mean_tag}:{mean_val:.3e}'
            writer.add_scalar(mean_tag, mean_val, it)
    print(mess)
