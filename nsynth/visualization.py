import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, center=1 / 256):
    fig, ax = plt.subplots()
    cmap = 'viridis'
    sns.heatmap(cm, annot=True, ax=ax, center=center, cmap=cmap, robust=True)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
