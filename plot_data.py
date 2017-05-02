import matplotlib.pyplot as plt
import numpy as np

def plot_precision_recall(plot_labels=[], precisions=[], recalls=[]):
    for i in range(0, len(plot_labels)):
        plt.plot(recalls[i], precisions[i], color=np.random.rand(3, 1),
                 lw=2.5, label=plot_labels[i])
    plt.plot([0, 1], [0, 1], color='red', lineStyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.show()