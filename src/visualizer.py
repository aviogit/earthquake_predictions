import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_plot_data(data: pd.core.frame.DataFrame, step: int = 1000):
    figure, axis1 = plt.subplots()
    x_axis = np.arange(0, len(data), step)

    axis1.plot(x_axis, data.iloc[:, 0][0::step], '-b')
    axis1.set_ylabel('sequence')
    axis1.set_ylabel('seismic activity', color='b')

    axis2 = axis1.twinx()
    axis2.plot(x_axis, data.iloc[:, 1][0::step], '-r')
    axis2.set_ylabel('time to failure', color='r')
    plt.savefig('../plots/summary.png', bbox_inches='tight')


def save_plot_loss(model, target: str = 'loss'):
    x = model.history.history[target]
    val_x = model.history.history['val_' + target]
    epochs = np.asarray(model.history.epoch) + 1

    plt.plot(epochs, x, 'bo', label="Training " + target)
    plt.plot(epochs, val_x, 'b', label="Validation " + target)
    plt.title("Training and validation " + target)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('../plots/Test results.png', bbox_inches='tight')
