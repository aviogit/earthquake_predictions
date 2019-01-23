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
