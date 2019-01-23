import pytest
import pandas as pd
import numpy as np
import mock

from visualizer import save_plot_data


def get_test_frame():
    frame = pd.DataFrame(dtype=np.float64, columns=['data', 'time_to_failure'])

    for i in range(0, 1000, 3):
        frame.loc[i, :] = pd.Series({'data': 12, 'time_to_failure': 14.9}).values
        frame.loc[i + 1, :] = pd.Series({'data': -13, 'time_to_failure': 14.8}).values
        frame.loc[i + 2, :] = pd.Series({'data': 24, 'time_to_failure': 14.7}).values

    return frame


@pytest.fixture
def data_frame():
    return get_test_frame()


def test_save_plot_data_throws_with_none_data():
    with mock.patch('matplotlib.pyplot.savefig') as savefig_mock:
        with pytest.raises(Exception):
            save_plot_data(None)


def test_save_plot_data_works(data_frame: pd.core.frame.DataFrame):
    with mock.patch('matplotlib.pyplot.savefig') as savefig_mock:
        save_plot_data(data_frame)

    assert savefig_mock.call_count == 1
