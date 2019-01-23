import pytest
import pandas as pd
import numpy as np
import mock

from preprocessor import get_stat_summaries, split_sequence, _export_sequence
from open_mock import get_open_mock


def _approx_001(x: np.float64):
    return pytest.approx(x, 0.01)


def _assert_stat_properties(summary: pd.core.frame.DataFrame):
    assert len(summary) == 1

    assert 7.6667 == _approx_001(summary.loc[0, 'mean'])
    assert 15.42 == _approx_001(summary.loc[0, 'std'])
    assert -13 == summary.loc[0, 'min']
    assert 24 == summary.loc[0, 'max']
    assert 16.33 == _approx_001(summary.loc[0, 'abs_mean'])
    assert 5.439 == _approx_001(summary.loc[0, 'abs_std'])
    assert 12.0 == _approx_001(summary.loc[0, 'abs_min'])
    assert 24.0 == _approx_001(summary.loc[0, 'abs_max'])
    assert 24.0 == _approx_001(summary.loc[0, 'q95'])
    assert 24.0 == _approx_001(summary.loc[0, 'q99'])
    assert -13.0 == _approx_001(summary.loc[0, 'q05'])
    assert -13.0 == _approx_001(summary.loc[0, 'q01'])
    assert 7.66 == _approx_001(summary.loc[0, 'std_first5k'])
    assert 15.42 == _approx_001(summary.loc[0, 'mean_first5k'])
    assert -13.0 == _approx_001(summary.loc[0, 'min_first5k'])
    assert 24.0 == _approx_001(summary.loc[0, 'max_first5k'])


def get_test_frame():
    frame = pd.DataFrame(dtype=np.float64, columns=['data', 'time_to_failure'])

    for i in range(0, 1000, 3):
        frame.loc[i, :] = pd.Series({'data': 12, 'time_to_failure': 14.9}).values
        frame.loc[i + 1, :] = pd.Series({'data': -13, 'time_to_failure': 14.8}).values
        frame.loc[i + 2, :] = pd.Series({'data': 24, 'time_to_failure': 14.7}).values

    return frame


@pytest.fixture
def data_frame_xy():
    return get_test_frame()


@pytest.fixture
def data_frame_x():
    data_column = get_test_frame().iloc[:, 0]
    return pd.DataFrame(data_column)


@pytest.fixture
def open_mock():
    return get_open_mock('data, time_to_failure\n27, 0.1\n-3, 0\n15, 0.1\n-13, 0\n14, 1')


def test_get_stat_summaries_with_default_params(data_frame_xy: pd.core.frame.DataFrame):
    summary = get_stat_summaries(data_frame_xy)

    _assert_stat_properties(summary)
    assert 14.7 == summary.loc[0, 'time_to_failure']


def test_get_stat_summaries_without_y(data_frame_x: pd.core.frame.DataFrame):
    summary = get_stat_summaries(data_frame_x, include_y=False)

    _assert_stat_properties(summary)

    with pytest.raises(Exception):
        assert 14.7 == summary.loc[0, 'time_to_failure']


def test_split_sequence(open_mock):
    export_mock = mock.MagicMock()
    with mock.patch('preprocessor._export_sequence', export_mock):
        with mock.patch('builtins.open', open_mock):
            split_sequence('mock_file')

    assert export_mock.call_count == 2
    assert open_mock.call_count == 1
