import mock


def get_open_mock(lines):
    opener_mock = mock.mock_open(read_data=lines)
    opener_mock.return_value.__iter__ = lambda self: iter(self.readline, '')
    opener_mock.return_value.__next__ = lambda self: next(iter(self.readline, ''))

    return opener_mock
