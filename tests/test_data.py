import os

import pandas as pd
import pytest

from thesis import config
from thesis.data import build_train_test

needs_raw_data = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [*config.DATASTREAM_EXPORTS, config.LATER_EXPORT,
                                        config.TRAIN_PATH, config.TEST_PATH]),
    reason="raw data files are not in the repository",
)


@needs_raw_data
def test_train_and_test_are_rebuilt_exactly_from_raw_files():
    train, test = build_train_test()
    expected_train = pd.read_csv(config.TRAIN_PATH, index_col="Date", parse_dates=True)
    expected_test = pd.read_csv(config.TEST_PATH, index_col="Date", parse_dates=True)
    pd.testing.assert_frame_equal(train, expected_train, check_exact=True, check_freq=False)
    pd.testing.assert_frame_equal(test, expected_test, check_exact=True, check_freq=False)
