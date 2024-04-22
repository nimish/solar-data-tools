import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from solardatatools import standardize_time_axis, make_2d


@pytest.fixture
def data_transforms_files(fixtures_dir: Path) -> dict[str, pd.DataFrame]:
    # importing input for fit_longitude
    data_file_path = fixtures_dir / "data_transforms" / "timeseries.csv"
    data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
    timeseries_standardized_file_path = (
        fixtures_dir / "data_transforms" / "timeseries_standardized.csv"
    )
    timeseries_standardized = pd.read_csv(
        timeseries_standardized_file_path, index_col=0, parse_dates=True
    )
    power_mat_file_path = fixtures_dir / "data_transforms" / "power_mat.csv"
    with open(power_mat_file_path) as file:
        power_mat = np.genfromtxt(file, delimiter=",")
    return {
        "data": data,
        "timeseries_standardized": timeseries_standardized,
        "power_mat": power_mat,
    }


def test_standardize_time_axis(data_transforms_files: dict[str, pd.DataFrame]):
    data = data_transforms_files["data"]
    expected_output = data_transforms_files["timeseries_standardized"]
    actual_output, _ = standardize_time_axis(data, timeindex=True)
    np.testing.assert_array_almost_equal(expected_output, actual_output)


def test_make_2d_with_freq_set(data_transforms_files: dict[str, pd.DataFrame]):
    data = data_transforms_files["timeseries_standardized"]
    expected_output = data_transforms_files["power_mat"]
    data.index.freq = pd.tseries.offsets.Second(300)
    key = data.columns[0]
    actual_output = make_2d(data, key=key, trim_start=True, trim_end=True)
    assert isinstance(actual_output, np.ndarray)
    np.testing.assert_array_almost_equal(expected_output, actual_output)


def test_make_2d_no_freq(data_transforms_files: dict[str, pd.DataFrame]):
    data = data_transforms_files["timeseries_standardized"]
    expected_output = data_transforms_files["power_mat"]
    key = data.columns[0]
    actual_output = make_2d(data, key=key, trim_start=True, trim_end=True)
    assert isinstance(actual_output, np.ndarray)
    np.testing.assert_array_almost_equal(expected_output, actual_output)
