import pytest

from pathlib import Path
import numpy as np
import numpy.typing as npt
from solardatatools.data_filling import zero_nighttime, interp_missing


@pytest.fixture
def data_filling_test_data(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    # importing input for zero_nighttime
    data_file_path = fixtures_dir / "data_filling" / "pvdaq_2d_data_input.csv"
    data = np.genfromtxt(data_file_path, delimiter=",")

    # importing expected output for zero_nighttime
    expected_output_file_path = (
        fixtures_dir / "data_filling" / "expected_zero_nighttime_output.csv"
    )
    expected_output = np.genfromtxt(expected_output_file_path, delimiter=",")

    # importing expected output for interp_missing
    expected_output_file_path = (
        fixtures_dir / "data_filling" / "expected_interp_missing_output.csv"
    )
    expected_output_interp_missing = np.genfromtxt(
        expected_output_file_path, delimiter=","
    )

    return {
        "expected_output": expected_output,
        "expected_output_interp_missing": expected_output_interp_missing,
        "data": data,
    }


def test_zero_nighttime(data_filling_test_data: dict[str, npt.NDArray[np.float64]]):
    input_data = data_filling_test_data["data"]
    expected_output = data_filling_test_data["expected_output"]

    actual_output = zero_nighttime(input_data)
    np.testing.assert_array_almost_equal(actual_output, expected_output)


def test_interp_missing(data_filling_test_data: dict[str, npt.NDArray[np.float64]]):
    # using zero_nighttime expected output as interp_missing() input
    input_data = data_filling_test_data["expected_output"]

    expected_output = data_filling_test_data["expected_output_interp_missing"]

    actual_output = interp_missing(input_data)
    np.testing.assert_array_almost_equal(actual_output, expected_output)
