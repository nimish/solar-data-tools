import pytest
from pathlib import Path
import numpy as np
from solardatatools.clear_day_detection import ClearDayDetection


@pytest.fixture
def clear_day_detection_test_data(fixtures_dir: Path) -> dict[str, np.ndarray]:
    # importing input for fit_longitude
    data_file_path = (
        fixtures_dir / "clear_day_detection" / "one_year_power_signals_1.csv"
    )
    data = np.genfromtxt(data_file_path, delimiter=",")

    # importing input for fit_longitude
    expected_output_file_path = (
        fixtures_dir / "clear_day_detection" / "one_year_weights_1.csv"
    )
    expected_output = np.genfromtxt(expected_output_file_path, delimiter=",")

    return {
        "expected_output": expected_output,
        "data": data,
    }


def test_find_clear_days(clear_day_detection_test_data: dict[str, np.ndarray]):
    data = clear_day_detection_test_data["data"]
    expected_output = clear_day_detection_test_data["expected_output"]
    expected_output = expected_output >= 1e-3

    clear_day_detection = ClearDayDetection()
    actual_output = clear_day_detection.find_clear_days(data)

    np.testing.assert_array_equal(expected_output, actual_output)


def test_clear_day_weights(clear_day_detection_test_data: dict[str, np.ndarray]):
    data = clear_day_detection_test_data["data"]
    expected_output = clear_day_detection_test_data["expected_output"]

    clear_day_detection = ClearDayDetection()
    actual_output = clear_day_detection.find_clear_days(data, boolean_out=False)

    np.testing.assert_array_almost_equal(expected_output, actual_output, 4)
