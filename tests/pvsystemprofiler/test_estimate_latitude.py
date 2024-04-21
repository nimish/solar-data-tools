import pytest
from pathlib import Path
import numpy as np
from pvsystemprofiler.algorithms.latitude.estimation import estimate_latitude


@pytest.fixture
def estimate_latitude_data(fixtures_dir: Path) -> dict[str, np.ndarray]:
    # importing input for estimate_latitude
    hours_daylight_file_path = fixtures_dir / "latitude" / "hours_daylight.csv"
    hours_daylight = np.genfromtxt(hours_daylight_file_path, delimiter=",")

    # importing expected output for estimate_latitude
    delta_file_path = fixtures_dir / "latitude" / "delta.csv"
    delta = np.genfromtxt(delta_file_path, delimiter=",")

    return {
        "hours_daylight": hours_daylight,
        "delta": delta,
    }


def test_estimate_latitude(estimate_latitude_data: dict[str, np.ndarray]):
    # Expected Latitude Output is generated in tests/fixtures/latitude/latitude_test_data_creator.ipynb
    expected_output = 38.58601372121755

    actual_output = estimate_latitude(
        estimate_latitude_data["hours_daylight"], estimate_latitude_data["delta"]
    )
    np.testing.assert_almost_equal(actual_output, expected_output, decimal=1)
