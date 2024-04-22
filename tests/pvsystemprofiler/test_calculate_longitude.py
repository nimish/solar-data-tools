import numpy as np
from pathlib import Path
import pytest
from pvsystemprofiler.algorithms.longitude.calculation import calculate_longitude


@pytest.fixture
def longitude_fitting_and_calculation_test_data(
    fixtures_dir: Path,
) -> dict[str, np.ndarray]:
    # importing input for fit_longitude
    # INPUTS
    # eot_duffie
    eot_duffie_file_path = fixtures_dir / "longitude" / "eot_duffie_output.csv"
    eot_duffie = np.genfromtxt(eot_duffie_file_path, delimiter=",")
    # solarnoon
    solarnoon_file_path = fixtures_dir / "longitude" / "solarnoon.csv"
    solarnoon = np.genfromtxt(solarnoon_file_path, delimiter=",")
    # days
    days_file_path = fixtures_dir / "longitude" / "days.csv"
    days = np.genfromtxt(days_file_path, delimiter=",")
    days = days.astype(dtype=bool)

    return {
        "eot_duffie": eot_duffie,
        "solarnoon": solarnoon,
        "days": days,
    }


def test_calculate_longitude(
    longitude_fitting_and_calculation_test_data: dict[str, np.ndarray],
):
    # gmt_offset
    gmt_offset = -5
    # loss

    # Expected Longitude Output is generated in tests/fixtures/longitude/longitude_fitting_and_calculation_test_data_creator.ipynb
    expected_output = -77.10636729272031
    eot_duffie = longitude_fitting_and_calculation_test_data["eot_duffie"]
    solarnoon = longitude_fitting_and_calculation_test_data["solarnoon"]
    days = longitude_fitting_and_calculation_test_data["days"]

    actual_output = calculate_longitude(eot_duffie, solarnoon, days, gmt_offset)
    np.testing.assert_almost_equal(actual_output, expected_output, decimal=1)
