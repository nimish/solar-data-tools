import pytest
from pathlib import Path
import numpy as np
from pvsystemprofiler.algorithms.longitude.fitting import fit_longitude


@pytest.fixture
def longitude_fitting_and_calculation_test_data(
    fixtures_dir: Path,
) -> dict[str, np.ndarray]:
    # importing input for fit_longitude
    eot_duffie_file_path = fixtures_dir / "longitude" / "eot_duffie_output.csv"
    eot_duffie = np.genfromtxt(eot_duffie_file_path, delimiter=",")

    solarnoon_file_path = fixtures_dir / "longitude" / "solarnoon.csv"
    solarnoon = np.genfromtxt(solarnoon_file_path, delimiter=",")

    days_file_path = fixtures_dir / "longitude" / "days.csv"
    days = np.genfromtxt(days_file_path, delimiter=",")

    return {
        "eot_duffie_output": eot_duffie,
        "solarnoon": solarnoon,
        "days": days,
    }


def test_fit_longitude(
    longitude_fitting_and_calculation_test_data: dict[str, np.ndarray],
):
    # INPUTS
    gmt_offset = -5
    loss = "l2"
    eot_duffie = longitude_fitting_and_calculation_test_data["eot_duffie_output"]
    solarnoon = longitude_fitting_and_calculation_test_data["solarnoon"]
    days = longitude_fitting_and_calculation_test_data["days"]

    # Expected Longitude Output is generated in tests/fixtures/longitude/longitude_fitting_and_calculation_test_data_creator.ipynb
    expected_output = -77.22534574490635

    actual_output = fit_longitude(eot_duffie, solarnoon, days, gmt_offset, loss=loss)
    np.testing.assert_almost_equal(actual_output, expected_output, decimal=1)
