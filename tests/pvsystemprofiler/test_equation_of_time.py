import pytest
from pathlib import Path
import numpy as np
from pvsystemprofiler.utilities.equation_of_time import eot_da_rosa, eot_duffie


@pytest.fixture
def eot_data_files(fixtures_dir: Path) -> dict[str, np.ndarray]:
    # importing input for both eot tests
    input_data_file_path = fixtures_dir / "longitude" / "eot_input.csv"
    input_data = np.genfromtxt(input_data_file_path, delimiter=",")

    # importing expected output for both eot tests
    expected_data_file_path = fixtures_dir / "longitude" / "eot_duffie_output.csv"
    expected_output_duffie = np.genfromtxt(expected_data_file_path, delimiter=",")

    expected_data_file_path = fixtures_dir / "longitude" / "eot_da_rosa_output.csv"
    expected_output_da_rosa = np.genfromtxt(expected_data_file_path, delimiter=",")

    return {
        "input_data": input_data,
        "expected_output_duffie": expected_output_duffie,
        "expected_output_da_rosa": expected_output_da_rosa,
    }


def test_eot_duffie(eot_data_files: dict[str, np.ndarray]):
    actual_output = eot_duffie(eot_data_files["input_data"])
    np.testing.assert_array_almost_equal(
        actual_output, eot_data_files["expected_output_duffie"]
    )


def test_eot_da_rosa(eot_data_files: dict[str, np.ndarray]):
    actual_output = eot_da_rosa(eot_data_files["input_data"])
    np.testing.assert_array_almost_equal(
        actual_output, eot_data_files["expected_output_da_rosa"]
    )
