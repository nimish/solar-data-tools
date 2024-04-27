from pathlib import Path
import pytest
import numpy as np
import numpy.typing as npt
from statistical_clear_sky.algorithm.initialization.weight_setting import WeightSetting


@pytest.fixture
def weight_setting_data_files(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    # importing input for obtain_weights
    input_power_signals_file_path = (
        fixtures_dir / "initialization" / "one_year_power_signals_1.csv"
    )
    with open(input_power_signals_file_path) as file:
        power_signals_d = np.loadtxt(file, delimiter=",")
    weights_file_path = fixtures_dir / "initialization" / "one_year_weights_1.csv"
    with open(weights_file_path) as file:
        expected_weights = np.loadtxt(file, delimiter=",")

    return {
        "power_signals_d": power_signals_d,
        "expected_weights": expected_weights,
    }


def test_obtain_weights_with_large_data(
    weight_setting_data_files: dict[str, npt.NDArray[np.float64]],
):
    power_signals_d = weight_setting_data_files["power_signals_d"]
    expected_weights = weight_setting_data_files["expected_weights"]

    weight_setting = WeightSetting()
    actual_weights = weight_setting.obtain_weights(power_signals_d)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5)
