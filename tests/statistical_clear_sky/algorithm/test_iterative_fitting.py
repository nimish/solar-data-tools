from pathlib import Path
import pytest
import numpy as np
import numpy.typing as npt
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting


data_subdir = Path("objective_calculation")

data_files = {
    "power_signals_d": "three_years_power_signals_d_1.csv",
    "weights": "three_years_weights.csv",
    "initial_l_cs_value": "three_years_initial_l_cs_value.csv",
    "initial_r_cs_value": "three_years_initial_r_cs_value.csv",
}


@pytest.fixture
def objective_calculation_data(
    fixtures_dir: Path,
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        k: np.loadtxt(fixtures_dir / data_subdir / v, delimiter=",")
        for k, v in data_files.items()
    }


def test_calculate_objective(
    objective_calculation_data: dict[str, npt.NDArray[np.float64]],
):
    power_signals_d = objective_calculation_data["power_signals_d"]
    l_cs_value = objective_calculation_data["initial_l_cs_value"]
    r_cs_value = objective_calculation_data["initial_r_cs_value"]
    weights = objective_calculation_data["weights"]

    rank_k = 6
    mu_l = 5e2
    mu_r = 1e3
    tau = 0.9
    beta_value = 0.0

    expected_objective_values = np.array(
        [
            117277.71151791142,
            478.8539994379723,
            23800125.708200675,
            228653.22102385858,
        ]
    )

    objective_calculation = IterativeFitting(power_signals_d, rank_k=rank_k)

    actual_objective_values = objective_calculation._calculate_objective(
        mu_l,
        mu_r,
        tau,
        l_cs_value,
        r_cs_value,
        beta_value,
        weights,
        sum_components=False,
    )

    np.testing.assert_almost_equal(
        actual_objective_values, expected_objective_values, decimal=8
    )
