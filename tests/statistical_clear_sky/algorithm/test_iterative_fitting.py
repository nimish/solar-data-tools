from pathlib import Path
import pytest
import numpy as np
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting


@pytest.fixture
def objective_calculation_data(fixtures_dir: Path) -> dict[str, np.ndarray]:
    # importing input for minimize
    power_signals_d_file_path = (
        fixtures_dir / "objective_calculation" / "three_years_power_signals_d_1.csv"
    )
    with open(power_signals_d_file_path) as file:
        power_signals_d = np.loadtxt(file, delimiter=",")
    weights_file_path = (
        fixtures_dir / "objective_calculation" / "three_years_weights.csv"
    )
    with open(weights_file_path) as file:
        weights = np.loadtxt(file, delimiter=",")
    initial_l_cs_value_file_path = (
        fixtures_dir / "objective_calculation" / "three_years_initial_l_cs_value.csv"
    )
    with open(initial_l_cs_value_file_path) as file:
        initial_l_cs_value = np.loadtxt(file, delimiter=",")
    initial_r_cs_value_file_path = (
        fixtures_dir / "objective_calculation" / "three_years_initial_r_cs_value.csv"
    )
    with open(initial_r_cs_value_file_path) as file:
        initial_r_cs_value = np.loadtxt(file, delimiter=",")
    return {
        "power_signals_d": power_signals_d,
        "weights": weights,
        "initial_l_cs_value": initial_l_cs_value,
        "initial_r_cs_value": initial_r_cs_value,
    }


def test_calculate_objective(objective_calculation_data: dict[str, np.ndarray]):
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
