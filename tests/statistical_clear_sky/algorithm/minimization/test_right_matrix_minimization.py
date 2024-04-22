from pathlib import Path
import pytest
import numpy as np
from statistical_clear_sky.algorithm.minimization.right_matrix import (
    RightMatrixMinimization,
)


@pytest.fixture
def right_matrix_minimization_data_files(fixtures_dir: Path) -> dict[str, np.ndarray]:
    # importing input for minimize
    power_signals_d_file_path = (
        fixtures_dir / "right_matrix_minimization" / "three_years_power_signals_d_1.csv"
    )
    with open(power_signals_d_file_path) as file:
        power_signals_d = np.loadtxt(file, delimiter=",")
    weights_file_path = (
        fixtures_dir / "right_matrix_minimization" / "three_years_weights.csv"
    )
    with open(weights_file_path) as file:
        weights = np.loadtxt(file, delimiter=",")
    initial_l_cs_value_file_path = (
        fixtures_dir
        / "right_matrix_minimization"
        / "l_cs_value_after_left_matrix_minimization_iteration_1.csv"
    )
    with open(initial_l_cs_value_file_path) as file:
        initial_l_cs_value = np.loadtxt(file, delimiter=",")
    initial_r_cs_value_file_path = (
        fixtures_dir
        / "right_matrix_minimization"
        / "r_cs_value_after_left_matrix_minimization_iteration_1.csv"
    )
    with open(initial_r_cs_value_file_path) as file:
        initial_r_cs_value = np.loadtxt(file, delimiter=",")
    initial_component_r0_file_path = (
        fixtures_dir
        / "right_matrix_minimization"
        / "three_years_initial_component_r0.csv"
    )
    with open(initial_component_r0_file_path) as file:
        initial_component_r0 = np.loadtxt(file, delimiter=",")

    l_cs_value_after_right_matrix_minimization_iteration_1_file_path = (
        fixtures_dir
        / "right_matrix_minimization"
        / "l_cs_value_after_right_matrix_minimization_iteration_1.csv"
    )
    with open(l_cs_value_after_right_matrix_minimization_iteration_1_file_path) as file:
        expected_l_cs_value = np.loadtxt(file, delimiter=",")

    r_cs_value_after_right_matrix_minimization_iteration_1_file_path = (
        fixtures_dir
        / "right_matrix_minimization"
        / "r_cs_value_after_right_matrix_minimization_iteration_1.csv"
    )
    with open(r_cs_value_after_right_matrix_minimization_iteration_1_file_path) as file:
        expected_r_cs_value = np.loadtxt(file, delimiter=",")

    return {
        "power_signals_d": power_signals_d,
        "weights": weights,
        "initial_l_cs_value": initial_l_cs_value,
        "initial_r_cs_value": initial_r_cs_value,
        "initial_component_r0": initial_component_r0,
        "expected_l_cs_value": expected_l_cs_value,
        "expected_r_cs_value": expected_r_cs_value,
    }


def test_minimize_with_large_data(
    right_matrix_minimization_data_files: dict[str, np.ndarray],
):
    pytest.importorskip(
        "mosek",
        reason="MOSEK is not installed and this test is too slow with other solvers",
    )
    power_signals_d = right_matrix_minimization_data_files["power_signals_d"]
    weights = right_matrix_minimization_data_files["weights"]
    initial_l_cs_value = right_matrix_minimization_data_files["initial_l_cs_value"]
    initial_r_cs_value = right_matrix_minimization_data_files["initial_r_cs_value"]
    initial_component_r0_value = right_matrix_minimization_data_files[
        "initial_component_r0"
    ]
    expected_l_cs_value = right_matrix_minimization_data_files["expected_l_cs_value"]
    expected_r_cs_value = right_matrix_minimization_data_files["expected_r_cs_value"]
    rank_k = 6
    tau = 0.9
    mu_r = 1e3

    initial_beta_value = 0.0

    expected_beta_value = -0.04015762

    right_matrix_minimization = RightMatrixMinimization(
        power_signals_d, rank_k, weights, tau, mu_r, solver_type="MOSEK"
    )

    actual_l_cs_value, actual_r_cs_value, actual_beta_value = (
        right_matrix_minimization.minimize(
            initial_l_cs_value,
            initial_r_cs_value,
            initial_beta_value,
            initial_component_r0_value,
        )
    )
    np.testing.assert_array_almost_equal(
        actual_l_cs_value, expected_l_cs_value, decimal=2
    )
    np.testing.assert_array_almost_equal(
        actual_r_cs_value, expected_r_cs_value, decimal=1
    )
    np.testing.assert_almost_equal(actual_beta_value, expected_beta_value, decimal=4)
