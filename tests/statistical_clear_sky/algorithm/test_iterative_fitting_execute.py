from unittest.mock import Mock
import numpy as np
import numpy.typing as npt
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.algorithm.initialization.linearization_helper import (
    LinearizationHelper,
)
from statistical_clear_sky.algorithm.initialization.weight_setting import WeightSetting
from statistical_clear_sky.algorithm.minimization.left_matrix import (
    LeftMatrixMinimization,
)
from statistical_clear_sky.algorithm.minimization.right_matrix import (
    RightMatrixMinimization,
)


import pytest
from pathlib import Path

mock_subdir = Path("for_mock")

data_files = {
    "power_signals_d": "three_years_power_signals_d_1.csv",
    "initial_component_r0": "three_years_initial_component_r0.csv",
    "weights": "three_years_weights.csv",
    "clear_sky_signals": "three_years_clear_sky_signals.csv",
    **{
        f"l_cs_value_after_left_matrix_minimization_iteration_{i}": f"l_cs_value_after_left_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
    **{
        f"r_cs_value_after_left_matrix_minimization_iteration_{i}": f"r_cs_value_after_left_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
    **{
        f"beta_value_after_left_matrix_minimization_iteration_{i}": f"beta_value_after_left_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
    **{
        f"l_cs_value_after_right_matrix_minimization_iteration_{i}": f"l_cs_value_after_right_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
    **{
        f"r_cs_value_after_right_matrix_minimization_iteration_{i}": f"r_cs_value_after_right_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
    **{
        f"beta_value_after_right_matrix_minimization_iteration_{i}": f"beta_value_after_right_matrix_minimization_iteration_{i}.csv"
        for i in range(1, 14)
    },
}

data_files.update()


@pytest.fixture
def iter_fitting_mock(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64] | float]:
    mock_data = {
        k: np.loadtxt(fixtures_dir / mock_subdir / v, delimiter=",")
        for k, v in data_files.items()
    }

    mock_linearization_helper = Mock(spec=LinearizationHelper)
    mock_linearization_helper.obtain_component_r0.return_value = mock_data[
        "initial_component_r0"
    ]

    mock_weight_setting = Mock(spec=WeightSetting)
    mock_weight_setting.obtain_weights.return_value = mock_data["weights"]

    beta_value_left_matrix = [
        mock_data[f"beta_value_after_left_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]
    beta_value_right_matrix = [
        mock_data[f"beta_value_after_right_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]
    l_cs_value_left_matrix = [
        mock_data[f"l_cs_value_after_left_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]
    l_cs_value_right_matrix = [
        mock_data[f"l_cs_value_after_right_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]
    r_cs_value_left_matrix = [
        mock_data[f"r_cs_value_after_left_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]
    r_cs_value_right_matrix = [
        mock_data[f"r_cs_value_after_right_matrix_minimization_iteration_{i}"]
        for i in range(1, 14)
    ]

    mock_left_matrix_minimization = Mock(spec=LeftMatrixMinimization)
    mock_left_matrix_minimization.minimize.side_effect = list(
        zip(l_cs_value_left_matrix, r_cs_value_left_matrix, beta_value_left_matrix)
    )

    mock_right_matrix_minimization = Mock(spec=RightMatrixMinimization)
    mock_right_matrix_minimization.minimize.side_effect = list(
        zip(l_cs_value_right_matrix, r_cs_value_right_matrix, beta_value_right_matrix)
    )

    iter_fit = IterativeFitting(mock_data["power_signals_d"], rank_k=6)

    iter_fit.set_linearization_helper(mock_linearization_helper)
    iter_fit.set_weight_setting(mock_weight_setting)
    iter_fit.set_left_matrix_minimization(mock_left_matrix_minimization)
    iter_fit.set_right_matrix_minimization(mock_right_matrix_minimization)

    return iter_fit


@pytest.fixture
def expected_data(fixtures_dir: Path) -> dict[str, np.ndarray]:
    return {
        "clear_sky_signals": np.loadtxt(
            fixtures_dir / mock_subdir / "three_years_clear_sky_signals.csv",
            delimiter=",",
        ),
        "degradation_rate": -0.04069624,
    }


def test_iterative_fitting_execute(
    iter_fitting_mock: IterativeFitting,
    expected_data: dict[str, float | npt.NDArray[np.float64]],
):
    expected_clear_sky_signals = expected_data["clear_sky_signals"]
    expected_degradation_rate = expected_data["degradation_rate"]

    iter_fitting_mock.execute(
        mu_l=5e2, mu_r=1e3, tau=0.9, max_iteration=15, verbose=False
    )

    actual_clear_sky_signals = iter_fitting_mock.clear_sky_signals()
    actual_degradation_rate = iter_fitting_mock.degradation_rate()

    # Note: Discrepancy is due to the difference in Python 3.6 and 3.7.
    # np.testing.assert_array_equal(actual_clear_sky_signals,
    #                               expected_clear_sky_signals)
    np.testing.assert_almost_equal(
        actual_clear_sky_signals, expected_clear_sky_signals, decimal=13
    )
    # np.testing.assert_array_equal(actual_degradation_rate,
    #                               expected_degradation_rate)
    np.testing.assert_almost_equal(
        actual_degradation_rate, expected_degradation_rate, decimal=8
    )
