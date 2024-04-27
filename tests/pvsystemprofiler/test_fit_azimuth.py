import pytest
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pvsystemprofiler.algorithms.angle_of_incidence.lambda_functions import (
    select_function,
)
from pvsystemprofiler.algorithms.angle_of_incidence.curve_fitting import run_curve_fit


@pytest.fixture
def tilt_data_files(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    # importing input for fit_tilt
    # INPUTS
    # delta_f
    delta_f_file_path = fixtures_dir / "tilt_azimuth" / "delta_f.csv"
    delta_f = np.genfromtxt(delta_f_file_path, delimiter=",")
    # omega_f
    omega_f_file_path = fixtures_dir / "tilt_azimuth" / "omega_f.csv"
    omega_f = np.genfromtxt(omega_f_file_path, delimiter=",")
    # costheta_fit
    costheta_fit_file_path = fixtures_dir / "tilt_azimuth" / "costheta_fit.csv"
    costheta_fit = np.genfromtxt(costheta_fit_file_path, delimiter=",")
    # boolean_filter
    boolean_filter_file_path = fixtures_dir / "tilt_azimuth" / "boolean_filter.csv"
    boolean_filter = np.genfromtxt(boolean_filter_file_path, delimiter=",")
    boolean_filter = boolean_filter.astype(dtype=bool)

    return {
        "delta_f": delta_f,
        "omega_f": omega_f,
        "costheta_fit": costheta_fit,
        "boolean_filter": boolean_filter,
    }


def test_fit_azimuth(tilt_data_files: dict[str, npt.NDArray[np.float64]]):
    # INPUTS
    # keys
    keys = ["tilt", "azimuth"]
    # init_values
    init_values = [30, 30]

    delta_f = tilt_data_files["delta_f"]
    omega_f = tilt_data_files["omega_f"]
    costheta_fit = tilt_data_files["costheta_fit"]
    boolean_filter = tilt_data_files["boolean_filter"]

    # Expected Tilt and azimuth output is generated in tests/fixtures/tilt_azimuth/tilt_azimuth_Estimation_data_creator.ipynb
    expected_output = 1.654457422429566
    func_customized, bounds = select_function(39.4856, None, None)
    actual_output = run_curve_fit(
        func=func_customized,
        keys=keys,
        delta=delta_f,
        omega=omega_f,
        costheta=costheta_fit,
        boolean_filter=boolean_filter,
        init_values=init_values,
        fit_bounds=bounds,
    )[1]
    np.testing.assert_almost_equal(actual_output, expected_output, decimal=4)
