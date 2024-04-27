import pytest
from pathlib import Path
import numpy as np
import numpy.typing as npt
from solardatatools.algorithms import TimeShift


@pytest.fixture
def time_shifts_data_files(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    # importing input for fix_time_shifts
    input_power_signals_file_path = (
        fixtures_dir / "time_shifts" / "two_year_signal_with_shift.csv"
    )
    with open(input_power_signals_file_path) as file:
        power_data_matrix = np.loadtxt(file, delimiter=",")
    use_days_file_path = fixtures_dir / "time_shifts" / "clear_days.csv"
    with open(use_days_file_path) as file:
        use_days = np.loadtxt(file, delimiter=",")
    output_power_signals_file_path = (
        fixtures_dir / "time_shifts" / "two_year_signal_fixed.csv"
    )
    with open(output_power_signals_file_path) as file:
        power_data_fix = np.loadtxt(file, delimiter=" ")

    return {
        "power_data_matrix": power_data_matrix,
        "use_days": use_days,
        "power_data_fix": power_data_fix,
    }


def test_fix_time_shifts(time_shifts_data_files: dict[str, npt.NDArray[np.float64]]):
    power_data_matrix = time_shifts_data_files["power_data_matrix"]
    use_days = time_shifts_data_files["use_days"]
    expected_power_data_fix = time_shifts_data_files["power_data_fix"]

    time_shift_analysis = TimeShift()
    time_shift_analysis.run(power_data_matrix, use_ixs=use_days, w1=100, solver="QSS")
    actual_power_data_fix = time_shift_analysis.corrected_data

    np.testing.assert_almost_equal(
        actual_power_data_fix, expected_power_data_fix, decimal=3
    )
