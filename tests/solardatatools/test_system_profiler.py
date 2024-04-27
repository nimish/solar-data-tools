import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from solardatatools import DataHandler


@pytest.fixture
def system_profiler_data_files(fixtures_dir: Path) -> pd.DataFrame:
    data_file_path = fixtures_dir / "system_profiler" / "data_handler_input.csv"
    data = pd.read_csv(data_file_path, parse_dates=[1], index_col=1)
    return data


def test_system_profiler(system_profiler_data_files: pd.DataFrame):
    dh = DataHandler(system_profiler_data_files)
    dh.fix_dst()
    dh.run_pipeline(
        power_col="ac_power", fix_shifts=False, correct_tz=False, verbose=False
    )
    # dh.report()
    dh.setup_location_and_orientation_estimation(-5)

    estimate_latitude = dh.estimate_latitude()
    estimate_longitude = dh.estimate_longitude()

    _ = dh.estimate_orientation()

    # Based on site documentation
    actual_latitude = 39.4856
    actual_longitude = -76.6636

    estimate_orientation_real_loc = dh.estimate_orientation(
        latitude=actual_latitude, longitude=actual_longitude
    )

    # Current algorithms output after changes in PR. Note, this
    # changes slightly each run, depending on the results of the
    # sunrise sunset evaluation, which uses holdout validation
    # to pick some hyperparameters.
    ref_latitude = 37.5  # +/- 0.8
    ref_longitude = -77.0  # +/- 0.03
    ref_tilt_real_loc = 22.5  # +/- 0.015
    ref_az_real_loc = 0.28  # +/- 0.015

    # Updated tolerances based on new updates for sunset/sunrise SDs
    np.testing.assert_allclose(estimate_latitude, ref_latitude, atol=2)
    np.testing.assert_allclose(estimate_longitude, ref_longitude, atol=0.2)
    np.testing.assert_allclose(
        estimate_orientation_real_loc,
        (ref_tilt_real_loc, ref_az_real_loc),
        atol=0.5,
    )
