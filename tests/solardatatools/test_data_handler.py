import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.typing as npt
from solardatatools import DataHandler


@pytest.fixture
def scoring_data_files(fixtures_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    # importing input for fit_longitude
    data_file_path = fixtures_dir / "data_transforms" / "timeseries.csv"
    data = pd.read_csv(data_file_path, parse_dates=[0], index_col=0)
    clipping_1_file_path = fixtures_dir / "scoring" / "clipping_1.csv"
    clipping_1 = np.genfromtxt(clipping_1_file_path, delimiter=",")
    clipping_2_file_path = fixtures_dir / "scoring" / "clipping_2.csv"
    clipping_2 = np.genfromtxt(clipping_2_file_path, delimiter=",")
    density_file_path = fixtures_dir / "scoring" / "density.csv"
    density = np.genfromtxt(density_file_path, delimiter=",")
    linearity_file_path = fixtures_dir / "scoring" / "linearity.csv"
    linearity = np.genfromtxt(linearity_file_path, delimiter=",")
    quality_clustering_file_path = fixtures_dir / "scoring" / "quality_clustering.csv"
    quality_clustering = np.genfromtxt(quality_clustering_file_path, delimiter=",")

    return {
        "data": data,
        "clipping_1": clipping_1,
        "clipping_2": clipping_2,
        "density": density,
        "linearity": linearity,
        "quality_clustering": quality_clustering,
    }


def test_load_and_run(scoring_data_files: dict[str, npt.NDArray[np.float64]]):
    df = scoring_data_files["data"]
    dh = DataHandler(df)
    dh.fix_dst()
    dh.run_pipeline(power_col="ac_power_01", fix_shifts=True, verbose=False)
    # dh.report()
    np.testing.assert_allclose(dh.capacity_estimate, 6.7453649044036865)
    np.testing.assert_allclose(dh.data_quality_score, 0.9948186528497409)
    np.testing.assert_allclose(dh.data_clearness_score, 0.49222797927461137)
    assert dh.inverter_clipping
    assert not dh.time_shifts
    expected_scores = scoring_data_files["clipping_1"]
    np.testing.assert_allclose(dh.daily_scores.clipping_1, expected_scores, atol=1e-3)
    expected_scores = scoring_data_files["clipping_2"]
    np.testing.assert_allclose(dh.daily_scores.clipping_2, expected_scores, atol=2e-3)

    expected_scores = scoring_data_files["density"]
    np.testing.assert_allclose(dh.daily_scores.density, expected_scores, atol=1e-3)

    expected_scores = scoring_data_files["linearity"]
    np.testing.assert_allclose(dh.daily_scores.linearity, expected_scores, atol=2e-2)

    expected_scores = scoring_data_files["quality_clustering"]
    np.testing.assert_allclose(
        dh.daily_scores.quality_clustering, expected_scores, atol=1e-3
    )
