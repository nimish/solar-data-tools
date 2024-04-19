import unittest
import os
from pathlib import Path
import numpy as np

path = Path.cwd().parent.parent
os.chdir(path)
from pvsystemprofiler.algorithms.latitude.estimation import estimate_latitude


class TestEstimateLatitude(unittest.TestCase):
    def test_estimate_latitude(self):
        # INPUTS
        filepath = Path(__file__).parent.parent
        # hours daylight
        hours_daylight_file_path = (
            filepath / "fixtures" / "latitude" / "hours_daylight.csv"
        )
        hours_daylight = np.genfromtxt(hours_daylight_file_path, delimiter=",")
        # Delta
        delta_file_path = filepath / "fixtures" / "latitude" / "delta.csv"
        delta = np.genfromtxt(delta_file_path, delimiter=",")

        # Expected Latitude Output is generated in tests/fixtures/latitude/latitude_test_data_creator.ipynb
        expected_output = 38.58601372121755

        actual_output = estimate_latitude(hours_daylight, delta)
        np.testing.assert_almost_equal(actual_output, expected_output, decimal=1)


if __name__ == "__main__":
    unittest.main()
