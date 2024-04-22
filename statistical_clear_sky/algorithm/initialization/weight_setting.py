"""
This module defines a class for Weight Setting Algorithm.
"""

import numpy as np
import numpy.typing as npt


class WeightSetting:
    """
    Delegate class.
    Weight Setting Algorithm:
    Two metrics are calculated and normalized to the interval [0, 1],
    and then the geometric mean is taken.
    Metric 1: daily smoothness
    Metric 2: seasonally weighted daily energy
    After calculating the geometric mean of these two values, weights below
    """

    def __init__(self, solver_type="ECOS"):
        self._solver_type = solver_type

    def obtain_weights(
        self, power_signals_d: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        try:
            from solardatatools.clear_day_detection import ClearDayDetection
        except ImportError:
            print("Weights not set!")
            print("Please make sure you have solar-data-tools installed")
            weights = np.ones(power_signals_d.shape[1])
        else:
            clear_day_detection = ClearDayDetection()
            weights = clear_day_detection.find_clear_days(
                power_signals_d, boolean_out=False
            )
        return weights
