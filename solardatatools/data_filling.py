"""Data Filling Module

This module contains functions for filling missing data in a PV power matrix

"""

from typing import cast
import numpy as np
import pandas as pd
from solardatatools.algorithms import SunriseSunset
import numpy.typing as npt


def zero_nighttime(
    data_matrix: npt.NDArray[np.float64],
    night_mask: npt.NDArray[np.bool_] | None = None,
    daytime_threshold: float = 0.005,
) -> npt.NDArray[np.float64]:
    D = np.copy(data_matrix)
    D[D < 0] = 0
    if night_mask is None:
        ss = SunriseSunset()
        ss.calculate_times(data_matrix, threshold=daytime_threshold)
        night_mask = cast(npt.NDArray[np.bool_], ~ss.sunup_mask_estimated)
    D[np.logical_and(night_mask, np.isnan(D))] = 0
    return D


def interp_missing(data_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    D = np.copy(data_matrix)
    D_df = pd.DataFrame(data=D)
    D = D_df.interpolate().values
    return D
