"""
This module defines a class that holds the current state of algorithm object.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StateData:
    """
    Holds the data to be serialized.
    """

    beta_value: float = np.nan
    mu_l: float = np.nan
    mu_r: float = np.nan
    tau: float = np.nan
    auto_fix_time_shifts: bool = False
    power_signals_d: np.ndarray = np.array([])
    rank_k: int = 0
    matrix_l0: np.ndarray = np.array([])
    matrix_r0: np.ndarray = np.array([])
    l_value: np.ndarray = np.array([])
    r_value: np.ndarray = np.array([])
    component_r0: np.ndarray = np.array([])
    is_solver_error: bool = False
    is_problem_status_error: bool = False
    f1_increase: bool = False
    obj_increase: bool = False
    residuals_median: float = np.nan
    residuals_variance: float = np.nan
    residual_l0_norm: float = np.nan
    weights: np.ndarray = np.array([])
