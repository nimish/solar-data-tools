"""
This module is used to set the hour_angle_equation in terms of the unknowns. The hour equation is a function of the
declination (delta), the hour angle (omega) , latitude (phi), tilt (beta) and azimuth (gamma). The declination and the
hour angle are treated as input parameters for all cases. Latitude, tilt and azimuth can be given as input parameters
 or left as unknowns (`None`). In total, seven different combinations arise from having these three parameters
as an inputs or as a unknowns. The seven conditionals below correspond to those combinations. The output function `func`
is used as one of the inputs to run_curve_fit which in turn is used to fit the unknowns. The function other outputs is
the 'bounds' tuple containing the bounds for the variables. Bounds for latitude are -90 to 90. Bounds for tilt are 0 to
90. Bounds for azimuth  are -180 to 180. It is noted that, theoretically, bounds for tilt are 0 to 180 (Duffie, John A.,
 and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.). However a value of tilt >90
 would mean that that the surface has a downward-facing component, which is not the case of the current application.
"""
from typing import Callable
from pvsystemprofiler.utilities.angle_of_incidence_function import func_costheta
import numpy as np

from functools import partial


def select_function(
    latitude=None, tilt=None, azimuth=None
) -> tuple[Callable, tuple[list[float]]]:
    """
    :param latitude: (optional) latitude input value in Degrees.
    :param tilt: (optional) Tilt input value in Degrees.
    :param azimuth: (optional) Azimuth input value in Degrees.
    :return: Customized function 'func' and 'bounds' tuple.
    """
    func = func_costheta
    if latitude is not None:
        func = partial(func_costheta, phi=np.deg2rad(latitude))

    if tilt is not None:
        func = partial(func_costheta, beta=np.deg2rad(tilt))

    if azimuth is not None:
        func = partial(func_costheta, gamma=np.deg2rad(azimuth))

    bounds = []

    if latitude is None:
        bounds.append([-np.pi / 2, np.pi / 2])
    if tilt is None:
        bounds.append([0, np.pi / 2])
    if azimuth is None:
        bounds.append([-np.inf, np.inf])

    bounds = tuple(np.transpose(bounds).tolist())

    return func, bounds
