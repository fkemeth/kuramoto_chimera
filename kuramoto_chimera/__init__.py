"""
Copyright 2022 Felix P. Kemeth

This file is part of the program kuramoto_chimera.

kuramoto_chimera is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

kuramoto_chimera is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with kuramoto_chimera.  If not, see <http://www.gnu.org/licenses/>.

Integrate Kuramoto phase oscillator model with nonlocal coupling..
"""

###############################################################################
#                                                                             #
# http://dx.doi.org/10.1063/1.4959804                                         #
#                                                                             #
# Jun 2022                                                                    #
# felix@kemeth.de                                                             #
#                                                                             #
###############################################################################

from time import time
import numpy as np
import sys
from typing import Tuple


def initial_conditions(num_grid_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Specify initial conditions for the phases in intrinsic frequencies.

    :param num_grid_points: number of spatial grid points
    :return: tuple of initial phases and frequencies per oscillator
    """
    omega = np.zeros(num_grid_points)
    for k in range(0, num_grid_points):
        omega[k] = 0
    phi = np.zeros(num_grid_points)
    for k in range(0, num_grid_points):
        phi[k] = 6.0 * np.exp(-30.0 * (float(k) / float(num_grid_points) - 0.5)
                              ** 2) * (np.random.random() - 0.5)
    return phi, omega


def calc_dy(kappa: float, num_grid_points: int) -> np.ndarray:
    """
    Calculate the distance kernel delta_y.

    :param kappa: parameter kappa
    :param num_grid_points: number of spatial grid points
    :return: numpy array containing the coupling kernel
    """
    delta_y = np.zeros(num_grid_points)
    for k in range(0, num_grid_points):
        delta_y[k] = (kappa / 2.0) / (1.0 - np.exp(-kappa / 2.0)) * \
            np.exp(-kappa * (min(num_grid_points - k, k)) /
                   float(num_grid_points)) / float(num_grid_points)
    return delta_y


def calc_coupling(alpha: float, delta_y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Calculate the coupling for the given snapshot.

    :param alpha: parameter alpha
    :param delta_y: numpy array containing the coupling kernel
    :param phi: numpy array containing the phases of the snapshot
    :return: numpay array containing the coupling
    """
    coupling = 1.0 / (2.0 * 1.0j) * \
        (np.exp(1.0j * alpha + 1.0j * phi) *
         np.fft.ifft(np.fft.fft(delta_y) *
                     np.fft.fft(np.exp(-1j * phi))) - np.exp(-1.0j * alpha - 1.0j * phi) *
         np.fft.ifft(np.fft.fft(delta_y) * np.fft.fft(np.exp(1j * phi))))
    return coupling


def f_kuramoto(alpha: float, delta_y: np.ndarray, omega: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Calculate the temporal derivative.

    :param alpha: parameter alpha
    :param delta_y: numpy array containing the coupling kernel
    :param omega: numpy array containing the frequencies of the snapshot
    :param phi: numpy array containing the phases of the snapshot
    :return: numpay array containing the time derivatives of the phases
    """
    return omega - calc_coupling(alpha, delta_y, phi).real


def integrate(kappa: float = 4.0, alpha: float = 1.457, delta_t: float = 0.01,
              tmin: float = 500, tmax: float = 1000, num_time_steps: int = 1000,
              num_grid_points: int = 200) -> dict:
    """
    Integrate the Kuramoto model with nonlocal coupling using 4th order Runge-Kutta.

    :param alpha: parameter alpha
    :param kappa: parameter kappa
    :param delta_t: delta t used for integration
    :param tmin: time at which to start collecting simulation data
    :param tmax: time until which to integrate and collect data
    :param num_time_steps: number of equally-spaced snapshots to collect between tmin and tmax
    :param num_grid_points: number of spatial grid points
    :return: dictionary containing parameters and simulation data

    """
    tstart = time()

    # Write the parameters into a dictionary for future use.
    data_dict = dict()
    data_dict["alpha"] = alpha
    data_dict["kappa"] = kappa
    data_dict["L"] = 1.0
    data_dict["N"] = num_grid_points
    data_dict["tmin"] = tmin
    data_dict["tmax"] = tmax
    data_dict["dt"] = delta_t
    data_dict["T"] = num_time_steps

    nmax = int(tmax / delta_t)
    n_above_thresh = int((tmax - tmin) / delta_t)
    # Threshold itself
    n_thresh = nmax - n_above_thresh
    # Every nplt'th step is plotted
    nplt = n_above_thresh / num_time_steps

    (phi, omega) = initial_conditions(num_grid_points)
    delta_y = calc_dy(kappa, num_grid_points)

    data_dict["init"] = phi

    save_data = list()

    for i in np.arange(1, nmax + 1):
        k_1 = f_kuramoto(alpha, delta_y, omega, phi)
        k_2 = f_kuramoto(alpha, delta_y, omega, phi + 0.5 * delta_t * k_1)
        k_3 = f_kuramoto(alpha, delta_y, omega, phi + 0.5 * delta_t * k_2)
        k_4 = f_kuramoto(alpha, delta_y, omega, phi + delta_t * k_3)
        phi = phi + delta_t / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
        phi[np.where(phi > np.pi)] = phi[np.where(phi > np.pi)] - 2 * np.pi
        phi[np.where(phi < -np.pi)] = phi[np.where(phi < -np.pi)] + 2 * np.pi

        if (i > n_thresh) and (i % nplt == 0):
            save_data.append(phi)

        if i % (np.floor(nmax / 100)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i)) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()

    data_dict["data"] = np.array(save_data)
    return data_dict
