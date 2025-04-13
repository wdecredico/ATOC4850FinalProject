from abc import ABC, abstractmethod

import numpy as np

from constants import Constants
from grid3d import Grid3d

class DyDtEquation(ABC):
    """Parent class for first order differential equation of form \frac{dy}{dt} = f(t, y)"""

    @abstractmethod
    def __call__(self, t, y):
        pass


class Diffusion(DyDtEquation):

    def __init__(self, diffusion_coefficient: float, grid: Grid3d):
        self.diffusion_coefficient = diffusion_coefficient
        self.grid = grid

    def __call__(self, time: float, temperatures: np.ndarray):
        """Calculates the temperature change in units of kelvin/second for each cell midpoint of the inputted array.

                Parameters
                ----------
                time: function does not depend on time, function only takes it for consistency, input a float of whatever
                temperatures: A 3d numpy array of temperature at each grid cell. Must include boundary."""

        incoming_shortwave_radiation: float = (Constants.SOLAR_IRRADIANCE / 4) * (1 - Constants.EARTH_ALBEDO)
        outgoing_longwave_radiation: np.ndarray = ((1 - Constants.EMISSIVITY/2) * Constants.STEFAN_BOLTZMANN_CONSTANT *
                                                   (temperatures[1:-1, 1:-1, 1:-1] ** 4))

        def x_direction_diffusion() -> np.ndarray:

            derivative_dT_dx: np.ndarray = ((temperatures[1:, 1:-1, 1:-1] - temperatures[:-1, 1:-1, 1:-1]) /
                                            self.grid.x_axis_cell_length)

            flux: np.ndarray = -1 * self.diffusion_coefficient * derivative_dT_dx
            derivative_dF_dx: np.ndarray = (flux[1:, :, :] - flux[:-1, :, :]) / self.grid.x_axis_cell_length

            x_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - derivative_dF_dx) /
                                     self.grid.heat_capacity)

            return x_direction_diffusion
        def y_direction_diffusion() -> np.ndarray:

            derivative_dT_dy: np.ndarray = ((temperatures[1:-1, 1:, 1:-1] - temperatures[1:-1, :-1, 1:-1]) /
                                            self.grid.y_axis_cell_length)
            flux: np.ndarray = -1 * self.diffusion_coefficient * derivative_dT_dy
            derivative_dF_dy: np.ndarray = (flux[:, 1:, :] - flux[:, :-1, :]) / self.grid.y_axis_cell_length

            y_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - derivative_dF_dy) /
                                     self.grid.heat_capacity)

            return y_direction_diffusion
        def z_direction_diffusion() -> np.ndarray:

            derivative_dT_dz: np.ndarray = ((temperatures[1:-1, 1:-1, 1:] - temperatures[1:-1, 1:-1, :-1]) /
                                            self.grid.z_axis_cell_length)
            flux: np.ndarray = -1 * self.diffusion_coefficient * derivative_dT_dz
            derivative_dF_dz: np.ndarray = (flux[:, :, 1:] - flux[:, :, :-1]) / self.grid.z_axis_cell_length

            z_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - derivative_dF_dz) /
                                     self.grid.heat_capacity)

            return z_direction_diffusion

        return x_direction_diffusion() + y_direction_diffusion() + z_direction_diffusion()


class Advection(DyDtEquation):

    def __init__(self, grid: Grid3d):
        pass

    def __call__(self, time: float, winds: np.ndarray) -> np.ndarray:
        pass









