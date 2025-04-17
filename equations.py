from abc import ABC, abstractmethod

import numpy as np

from constants import Constants
from grid2d import Grid


def solar_irradiance(solar_zenith_angle: float):

    optical_depth = .1

    solar_irradiance_at_ground = Constants.SOLAR_IRRADIANCE * np.exp(-1*optical_depth/np.cos(solar_zenith_angle))

    return solar_irradiance_at_ground


class DyDtEquation(ABC):
    """Parent class for first order differential equation of form \frac{dy}{dt} = f(t, y)"""

    @abstractmethod
    def __call__(self, t, y):
        pass

class Diffusion(DyDtEquation):

    def __init__(self, diffusion_coefficient: float, grid: Grid):
        self.diffusion_coefficient = diffusion_coefficient
        self.grid = grid

    def __call__(self, time: float, temperatures: np.ndarray) -> np.ndarray:
        """Calculates the temperature change in units of [units here] for each cell midpoint of the inputted array.

        Parameters
        ----------
        time: function does not depend on time, function only takes it for consistency, input a float of whatever
        temperatures: A 2d numpy array of temperature at each grid cell. Must include boundary."""

        def x_direction_diffusion() -> np.ndarray:
            """Calculates temperature diffusion along the x-axis.  Returns a 2d numpy array
                of size (y_dimension, x_dimension) with values of diffusion (dT/dt) in units
                of [units here] for each grid midpoint position."""

            # temperature change across cells in the x direction
            dt_dx = (temperatures[1:-1, 1:] - temperatures[1:-1, :-1]) / self.grid.x_segment_length

            flux: np.ndarray = -1 * self.diffusion_coefficient * dt_dx

            incoming_shortwave_radiation = (Constants.SOLAR_IRRADIANCE / 4) * (1 - Constants.EARTH_ALBEDO)
            outgoing_longwave_radiation = (1 - Constants.EMISSIVITY / 2) * Constants.STEFAN_BOLTZMANN_CONSTANT * (
                    temperatures[1:-1, 1:-1] ** 4)
            flux_derivative = (flux[:, 1:] - flux[:, :-1]) / self.grid.x_segment_length

            x_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - flux_derivative) /
                                     self.grid.heat_capacity)

            return x_direction_diffusion

        def y_direction_diffusion() -> np.ndarray:
            """Calculates temperature diffusion along the y-axis.  Returns a 2d numpy array
            of size (y_dimension, x_dimension) with values of diffusion (dT/dt) in units
            of [units here] for each grid midpoint position."""

            # temperature change across cells in the y direction
            dt_dy = (temperatures[1: , 1:-1] - temperatures[:-1, 1:-1]) / self.grid.y_segment_length

            flux: np.ndarray = -1 * self.diffusion_coefficient * dt_dy

            incoming_shortwave_radiation = (Constants.SOLAR_IRRADIANCE / 4) * (1 - Constants.EARTH_ALBEDO)
            outgoing_longwave_radiation = ((1 - Constants.EMISSIVITY / 2) * Constants.STEFAN_BOLTZMANN_CONSTANT *
                                           (temperatures[1:-1, 1:-1] ** 4))
            flux_derivative = (flux[1:, :] - flux[:-1, :]) / self.grid.y_segment_length

            x_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - flux_derivative) /
                                     self.grid.heat_capacity)

            return x_direction_diffusion

        return x_direction_diffusion() + y_direction_diffusion()



