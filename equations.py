from abc import ABC, abstractmethod

import numpy as np

from constants import Constants
from grid2d import Grid


class DyDtEquation(ABC):
    """Parent class for first order differential equation of form \frac{dy}{dt} = f(t, y)"""

    @abstractmethod
    def __call__(self, t, y):
        pass


class GreybodyConservationOfEnergyEquation(DyDtEquation):

    def __init__(self, final_time):
        self.final_time = final_time

    STEFAN_BOLTZMANN_CONSTANT = 5.67E-8  # units of W/(m^2 * K^4)
    SIGMA = STEFAN_BOLTZMANN_CONSTANT

    DEFAULT_SOLAR_CONSTANT = 1361  # units of W/m^2
    S_0 = DEFAULT_SOLAR_CONSTANT

    PLANET_RADIUS = 6.371E6  # units of meters
    R = PLANET_RADIUS

    EMISSIVITY = .78
    EPSILON = EMISSIVITY

    SILICON_HEAT_CAPACITY = 712  # units of J/(kg*K)
    C_S = SILICON_HEAT_CAPACITY

    SILICION_DENSITY = 2650  # units of kg/m^3
    RHO_S = SILICION_DENSITY

    PLANET_ALBEDO = .3
    ALPHA = PLANET_ALBEDO

    INITIAL_TEMPERATURE = 263.5  # units of kelvins
    T_0 = INITIAL_TEMPERATURE

    PLANET_HEAT_CAPACITY = 2.0437948107045676e27
    C_E = PLANET_HEAT_CAPACITY

    def calculate_solar_constant(self, t):
        rise = .3 * self.S_0
        run = self.final_time
        slope = rise / run
        y_intercept = .7 * self.S_0

        solar_constant = (slope * t) + y_intercept
        return solar_constant

    def __call__(self, t, T):
        solar_constant = self.calculate_solar_constant(t)

        dTdt = (((np.pi * (self.R ** 2)) / self.C_E) *
                (solar_constant * (1 - self.ALPHA) - ((1 - (self.EPSILON / 2)) * 4 * self.SIGMA * (T ** 4))))

        # returns in units of kelvin/second
        return dTdt

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

        def x_direction_diffusion(temperature: np.ndarray) -> np.ndarray:
            """Calculates temperature diffusion along the x-axis.  Returns a 2d numpy array
                of size (y_dimension, x_dimension) with values of diffusion (dT/dt) in units
                of [units here] for each grid midpoint position."""

            # temperature change across cells in the x direction
            dt_dx = (temperature[1:-1, 1:] - temperature[1:-1, :-1]) / self.grid.x_segment_length

            flux: np.ndarray = -1 * self.diffusion_coefficient * dt_dx

            incoming_shortwave_radiation = (Constants.SOLAR_IRRADIANCE / 4) * (1 - Constants.EARTH_ALBEDO)
            outgoing_longwave_radiation = (1 - Constants.EMISSIVITY / 2) * Constants.STEFAN_BOLTZMANN_CONSTANT * (
                    temperature[1:-1, 1:-1] ** 4)
            flux_derivative = (flux[:, 1:] - flux[:, :-1]) / self.grid.x_segment_length

            x_direction_diffusion = ((incoming_shortwave_radiation - outgoing_longwave_radiation - flux_derivative) /
                                     self.grid.heat_capacity)

            return x_direction_diffusion

        def y_direction_diffusion(temperature: np.ndarray) -> np.ndarray:
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

        return x_direction_diffusion(temperatures) + y_direction_diffusion(temperatures)





class XDirectionDiffusion(DyDtEquation):

    def __init__(self, diffusion_constant: float, grid: Grid):
        self.diffusion_constant = diffusion_constant
        self.grid = grid

    def __call__(self, time: float, temperature: np.ndarray) -> np.ndarray:
        """Calculates temperature diffusion on the x-axis.  Returns a 2d numpy array
        of diffusion (dT/dt) in units of [units here] for each x interface position."""

        # temperature change across cells in the x direction
        dt_dx = (temperature[1:-1, 1:] - temperature[1:-1, :-1]) / self.grid.x_segment_length

        flux: np.ndarray = -1 * self.diffusion_constant * dt_dx

        incoming_shortwave_radiation = (Constants.SOLAR_IRRADIANCE/4) * (1-Constants.EARTH_ALBEDO)
        outgoing_longwave_radiation = (1 - Constants.EMISSIVITY/2) * Constants.STEFAN_BOLTZMANN_CONSTANT * (temperature[1:-1, 1:-1]**4)
        x_direction_flux_term = (flux[:, 1:] - flux[:, :-1]) / self.grid.x_segment_length

        x_direction_diffusion = (incoming_shortwave_radiation - outgoing_longwave_radiation - x_direction_flux_term) / self.grid.heat_capacity

        return x_direction_diffusion


class XDiffusionNoSun(DyDtEquation):

    def __init__(self, diffusion_constant: float, grid: Grid):
        self.diffusion_constant = diffusion_constant
        self.grid = grid


    def __call__(self, time: float, temperature: np.ndarray) -> np.ndarray:

        dt_dx = (temperature[1:-1, 1:] - temperature[1:-1, :-1]) / self.grid.x_segment_length
        flux: np.ndarray = -1 * self.diffusion_constant * dt_dx

        flux_derivative: np.ndarray = (flux[:, 1:] - flux[:, :-1]) / self.grid.x_segment_length

        return -1 * flux_derivative/self.grid.heat_capacity

class YDirectionDiffusion(DyDtEquation):

    def __init__(self, diffusion_constant: float, grid: Grid):
        self.diffusion_constant = diffusion_constant
        self.grid = grid

    def __call__(self, time: float, temperature: np.ndarray) -> np.ndarray:
        """Calculates temperature diffusion on the y-axis.  Returns a 2d numpy array
            of diffusion in units of [units here] for each y interface position"""

        # temperature change across cells in the y direction
        dt_dy = (self.grid.current_temperatures[1:, 1:-1] - self.grid.current_temperatures[:-1, 1:-1]) / self.grid.y_segment_length

        y_direction_flux: np.ndarray = -1 * self.diffusion_constant * dt_dy

        incoming_shortwave_radiation = Constants.SOLAR_IRRADIANCE / 4 * (1 - Constants.EARTH_ALBEDO)
        outgoing_longwave_radiation = (1 - Constants.EMISSIVITY / 2) * Constants.STEFAN_BOLTZMANN_CONSTANT * temperature ** 4
        y_direction_flux_term = (y_direction_flux[1:, :] - y_direction_flux[:-1, :]) / self.grid.y_segment_length

        y_direction_diffusion = (incoming_shortwave_radiation - outgoing_longwave_radiation - y_direction_flux_term) / self.grid.heat_capacity

        return y_direction_diffusion


