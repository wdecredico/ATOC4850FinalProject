import numpy as np
from constants import Constants
from grid2d import Grid
from abc import ABC, abstractmethod

class DyDtEquation(ABC):

    @abstractmethod
    def __call__(self, t, y):
        pass

class GreybodyConservationOfEnergyEquation(DyDtEquation):

    def __init__(self, final_time):
        self.final_time = final_time

    STEFAN_BOLTZMANN_CONSTANT = 5.67E-8 #units of W/(m^2 * K^4)
    SIGMA = STEFAN_BOLTZMANN_CONSTANT

    DEFAULT_SOLAR_CONSTANT = 1361  #units of W/m^2
    S_0 = DEFAULT_SOLAR_CONSTANT

    PLANET_RADIUS = 6.371E6 #units of meters
    R = PLANET_RADIUS

    EMISSIVITY = .78
    EPSILON = EMISSIVITY

    SILICON_HEAT_CAPACITY = 712 #units of J/(kg*K)
    C_S = SILICON_HEAT_CAPACITY

    SILICION_DENSITY = 2650  # units of kg/m^3
    RHO_S = SILICION_DENSITY

    PLANET_ALBEDO = .3
    ALPHA = PLANET_ALBEDO

    INITIAL_TEMPERATURE = 263.5  #units of kelvins
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


        dTdt = ((np.pi*(self.R**2))/self.C_E) * (solar_constant*(1-self.ALPHA) - ((1-(self.EPSILON/2))*4*self.SIGMA*(T**4)))

        #returns in units of kelvin/second
        return dTdt

class XDirectionDiffusion(DyDtEquation):

    def __init__(self, diffusion_constant: float, grid: Grid):

        self.diffusion_constant = diffusion_constant
        self.grid = grid

    def __call__(self, time, temperature):

        # temperature change across cells in the x direction
        dt_dx = np.zeros(self.grid.x_intersection_points.shape)
        dt_dx = (self.grid.temperatures[1:-1, 1:] - self.grid.temperatures[1:-1, :-1]) / self.grid.x_segment_length

        x_direction_flux = -1 * self.diffusion_constant * dt_dx

        incoming_shortwave_radiation = Constants.SOLAR_IRRADIANCE / 4 * (1 - Constants.EARTH_ALBEDO)
        outgoing_longwave_radiation = (1 - Constants.EMISSIVITY / 2) * Constants.STEFAN_BOLTZMANN_CONSTANT * temperature**4
        # x_direction_diffusion = -1 * x_direction_flux[]

        return

class DiffusionTemperatureChange(DyDtEquation):

    def __init__(self, diffusion_constant: float, grid: Grid):

        self.diffusion_constant = diffusion_constant
        self.grid = grid

    def solar_power(self, y):

        #figure out this later
        something = 4
        return something

    def __call__(self, time, temperature) -> np.ndarray:


        #temperature change across cells in the x direction
        dt_dx = np.zeros(self.grid.x_intersection_points.shape)
        dt_dx = (self.grid.temperatures[1:-1, 2:]  - self.grid.temperatures[1:-1, :-2]) / self.grid.x_segment_length



        #temperature change across cells in the y direction
        dt_dy = np.zeros(self.grid.y_intersection_points.shape)
        dt_dy[:] = (self.grid.temperatures[1:, :] - self.grid.temperatures[0:-1, :]) / self.grid.y_segment_length

        x_direction_flux = -1 * self.diffusion_constant * dt_dx
        y_direction_flux = -1 * self.diffusion_constant * dt_dy



        incoming_shortwave_radiation = Constants.SOLAR_IRRADIANCE/4 * (1-Constants.EARTH_ALBEDO)
        outgoing_longwave_radiation = (1 - Constants.EMISSIVITY/2) * Constants.STEFAN_BOLTZMANN_CONSTANT * temperature**4
        #x_direction_diffusion = -1 * x_direction_flux[]

        return





