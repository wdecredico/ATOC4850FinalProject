# third party imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Local application imports
from constants import Constants
from grid2d import Grid
import equations







def area_runge_kutta_calculator(grid: Grid, function: equations.DyDtEquation, time_step: float, num_steps: int) -> np.ndarray:
    """Uses the 4th order runge kutta method to solve a differential equation over an area that includes a boundary.

        Parameters:
        -----------
        grid: Grid object
        function: 1st order differential equation of type DyDtEquation. returns 2d numpy array of (dT/dt) at each grid cell midpoint
        initial_values: 2d numpy array with size (y_dimension, x_dimension) containing the initial value of what we are calculating at each grid cell
        time_step: float, length of the time step in seconds
        num_steps: int, how many steps to take"""

    time = 0
    y_values: np.ndarray = np.zeros((grid.y_dimension + 2, grid.x_dimension + 2, num_steps + 1))
    y_values[:, :, 0] = grid.current_temperatures

    for step in range(1, num_steps + 1):

        k1: np.ndarray = function(time, grid.current_temperatures)

        intermediate_temperature_storage: np.ndarray = grid.current_temperatures.copy()
        intermediate_temperature_storage[1:-1, 1:-1] += time_step *(k1[1:-1, 1:-1]/2)

        k2: np.ndarray = function((time + .5*time_step), intermediate_temperature_storage)

        intermediate_temperature_storage: np.ndarray = grid.current_temperatures.copy()
        intermediate_temperature_storage[1:-1, 1:-1] += time_step * (k2[1:-1, 1:-1] / 2)

        k3: np.ndarray = function((time + .5*time_step), (grid.current_temperatures + k2*(time_step/2)))

        intermediate_temperature_storage: np.ndarray = grid.current_temperatures.copy()
        intermediate_temperature_storage[1:-1, 1:-1] += time_step * k2[1:-1, 1:-1]

        k4: np.ndarray = function((time + time_step), (grid.current_temperatures + k3*time_step))

        temperature_change = (grid.current_temperatures * (1/6) * (k1 + 2*k2 + 2*k3 + k4) * time_step)

        grid.update_temperatures(grid.remove_boundary(temperature_change))

        y_values[:, :, step] = grid.current_temperatures

    return y_values


def area_euler_method_calculator(grid: Grid, function: equations.DyDtEquation, time_step: float, num_steps: int) -> np.ndarray:

    time = 0
    y_values: np.ndarray = np.zeros((grid.y_dimension+2, grid.x_dimension+2, num_steps+1))
    y_values[:, :, 0] = grid.current_temperatures.copy()

    for step in range(1, num_steps+1):

        grid.update_temperatures(time_step * function(time, grid.current_temperatures))
        y_values[:, :, step] = grid.current_temperatures

    return y_values


"""Set model parameters, constants and settings:"""
grid_x_dimension: int = 6
grid_y_dimension: int = 4

grid_cell_size: float = 1

diffusion_constant: float = .1

# Format: (low temperature (inclusive), high temperature (exclusive), (shape))
initial_temperatures: np.ndarray = np.random.randint(278, 283, (grid_y_dimension, grid_x_dimension))

boundary_condition: str = 'dirichlet'
boundary: np.ndarray = np.full((grid_y_dimension + 2, grid_x_dimension + 2),
                               280)  # Constant boundary temperature of 280K

silicon_heat_capacity = Constants.SILICON_SPECIFIC_HEAT * Constants.SILICON_ROCK_DENSITY * grid_cell_size
heat_capacity: np.ndarray = np.full((grid_y_dimension, grid_x_dimension), silicon_heat_capacity)

# change the model timing
end_time: float = 5 * Constants.YEARS
time_step: float = 60
model_times: np.ndarray = np.arange(0, end_time, time_step)
num_steps: int = 5





grid = Grid(grid_x_dimension, grid_y_dimension)

grid.set_initial_temperatures(initial_temperatures)
grid.set_boundary_condition(boundary_condition)
grid.set_boundary(boundary)
grid.set_heat_capacity(heat_capacity)

x_diffusion = equations.XDirectionDiffusion(diffusion_constant, grid)
y_diffusion = equations.YDirectionDiffusion(diffusion_constant, grid)

# x_direction_diffusion: np.ndarray = area_runge_kutta_calculator(grid, x_diffusion, time_step, num_steps)
# x_direction_diffusion: np.ndarray = area_euler_method_calculator(grid, x_diffusion, time_step, num_steps)

diffusion_equation = equations.Diffusion(diffusion_constant, grid)
diffusion_temperature_change = area_euler_method_calculator(grid, diffusion_equation, time_step, num_steps)


no_sun_diffusion = equations.XDiffusionNoSun(diffusion_constant, grid)

x_direction_diffusion: np.ndarray = area_euler_method_calculator(grid, no_sun_diffusion, time_step, num_steps)

literally_just_a_breakpoint = 0









# #create the tick marks on the x and y axis
# x_axis: np.ndarray = np.linspace((-1 * (grid_x_dimension-1)/2), (grid_x_dimension-1)/2, grid_x_dimension)
# y_axis: np.ndarray = np.linspace((grid_y_dimension-1)/2, -1 * (grid_y_dimension-1)/2, grid_y_dimension)
#
# #choose where the ticks go
# y_segment_length = grid.y_segment_length
# x_axis_tick_locations: np.ndarray = np.arange(0, grid_x_dimension, grid.x_segment_length)
# y_axis_tick_locations: np.ndarray = np.arange(0, grid_y_dimension, grid.y_segment_length)
#
# x_coordinates, y_coordinates = np.meshgrid(x_axis, y_axis)
#
# z = grid.current_temperatures[1:-1, 1:-1]
#
#
# fig, ax = plt.subplots()
#
# #set tick marks
# ax.set_xticks(x_axis_tick_locations)
# ax.set_xticklabels(x_axis)
# ax.set_yticks(y_axis_tick_locations)
# ax.set_yticklabels(y_axis)
# temperature_data = ax.imshow(grid.current_temperatures[1:-1, 1:-1], cmap='RdBu')
#
# #colorbar
# colorbar = fig.colorbar(temperature_data)


