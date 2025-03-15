from constants import Constants
import numpy as np
from scipy.integrate import solve_ivp

class Grid:


    def __init__(self, x_dimension: int, y_dimension: int):

        self.x_dimension = x_dimension
        self.y_dimension = y_dimension

        self.midpoints, self.interface_points = self.create_grid()

        self.x_segment_length = self.midpoints[0, 1] - self.midpoints[0, 0]
        self.y_segment_length = self.midpoints[1, 0] - self.midpoints[0, 0]

        self.temperatures = np.zeros((x_dimension+2, y_dimension+2))
        self.boundary_condition = None


    def create_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Creates a grid storing coordinate points. Includes coordinate points of boundary."""

        def create_midpoints() -> np.ndarray:

            #dimensions +2 to include the coordinates of the boundary cells
            grid_midpoints = np.zeros((self.x_dimension+2, self.y_dimension+2, 2))

            # fills the numpy array with coordinates of cell midpoints, including boundary cell midpoints
            for x_index in range(self.x_dimension+2):
                grid_midpoints[:, x_index, 0] = x_index - (self.x_dimension+2 - 1) / 2
                for y_index in range(self.y_dimension+2):
                    grid_midpoints[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension+2 - 1) / 2)

            return grid_midpoints

        def create_interface_points() -> np.ndarray:

            #dimensions +1 because we don't calculate exchange between boundary cells
            grid_interface_points = np.zeros((self.x_dimension + 1, self.y_dimension + 1, 2))

            # fills a numpy array with interface points between cells
            for x_index in range(self.x_dimension + 1):
                grid_interface_points[:, x_index, 0] = x_index - self.x_dimension / 2
                for y_index in range(self.y_dimension + 1):
                    grid_interface_points[y_index, x_index, 1] = -1 * y_index - self.y_dimension / 2

            return grid_interface_points

        midpoints = create_midpoints()
        interface_points = create_interface_points()

        return midpoints, interface_points


    def set_initial_temperatures(self, temperatures: np.ndarray) -> None:
        """Sets initial temperature based on a 2d numpy array passed in."""

        self.temperatures[1:-1, 1:-1] = temperatures

    def set_boundary(self, boundary: np.ndarray) -> None:

        self.temperatures[:, 0] = boundary[:, 0]
        self.temperatures[:, -1] = boundary[:, -1]
        self.temperatures[0, :] = boundary[0, :]
        self.temperatures[-1, :] = boundary[-1, :]


    def set_boundary_condition(self, boundary_condition: str) -> None:

        match boundary_condition.lower():
            case 'closed':
                self.boundary_condition = boundary_condition
            case 'dirichlet':
                self.boundary_condition = boundary_condition
            case 'neumann':
                self.boundary_condition = boundary_condition
            case 'radiation':
                self.boundary_condition = boundary_condition
            case 'periodic':
                self.boundary_condition = boundary_condition
            case 'nudging':
                self.boundary_condition = boundary_condition
            case _:
                self.boundary_condition = None

    def get_boundary_condition(self) -> str:
        return self.boundary_condition






