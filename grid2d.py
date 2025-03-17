from constants import Constants
import numpy as np
from scipy.integrate import solve_ivp

class Grid:


    def __init__(self, x_dimension: int, y_dimension: int):

        self.x_dimension = x_dimension
        self.y_dimension = y_dimension


        self.midpoints = self.create_grid_midpoints()
        self.x_intersection_points, self.y_intersection_points = self.create_grid_interface_points()

        self.x_segment_length = self.midpoints[0, 1] - self.midpoints[0, 0]
        self.y_segment_length = self.midpoints[1, 0] - self.midpoints[0, 0]

        self.temperatures = np.zeros((y_dimension+2, x_dimension+2))
        self.boundary_condition = None

    def create_grid_midpoints(self) -> np.ndarray:
        """Creates a grid storing coordinate points.  Includes coordinate points of the boundary."""

        # dimensions +2 to include the coordinates of the boundary cells
        grid_midpoints = np.zeros((self.y_dimension + 2, self.x_dimension + 2, 2))

        # fills the numpy array with coordinates of cell midpoints, including boundary cell midpoints
        for x_index in range(self.x_dimension + 2):
            grid_midpoints[:, x_index, 0] = x_index - (self.x_dimension + 1) / 2
            for y_index in range(self.y_dimension + 2):
                grid_midpoints[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension + 1) / 2)

        return grid_midpoints


    def create_grid_interface_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Creates two 2d numpy arrays.  One for coordinates of grid interface points on the x-axis,
            and one for grid interface points on the y-axis."""


        def create_x_interfaces() -> np.ndarray:

            grid_x_interfaces = np.zeros((self.y_dimension, self.x_dimension+1, 2))

            for x_index in range(self.x_dimension+1):
                grid_x_interfaces[:, x_index, 0] = x_index - (self.x_dimension/2)

                for y_index in range(self.y_dimension):
                    grid_x_interfaces[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension - 1) / 2)


            return grid_x_interfaces

        def create_y_interfaces() -> np.ndarray:

            grid_y_interfaces = np.zeros((self.y_dimension+1, self.x_dimension, 2))

            for x_index in range(self.x_dimension):
                grid_y_interfaces[:, x_index, 0] = x_index - (self.x_dimension - 1)/2

                for y_index in range(self.y_dimension+1):
                    grid_y_interfaces[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension/2))


            return grid_y_interfaces


        return create_x_interfaces(), create_y_interfaces()


    def set_initial_temperatures(self, temperatures: np.ndarray) -> None:
        """Sets initial temperature based on a 2d numpy array passed in.
            temperatures must be a 2d numpy array with size [y_dimension, x_dimension]"""

        #check that input is the right size
        if not temperatures.shape == (self.y_dimension, self.x_dimension):
            print("Input of initial temperatures is not the correct shape.")

        self.temperatures[1:-1, 1:-1] = temperatures

    def set_boundary(self, boundary: np.ndarray) -> None:
        """Sets the boundary conditions. Input must be a 2d numpy array
            with size (y_dimension+2, x_dimension+2)"""


        #check that the input is the right size
        if not boundary.shape == (self.y_dimension+2, self.x_dimension+2):
            print("Input of boundary is not the correct shape.")


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






