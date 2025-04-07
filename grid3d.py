import numpy as np

class Grid3d:

    def __init__(self, num_grid_cells_x_axis: int, num_grid_cells_y_axis: int, num_grid_cells_z_axis: int,
                        x_axis_cell_length: float, y_axis_cell_length: float, z_axis_cell_length: float):

        self.num_x_cells = num_grid_cells_x_axis
        self.num_y_cells = num_grid_cells_y_axis
        self.num_z_cells = num_grid_cells_z_axis

        self.x_axis_cell_length = x_axis_cell_length
        self.y_axis_cell_length = y_axis_cell_length
        self.z_axis_cell_length = z_axis_cell_length

        (self.x,
         self.y,
         self.z) = self._create_midpoints()

        (self.x_interface_points,
         self.y_interface_points,
         self.z_interface_points) = self._create_interface_points()

        # Temperature arrays include boundaries. Arrays are in order of cartesian coordinates, (x, y, z)
        self.initial_temperatures = np.zeros((self.num_x_cells+2, self.num_y_cells+2, self.num_z_cells+2))
        self.current_temperatures = np.zeros((self.num_x_cells+2, self.num_y_cells+2, self.num_z_cells+2))

        self.boundary = np.zeros((self.num_x_cells+2, self.num_y_cells+2, self.num_z_cells+2))

        # Heat capacity array does not include boundary
        self.heat_capacity = np.zeros((self.num_x_cells, self.num_y_cells, self.num_z_cells))


    def set_initial_temperatures(self, temperatures: np.ndarray) -> None:

        if self._input_array_not_equal_grid_dimensions(temperatures):
            print("In set_initial_temperatures, inputted array was not the correct shape")

        self.initial_temperatures[1:-1, 1:-1, 1:-1] = temperatures
        self.current_temperatures[1:-1, 1:-1, 1:-1] = temperatures
    def set_boundary(self, boundary_conditions: np.ndarray) -> None:

        if self._input_array_not_equal_grid_dimensions(boundary_conditions):
            print("In set_boundary, inputted array was not the correct shape")

        self.current_temperatures[:, :, 0] = boundary_conditions[:, :, 0]
        self.current_temperatures[:, :, -1] = boundary_conditions[:, :, -1]
        self.current_temperatures[:, 0, :] = boundary_conditions[:, 0, :]
        self.current_temperatures[:, -1, :] = boundary_conditions[:, -1, :]
        self.current_temperatures[0, :, :] = boundary_conditions[0, :, :]
        self.current_temperatures[-1, :, :] = boundary_conditions[-1, :, :]


        self.boundary[:, :, 0] = boundary_conditions[:, :, 0]
        self.boundary[:, :, -1] = boundary_conditions[:, :, -1]
        self.boundary[:, 0, :] = boundary_conditions[:, 0, :]
        self.boundary[:, -1, :] = boundary_conditions[:, -1, :]
        self.boundary[0, :, :] = boundary_conditions[0, :, :]
        self.boundary[-1, :, :] = boundary_conditions[-1, :, :]
    def set_heat_capacity(self, heat_capacity: np.ndarray) -> None:

        if self._input_array_not_equal_grid_dimensions(heat_capacity):
            print("In set_heat_capacity, inputted array was not the correct shape")

        self.heat_capacity = heat_capacity


    def set_temperatures(self, temperatures: np.ndarray) -> None:
        """Sets the temperatures according to the input array. Leaves the boundary unchanged."""

        if self._input_array_not_equal_grid_dimensions(temperatures):
            print("In set_temperatures, inputted array was not the correct shape")


        self.current_temperatures[1:-1, 1:-1, 1:-1] = temperatures
    def update_temperatures(self, temperature_change: np.ndarray) -> None:
        """Updates the temperatures by adding the input temperatures to the current temperatures"""

        if self._input_array_not_equal_grid_dimensions(temperature_change):
            print("In update_temperatures, inputted array was not the correct shape")

        self.current_temperatures[1:-1, 1:-1, 1:-1] += temperature_change

    def remove_temperature_boundary(self) -> np.ndarray:
        """Removes the boundary from the temperature array.
            Returns a 3d numpy array of size (num_x_cells, num_y_cells, num_z_cells)
            with the temperature of each cell."""

        return self.current_temperatures[1:-1, 1:-1, 1:-1]

    def _input_array_not_equal_grid_dimensions(self, array) -> bool:
        """Checks if an array is not the same shape as the grid dimensions.
            If the inputted array is not the same shape as the grid dimensions, return True.
            If the inputted array is the same shape as the grid dimensions, return False."""

        if array.shape != (self.num_x_cells, self.num_y_cells, self.num_z_cells):
            return True
        else:
            return False
    def _invalid_grid_shape(self) -> bool:
        """Checks is the grid is a valid size, (number of cells and cell length above zero)"""

        if (self.num_x_cells and self.num_y_cells and self.num_z_cells and
                self.x_axis_cell_length and self.y_axis_cell_length and self.z_axis_cell_length > 0):
            return False
        else:
            return True
    def _create_midpoints(self):
        """Creates the midpoint coordinates of the grid. Returns a list of 3 3d numpy arrays.
            One for each axis of grid. all together, grid makes 3d coordinates in (x, y, z)"""


        if self._invalid_grid_shape():
            print("Grid is not of valid size")
            quit()

        low_x_coordinate = -1 * (self.num_x_cells - 1)/2 * self.x_axis_cell_length
        high_x_coordinate = (self.num_x_cells - 1)/2 * self.x_axis_cell_length

        low_y_coordinate = -1 * (self.num_y_cells - 1)/2 * self.y_axis_cell_length
        high_y_coordinate = (self.num_y_cells - 1)/2 * self.y_axis_cell_length

        low_z_coordinate = self.z_axis_cell_length/2
        high_z_coordinate = (self.z_axis_cell_length*self.num_z_cells) - (self.z_axis_cell_length/2)

        x_axis = np.linspace(low_x_coordinate, high_x_coordinate, self.num_x_cells)
        y_axis = np.linspace(low_y_coordinate, high_y_coordinate, self.num_y_cells)
        z_axis = np.linspace(low_z_coordinate, high_z_coordinate, self.num_z_cells)

        coordinates = np.meshgrid(x_axis, y_axis, z_axis, indexing='xy')

        return coordinates
    def _create_interface_points(self):

        if self._invalid_grid_shape():
            print("Grid is not of valid size")
            quit()

        def create_x_interface_points():


            low_x_coordinate = -1 * self.num_x_cells / 2 * self.x_axis_cell_length
            high_x_coordinate = self.num_x_cells / 2 * self.x_axis_cell_length

            low_y_coordinate = -1 * (self.num_y_cells-1) / 2 * self.y_axis_cell_length
            high_y_coordinate = (self.num_y_cells-1) / 2 * self.y_axis_cell_length

            low_z_coordinate = self.z_axis_cell_length / 2
            high_z_coordinate = (self.z_axis_cell_length * self.num_z_cells) - (self.z_axis_cell_length / 2)

            x_axis = np.linspace(low_x_coordinate, high_x_coordinate, self.num_x_cells)
            y_axis = np.linspace(low_y_coordinate, high_y_coordinate, self.num_y_cells)
            z_axis = np.linspace(low_z_coordinate, high_z_coordinate, self.num_z_cells)

            x_interface_coordinates = np.meshgrid(x_axis, y_axis, z_axis, indexing='xy')

            return x_interface_coordinates

        def create_y_interface_points():

            low_x_coordinate = -1 * (self.num_x_cells - 1) / 2 * self.x_axis_cell_length
            high_x_coordinate = (self.num_x_cells - 1) / 2 * self.x_axis_cell_length

            low_y_coordinate = -1 * self.num_y_cells/2 * self.y_axis_cell_length
            high_y_coordinate = self.num_y_cells/2 * self.y_axis_cell_length

            low_z_coordinate = self.z_axis_cell_length / 2
            high_z_coordinate = (self.z_axis_cell_length * self.num_z_cells) - (self.z_axis_cell_length / 2)

            x_axis = np.linspace(low_x_coordinate, high_x_coordinate, self.num_x_cells)
            y_axis = np.linspace(low_y_coordinate, high_y_coordinate, self.num_y_cells)
            z_axis = np.linspace(low_z_coordinate, high_z_coordinate, self.num_z_cells)

            y_interface_coordinates = np.meshgrid(x_axis, y_axis, z_axis, indexing='xy')
            return y_interface_coordinates

        def create_z_interface_points():

            low_x_coordinate = -1 * (self.num_x_cells - 1) / 2 * self.x_axis_cell_length
            high_x_coordinate = (self.num_x_cells - 1) / 2 * self.x_axis_cell_length

            low_y_coordinate = -1 * (self.num_y_cells - 1) / 2 * self.y_axis_cell_length
            high_y_coordinate = (self.num_y_cells - 1) / 2 * self.y_axis_cell_length

            low_z_coordinate = 0
            high_z_coordinate = self.z_axis_cell_length * self.num_z_cells

            x_axis = np.linspace(low_x_coordinate, high_x_coordinate, self.num_x_cells)
            y_axis = np.linspace(low_y_coordinate, high_y_coordinate, self.num_y_cells)
            z_axis = np.linspace(low_z_coordinate, high_z_coordinate, self.num_z_cells)

            z_interface_coordinates = np.meshgrid(x_axis, y_axis, z_axis, indexing='xy')

            return z_interface_coordinates

        x_interface_points = create_x_interface_points()
        y_interface_points = create_y_interface_points()
        z_interface_points = create_z_interface_points()

        return x_interface_points, y_interface_points, z_interface_points




