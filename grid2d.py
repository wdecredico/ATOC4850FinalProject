import numpy as np



class Grid:

    def __init__(self, x_dimension: int, y_dimension: int):

        self.x_dimension = x_dimension
        self.y_dimension = y_dimension

        self.midpoints = self.create_grid_midpoints()
        self.x_intersection_points, self.y_intersection_points = self.create_grid_interface_points()

        self.x_segment_length = self.midpoints[0, 1, 0] - self.midpoints[0, 0, 0]
        self.y_segment_length = self.midpoints[0, 0, 1] - self.midpoints[1, 0, 1]

        # Current temperatures including boundary cells
        self.current_temperatures = np.zeros((y_dimension + 2, x_dimension + 2))

        self.initial_temperatures = np.zeros((y_dimension + 2, x_dimension + 2))
        self.boundary = np.zeros((y_dimension + 2, x_dimension + 2))

        self.boundary_condition: str = ""
        self.heat_capacity: np.ndarray = np.zeros((y_dimension, x_dimension))

    def set_temperatures(self, temperatures: np.ndarray) -> None:
        """Sets the inside of the temperature array. Leaves the boundary unchanged

        Parameter 'temperatures' must be 2d numpy array of size (y_dimension, x_dimension)"""

        #check for correct dimension
        if not temperatures.shape == (self.y_dimension, self.x_dimension):
            print("In set_temperatures(), temperatures was not of the right shape")

        self.current_temperatures[1:-1, 1:-1] = temperatures

    def update_temperatures(self, temperature_change: np.ndarray) -> None:
        """Updates the current temperatures by adding input values 'temperature' to the current temperatures

            Parameter 'temperatures' must be a 2d numpy array with size (y_dimension, x_dimension)"""

        if not temperature_change.shape == (self.y_dimension, self.x_dimension):
            print("In update_temperatures(), temperatures was not of the correct shape.")

        self.current_temperatures[1:-1, 1:-1] += temperature_change

    def remove_boundary(self, temperatures: np.ndarray) -> np.ndarray:
        """Removes the boundary cells, and returns a 2d numpy array of size (y_dimension, x_dimension) containing the temperatures of the cells"""

        return temperatures[1:-1, 1:-1]

    def add_boundary(self, temperatures: np.ndarray) -> np.ndarray:
        """Adds boundary cells to a 2d numpy array without it.
        
        Parameter
        ---------
        temperatures: 2d numpy array with dimensions (y_dimension, x_dimension)
        
        Return
        ------
        2d numpy array with dimensions (y_dimension+2, x_dimension+2)"""

        if not temperatures.shape == (self.y_dimension, self.x_dimension):
            print("In add_boundary(), temperatures was not of the right shape.")

        temperatures_with_boundary = self.current_temperatures
        temperatures_with_boundary[1:-1, 1:-1] = temperatures

        return temperatures_with_boundary


    def set_initial_temperatures(self, temperatures: np.ndarray) -> None:
        """Sets initial temperature based on a 2d numpy array passed in.
            temperatures must be a 2d numpy array with size [y_dimension, x_dimension]"""

        # check that input is the right size
        if not temperatures.shape == (self.y_dimension, self.x_dimension):
            print("Input of initial temperatures is not the correct shape.")

        self.current_temperatures[1:-1, 1:-1] = temperatures
        self.initial_temperatures[1:-1, 1:-1] = temperatures

    def set_boundary(self, boundary: np.ndarray) -> None:
        """Sets the boundary conditions. Input must be a 2d numpy array
            with size (y_dimension+2, x_dimension+2)"""

        # check that the input is the right size
        if not boundary.shape == (self.y_dimension + 2, self.x_dimension + 2):
            print("Input of boundary is not the correct shape.")

        self.current_temperatures[:, 0] = boundary[:, 0]
        self.current_temperatures[:, -1] = boundary[:, -1]
        self.current_temperatures[0, :] = boundary[0, :]
        self.current_temperatures[-1, :] = boundary[-1, :]

        self.boundary[:, 0] = boundary[:, 0]
        self.boundary[:, -1] = boundary[:, -1]
        self.boundary[0, :] = boundary[0, :]
        self.boundary[-1, :] = boundary[-1, :]


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

    def set_heat_capacity(self, heat_capacity: np.ndarray) -> None:
        """Takes in a 2d numpy array for what the heat capacity is at each cell."""

        # check that input is the right size
        if not heat_capacity.shape == (self.y_dimension, self.x_dimension):
            print(f"Heat Capacity array is the wrong size.")
            print(f"Grid dimension: ({self.y_dimension}, {self.x_dimension})")
            print(f"Heat capacity array: {heat_capacity.shape}")
            quit()

        self.heat_capacity = heat_capacity

    def create_grid_midpoints(self) -> np.ndarray:
        """Creates a grid storing coordinate points.  Includes coordinate points of the boundary."""

        # Check that dimensions are above zero
        if self.x_dimension <= 0 or self.y_dimension <= 0:
            print(f"Grid dimensions must be above zero.")

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

            grid_x_interfaces = np.zeros((self.y_dimension, self.x_dimension + 1, 2))

            for x_index in range(self.x_dimension + 1):
                grid_x_interfaces[:, x_index, 0] = x_index - (self.x_dimension / 2)

                for y_index in range(self.y_dimension):
                    grid_x_interfaces[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension - 1) / 2)

            return grid_x_interfaces

        def create_y_interfaces() -> np.ndarray:

            grid_y_interfaces = np.zeros((self.y_dimension + 1, self.x_dimension, 2))

            for x_index in range(self.x_dimension):
                grid_y_interfaces[:, x_index, 0] = x_index - (self.x_dimension - 1) / 2

                for y_index in range(self.y_dimension + 1):
                    grid_y_interfaces[y_index, x_index, 1] = -1 * (y_index - (self.y_dimension / 2))

            return grid_y_interfaces

        return create_x_interfaces(), create_y_interfaces()

