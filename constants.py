import numpy as np

class Constants:

    EARTH_RADIUS = 6.371E6  # in meters
    EMISSIVITY = 0.82
    STEFAN_BOLTZMANN_CONSTANT = 5.67E-8  # in W / m^2 / K^4

    SOLAR_IRRADIANCE = 1361 #units of W/m^2

    SILICON_SPECIFIC_HEAT = 712  # in J /kg / K
    SILICON_ROCK_DENSITY = 2650  # in kg/m^3


    WATER_HEAT_CAPACITY = 4000  # in J /kg / K
    WATER_DENSITY = 1000  # in kg/m^3

    DAYS_PER_YEAR = 365.25
    HOURS_PER_DAY = 24
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR
    SECONDS_PER_YEAR = DAYS_PER_YEAR * SECONDS_PER_DAY

    # renamed for readability in naming end time
    DAYS = SECONDS_PER_DAY
    YEARS = SECONDS_PER_YEAR

    SOLAR_INSOLATION = 1361  # in W m-2
    DEGREES_TO_RADIANS = np.pi / 180.0

    EARTH_ALBEDO = 0.32