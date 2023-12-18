import numpy as np
from numba import njit

@njit
def solveHeatEquation(GRID, dt: int):
    """Solve the heat equation on a non-uniform grid.

    Args:
        GRID (Grid): Glacier data mesh.
        dt: Integration time.
    """
    # number of layers
    nl = GRID.get_number_layers()

    # Define index arrays
    k = np.arange(1, nl - 1)  # center points
    kl = np.arange(2, nl)  # lower points
    ku = np.arange(0, nl - 2)  # upper points

    temperatures = get_new_temperatures_cds2(
        grid=GRID, lower=kl, middle=k, upper=ku, dt=dt
    )

    # Write results to GRID
    GRID.set_temperature(temperatures)


def get_new_temperatures_cds2(
    grid, lower, central, upper, dt: int
) -> np.ndarray:
    """Solve heat equation using a 2nd-order central-difference scheme.

    Args:
        grid (Grid): Glacier data mesh.
        lower: Sequence of lowermost spatial bounds (j-1).
        central: Seuquence of spatial midpoints (j).
        upper: Sequence of uppermost spatial bounds (j+1).

    Returns:
        Solved column temperatures.
    """

    total_layers = grid.get_number_layers()

    k_mid = np.asarray(grid.get_thermal_diffusivity())
    heights = np.asarray(grid.get_height())

    # Get grid spacing
    spacing = (heights[0 : total_layers - 1] / 2.0) + (
        heights[1:total_layers] / 2.0
    )
    h_lo = spacing[0 : total_layers - 2]  # between z-1 and z
    h_up = spacing[1 : total_layers - 1]  # between z and z+1

    # Get temperature array from grid|
    temperature = np.array(grid.get_temperature())
    temperature_new = temperature.copy()

    k_lo = (k_mid[1 : total_layers - 1] + k_mid[2:total_layers]) / 2.0
    k_up = (k_mid[0 : total_layers - 2] + k_mid[1 : total_layers - 1]) / 2.0

    stab_t = 0.0
    c_stab = 0.8
    dt_stab = c_stab * (
        min(
            [
                min(spacing[0 : total_layers - 2] ** 2 / (2 * k_up)),
                min(spacing[1 : total_layers - 1] ** 2 / (2 * k_lo)),
            ]
        )
    )

    while stab_t < dt:
        dt_use = np.minimum(dt_stab, dt - stab_t)
        stab_t = stab_t + dt_use

        # Update the temperatures
        temperature_new[central] += (
            (k_lo * dt_use * (temperature[lower] - temperature[central]) / (h_up))
            - (k_up * dt_use * (temperature[central] - temperature[upper]) / (h_lo))
        ) / (0.5 * (h_lo + h_up))
        temperature = temperature_new.copy()

    return temperature
