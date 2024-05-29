import numpy as np
from numba import njit
from config import use_debris

@njit
def percolate_column(grid) -> float:
    """Percolate water through column.

    Args:
        grid (Grid): Gridded glacier data.

    Returns:
        Residual meltwater runoff.
    """
    for idxNode in range(0, grid.number_nodes - 1, 1):
        # Get irreducible water content [-]
        theta_e = grid.get_node_irreducible_water_content(idxNode)

        # Get initial liquid water content [-]
        theta_w = grid.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0.0:
            grid.set_node_liquid_water_content(idxNode, theta_e)

            """
            old:
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1) + residual,
            )

            new:
            If water is pushed to next layer, the layer heights have to
            be considered because of fractions.
            """
            residual = residual * grid.get_node_height(idxNode)
            grid.set_node_liquid_water_content(
                idxNode + 1,
                grid.get_node_liquid_water_content(idxNode + 1)
                + residual / grid.get_node_height(idxNode + 1),
            )

    return residual


@njit
def get_runoff(grid) -> float:
    """Get meltwater runoff for a column.

    Runoff is equal to LWC in the last node & must be converted
    from kg/m3 to kg/m2. Converting from fraction to kg/m3 (*1000) and
    from mm to m (/1000) is unnecessary.

    Args:
        grid (Grid): Gridded glacier data.

    Returns:
        Meltwater runoff.
    """

    max_index = grid.number_nodes - 1
    runoff = grid.get_node_liquid_water_content(
        max_index
    ) * grid.get_node_height(max_index)
    grid.set_node_liquid_water_content(max_index, 0.0)

    return runoff


@njit
def percolate_column_debris(grid) -> float:
    """Percolate water through column.

    Args:
        grid (Grid): Gridded glacier data.

    Returns:
        Residual meltwater runoff.
    """
    for idxNode in range(0, grid.number_nodes - 1, 1):
        # Get irreducible water content [-]
        if grid.get_node_ntype(idxNode) != 1:
            theta_e = grid.get_node_irreducible_water_content(idxNode)
        else:
            theta_e = 0.0

        # Get initial liquid water content [-]
        theta_w = grid.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0:
            # then percolate to the next layer (add to the next layer)
            grid.set_node_liquid_water_content(idxNode, theta_e)

            ### old
            # GRID.set_node_liquid_water_content(idxNode+1, GRID.get_node_liquid_water_content(idxNode+1)+residual)

            ### new: if water is pushed to next layer, because of fractions the layer heights have to be considered
            residual = residual * grid.get_node_height(idxNode)

            # next_snow_idx = grid.get_next_snow_ice_layer(idxNode + 1)
            # grid.set_node_liquid_water_content(
            #     next_snow_idx,
            #     grid.get_node_liquid_water_content(next_snow_idx)
            #     + residual / grid.get_node_height(next_snow_idx),
            # )
            grid.set_node_liquid_water_content(
                idxNode + 1,
                grid.get_node_liquid_water_content(idxNode + 1)
                + residual / grid.get_node_height(idxNode + 1),
            )
        else:
            grid.set_node_liquid_water_content(idxNode, theta_w)

    return residual


@njit
def percolation(GRID, water: float, dt: float) -> float:
    """Percolation and refreezing of melt water through the snow- and firn pack

    Args:
        GRID: GRID object.
        water: Melt water at the surface, [m w.e.q.].
        dt: Integration time.

    Returns:
        float: Percolated meltwater.
    """

    # convert m to mm = kg/m2, not needed because of change to fraction
    # water = water * 1000

    # convert kg/m2 to kg/m3
    water = water / GRID.get_node_height(0)
    # kg/m3 to fraction
    # water = water / 1000

    # set liquid water of top layer (idx, LWCnew) in m
    GRID.set_node_liquid_water_content(
        0, GRID.get_node_liquid_water_content(0) + float(water)
    )

    # for consistency check
    # numba expect numpy type in np.sum()
    # total_start = np.nansum(np.array(GRID.get_liquid_water_content()))

    if use_debris:
        percolate_column_debris(GRID)
    else:
        percolate_column(GRID)
    Q = get_runoff(GRID)

    # check_lwc_conservation(GRID, total_start, dt)  # for consistency check
    return Q


@njit
def check_lwc_conservation(grid, start_lwc: float, dt: float):
    """Check total liquid water content is conserved.

    Args:
        grid: GRID object.
        start_lwc: Initial total liquid water content.
        dt: Integration time [s].
    """
    end_lwc = np.nansum(np.array(grid.get_liquid_water_content()))
    if not np.isclose(start_lwc, end_lwc):
        # can't index xarrays directly with njit
        if grid.new_snow_timestamp == 0.0:
            snow_time = grid.old_snow_timestamp
        else:
            snow_time = grid.new_snow_timestamp
        timestep = snow_time / dt
        delta = start_lwc - end_lwc
        warn_sanity = "\nWARNING: When percolating, the initial LWC is not equal to final LWC"
        # numba doesn't support warnings, and we don't want to raise an error
        print(f"{warn_sanity} at timestep {int(timestep)}. dLWC:")
        print(delta)
