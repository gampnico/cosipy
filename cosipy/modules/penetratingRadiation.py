import numpy as np
from numba import njit

import constants
from config import use_debris


def penetrating_radiation(GRID, SWnet, dt):
    penetrating_allowed = ["Bintanja95"]
    if constants.penetrating_method == "Bintanja95":
        if use_debris:
            subsurface_melt, Si = method_Bintanja_debris(GRID, SWnet, dt)
        else:
            subsurface_melt, Si = method_Bintanja(GRID, SWnet, dt)
    else:
        error_msg = (
            f'Penetrating method = "{constants.penetrating_method}"',
            f'is not allowed, must be one of {", ".join(penetrating_allowed)}',
        )
        raise ValueError(" ".join(error_msg))

    return subsurface_melt, Si


@njit(cache=False)
def set_node_attrs_bintanja(grid, idx: int, d_h: float, d_water: float):
    """Set node attributes after radiation has penetrated.

    Args:
        grid (Grid): Glacier data mesh.
        idx: Node index.
        d_h: Change in layer thickness [m].
        d_water: Change in water fraction [-].
    """
    grid.set_node_liquid_water_content(
        idx,
        grid.get_node_liquid_water_content(idx) + d_water,
    )
    lwc_temp = grid.get_node_liquid_water_content(idx) * grid.get_node_height(
        idx
    )
    grid.set_node_temperature(idx, constants.zero_temperature)
    grid.set_node_height(idx, (1 - d_h) * grid.get_node_height(idx))
    grid.set_node_liquid_water_content(
        idx, lwc_temp / grid.get_node_height(idx)
    )


@njit(cache=False)
def get_subsurface_melt(grid, shortwave: np.ndarray, dt: float) -> float:
    """Get subsurface meltwater.

    Args:
        grid (Grid): Glacier data mesh.
        shortwave: Penetrated shortwave radiation for each layer
            depth [Wm^-2].
        dt: Timestep duration [s].

    Returns:
        Subsurface meltwater [m.w.e].
    """
    subsurface_melt = 0.0  # Store total subsurface melt
    list_of_layers_to_remove = []  # Layer numbers to be removed

    # Only need to compute conversion factor A once
    A = (constants.spec_heat_ice * constants.ice_density) / (
        constants.water_density * constants.lat_heat_melting
    )
    density_ratio = constants.water_density / constants.ice_density

    for idxNode in range(0, grid.number_nodes - 1):
        # New temperature due to penetrating shortwave radiation
        T_rad = grid.get_node_temperature(idxNode) + (
            shortwave[idxNode]
            / (grid.get_node_density(idxNode) * constants.spec_heat_ice)
        ) * (dt / grid.get_node_height(idxNode))

        if T_rad - constants.zero_temperature > 0.0:
            # Difference between layer temperature and freezing temperature
            dT = T_rad - constants.zero_temperature

            """Changes in volumetric contents:
            * dtheta_w: change in water fraction
            * dtheta_i: change in ice fraction
            """
            node_ice_fraction = grid.get_node_ice_fraction(idxNode)
            dtheta_w = A * dT * node_ice_fraction
            dtheta_i = (density_ratio) * -dtheta_w
            if node_ice_fraction != 0.0:
                dh = -dtheta_i / node_ice_fraction
            else:
                dh = 0.0  # prevent zero division if ice fraction is zero

            if dh >= 1.0:
                list_of_layers_to_remove.append(idxNode)
            else:
                set_node_attrs_bintanja(grid, idxNode, dh, dtheta_w)
            subsurface_melt = (
                subsurface_melt + dtheta_w * grid.get_node_height(idxNode)
            )
        else:
            grid.set_node_temperature(idxNode, T_rad)

    # Remove melted layers
    if list_of_layers_to_remove:
        # numba jitclass can't compute fingerprint of empty list
        grid.remove_node(list_of_layers_to_remove)

    return subsurface_melt


@njit(cache=False)
def method_Bintanja(GRID, SWnet: float, dt: float) -> tuple:
    """Get penetrating radiation and subsurface meltwater.

    From Bintanja and Van Den Broeke, (1995).

    Args:
        GRID (Grid): Glacier data mesh.
        SWnet: Incoming net shortwave radiation [Wm^-2].
        dt: Timestep duration [s].

    Returns:
        tuple[float, float]: Subsurface meltwater and penetrated
        shortwave radiation at the surface.
    """
    subsurface_melt = 0.0  # Store total subsurface melt
    # Absorption of shortwave radiation
    # numba doesn't support np.insert
    depth = np.append(0.0, np.array(GRID.get_depth()))
    if GRID.get_node_density(0) <= constants.snow_ice_threshold:
        Si = float(SWnet) * 0.1
        decay = np.exp(17.1 * -depth)
    else:
        Si = float(SWnet) * 0.2
        decay = np.exp(2.5 * -depth)
    E = Si * np.abs(np.diff(decay))  # TODO: isn't this Si(z)?

    subsurface_melt = get_subsurface_melt(GRID, E, dt)
    return subsurface_melt, Si


@njit(cache=False)
def method_Bintanja_debris(GRID, SWnet: float, dt: float) -> tuple:
    """Get penetrating radiation and subsurface meltwater.

    Adapted from Bintanja and Van Den Broeke, (1995) and Lejeune et al.,
    (2013). Shortwave radiation cannot penetrate a debris layer.

    Args:
        GRID (Grid): Glacier data mesh.
        SWnet: Incoming net shortwave radiation [Wm^-2].
        dt: Timestep duration [s].

    Returns:
        tuple[float, float]: Subsurface melt and penetrated shortwave
        radiation at the surface.
    """
    subsurface_melt = 0.0  # Store total subsurface melt

    # Absorption of shortwave radiation
    depth = np.append(  # numba doesn't support np.insert
        0.0, np.array(GRID.get_depth())
    )
    if GRID.get_node_ntype(0) == 1:
        Si = 0.0
        decay = np.exp(0.0 * -depth)
    elif GRID.get_node_density(0) <= constants.snow_ice_threshold:
        Si = float(SWnet) * 0.1
        decay = np.exp(17.1 * -depth)
    else:
        Si = float(SWnet) * 0.2
        decay = np.exp(2.5 * -depth)

    E = Si * np.abs(np.diff(decay))  # TODO: isn't this Si(z)?

    subsurface_melt = get_subsurface_melt(GRID, E, dt)

    return subsurface_melt, Si
