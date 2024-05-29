import numpy as np
from numba import njit

import constants
from config import use_debris
from cosipy.modules.heatEquation import set_heat_conservation_debris


def penetrating_radiation(GRID, SWnet, dt):
    penetrating_allowed = ["Bintanja95"]
    if constants.penetrating_method == "Bintanja95":
        if use_debris:
            subsurface_melt, Si, E = method_Bintanja_debris(GRID, SWnet, dt)
        else:
            subsurface_melt, Si, E = method_Bintanja(GRID, SWnet, dt)
    else:
        error_msg = (
            f'Penetrating method = "{constants.penetrating_method}"',
            f'is not allowed, must be one of {", ".join(penetrating_allowed)}',
        )
        raise ValueError(" ".join(error_msg))

    return subsurface_melt, Si, E


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
        node_ntype = grid.get_node_ntype(idxNode)
        if node_ntype == 1:
            spec_heat = constants.spec_heat_debris
        else:
            spec_heat = constants.spec_heat_ice
        # New temperature due to penetrating shortwave radiation
        T_rad = grid.get_node_temperature(idxNode) + (
            shortwave[idxNode] / (grid.get_node_density(idxNode) * spec_heat)
        ) * (dt / grid.get_node_height(idxNode))

        if (T_rad > constants.zero_temperature) and (node_ntype == 0):
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
                dh = 1.0  # prevent zero division if ice fraction is zero

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
    return subsurface_melt, Si, E


@njit
def get_subdebris_decay(
    grid, depth: np.ndarray, decay: np.ndarray
) -> np.ndarray:
    _, debris_lowest = grid.get_debris_extents()

    if (
        grid.get_node_density(debris_lowest + 1)
        <= constants.snow_ice_threshold
    ):
        decay[debris_lowest + 1 :] = np.exp(17.1 * -depth[debris_lowest + 1 :])
    else:
        decay[debris_lowest + 1 :] = np.exp(2.5 * -depth[debris_lowest + 1 :])
    return decay


@njit
def get_decay_curve(grid, shortwave: float) -> tuple:
    """

    Returns:
        tuple[float,np.ndarray]
    """
    depth = np.append(  # numba doesn't support np.insert
        0.0, np.array(grid.get_depth())
    )

    if grid.get_node_ntype(0) == 1:
        Si = 0.0
        decay = np.exp(0.0 * -depth)
    elif grid.get_node_density(0) <= constants.snow_ice_threshold:
        Si = float(shortwave) * 0.1
        decay = np.exp(17.1 * -depth)
    else:
        Si = float(shortwave) * 0.2
        decay = np.exp(2.5 * -depth)
    decay = get_subdebris_decay(grid=grid, depth=depth, decay=decay)

    # E = Si * np.abs(np.diff(decay))  # TODO: isn't this Si(z)?

    return Si, decay


@njit(cache=False)
def method_Bintanja_debris_old(GRID, SWnet: float, dt: float) -> tuple:
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

    set_heat_conservation_debris(grid=GRID, dt=dt)

    E = get_debris_conduction(grid=GRID, flux=E)
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
        tuple[float, float, np.ndarray]: Subsurface melt and penetrated
        shortwave radiation at the surface.
    """
    subsurface_melt = 0.0  # Store total subsurface melt

    # Absorption of shortwave radiation
    # set_heat_conservation_debris(grid=GRID, dt=dt)
    Si, decay = get_decay_curve(grid=GRID, shortwave=SWnet)
    E = Si * np.abs(np.diff(decay))  # TODO: isn't this Si(z)?

    if GRID.get_number_debris_layers() > 0:
        _, debris_lowest = GRID.get_debris_extents()
        debris_flux = max(
            0,
            get_conductive_heat_flux(
                grid=GRID, idx=debris_lowest, direction=1
            ),
        )
        E[debris_lowest + 1 :] = debris_flux * np.abs(
            np.diff(decay[debris_lowest + 1 :])
        )
    # set_heat_conservation_debris(grid=GRID, dt=dt)

    # E = get_debris_conduction(grid=GRID, flux=E)
    subsurface_melt = get_subsurface_melt(GRID, E, dt)

    # subsurface_melt = (Si * dt) / (
    #     constants.ice_density * constants.lat_heat_melting
    # )

    return subsurface_melt, Si, E


@njit
def get_debris_melt_energy_reid(grid) -> float:
    """Get melt energy below a surface debris layer.

    Adapted from Reid et al., (2012). Only valid for contiguous debris
    layers.

    Args:
        grid (Grid): Glacier data mesh.

    Returns:
        Conductive heat flux through the surface debris layer.
    """
    _, debris_lowest = grid.get_debris_extents(0)
    delta_T = grid.get_node_temperature(
        debris_lowest
    ) - grid.get_node_temperature(0)
    debris_layer_heights = grid.get_total_debris_height()

    conductive_heat_flux = -grid.get_node_thermal_conductivity(
        debris_lowest
    ) * (delta_T / debris_layer_heights)

    return max(0.0, conductive_heat_flux)


@njit
def get_conductive_heat_flux(grid, idx: int, direction: int = 1) -> float:
    """
    Juen et al. (2013). This is the conductive heat flux into the initial
    layer, i.e. a negative flux flows from hot to cold.
    """
    flux = (
        grid.get_node_thermal_conductivity(idx)
        * (
            grid.get_node_temperature(idx + direction)
            - grid.get_node_temperature(idx)
        )
        / (
            (grid.get_node_height(idx) / 2)
            + (grid.get_node_height(idx + direction) / 2)
        )
    )
    return flux


@njit
def get_debris_conduction(grid, flux: np.ndarray) -> float:
    top_idx, bottom_idx = grid.get_debris_extents()

    if top_idx > 0:
        flux[top_idx] = get_conductive_heat_flux(
            grid=grid, idx=top_idx, direction=-1
        )
    if bottom_idx + 1 < grid.number_nodes:
        flux[bottom_idx] = get_conductive_heat_flux(
            grid=grid, idx=bottom_idx, direction=1
        )
        depth = 0.0
        for i in range(bottom_idx + 1, grid.number_nodes):
            depth -= grid.get_node_height(i)
            if grid.get_node_ntype(i) == 1:
                Si = 0.0
                decay = 0.0
            if grid.get_node_density(i) <= constants.snow_ice_threshold:
                Si = flux[bottom_idx] * 0.1
                decay = np.exp(17.1 * depth)
            else:
                Si = flux[bottom_idx] * 0.2
                decay = np.exp(2.5 * depth)
            flux[i] = Si * np.abs(decay)

    return flux
