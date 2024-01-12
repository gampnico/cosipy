import numpy as np
from numba import njit

import constants


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
        grid=GRID, lower=kl, central=k, upper=ku, dt=dt
    )

    # Write results to GRID
    GRID.set_temperature(temperatures)


@njit
def get_new_temperatures_cds2(
    grid, lower, central, upper, dt: int
) -> np.ndarray:
    """Solve heat equation using a 2nd-order central-difference scheme.

    Args:
        grid (Grid): Glacier data mesh.
        lower: Sequence of lowermost spatial bounds (j-1).
        central: Sequence of spatial midpoints (j).
        upper: Sequence of uppermost spatial bounds (j+1).
        dt: Integration time [s].

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

    # Get temperature array from grid
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


@njit
def get_contact_temperature(grid, idx: int) -> float:
    """Get contact temperature between two nodes.

    Args:
        grid (Grid): Glacier data mesh.
        idx: Index of uppermost node.

    Returns:
        Contact temperature between two adjacent nodes.
    """

    idx_lo = idx + 1
    effusivity_hi = grid.get_node_thermal_effusivity(idx)
    effusivity_lo = grid.get_node_thermal_effusivity(idx_lo)
    contact_temperature = (
        effusivity_hi * grid.get_node_temperature(idx)
        + effusivity_lo * grid.get_node_temperature(idx_lo)
    ) / (effusivity_hi + effusivity_lo)

    return contact_temperature


@njit
def get_skin_temperature(grid, idx: int, contact_temperature: float) -> float:
    """Get the lower skin temperature from two layers' contact temperature.

    Args:
        grid (Grid): Glacier data mesh.
        idx: Index of uppermost node.
        contact_temperature: Contact temperature between two adjacent nodes.

    Returns:
        The lowermost layer's skin temperature.
    """

    effusivity_hi = grid.get_node_thermal_effusivity(idx)
    effusivity_lo = grid.get_node_thermal_effusivity(idx + 1)
    skin_temperature = contact_temperature + (
        effusivity_hi / effusivity_lo
    ) * (contact_temperature - grid.get_node_temperature(idx))

    return skin_temperature


@njit
def get_midpoint_temperature(
    skin_lower: float, skin_upper: float, height: float
) -> float:
    """Get midpoint temperature of a plane (Incropera et al., 2013).

    Assumes 1D, steady-state conduction with no heat generation.

    Args:
        skin_lower: Lowermost skin temperature.
        skin_upper: Uppermost skin temperature.
        height: Height of layer.

    Returns:
        The layer's midpoint temperature.
    """

    midpoint = ((skin_upper - skin_lower) * (height / 2)) + skin_lower

    return midpoint


@njit
def set_midpoint_temperatures(grid, contact_temperature: float, idx: int):
    """Set midpoint temperatures of two layers from contact temperature.

    For simplicity, this assumes a perfectly smooth contact plane
    between two layers, each with an initially uniform temperature
    profile. The skin temperature opposite the contact plane remains
    fixed across a single timestep.

    TODO: embed uml in docstring.

    Args:
        grid (Grid): Glacier data mesh.
        contact_temperature: The contact temperature between two layers.
        idx: Index of the uppermost layer.
    """

    top_layer_temperature = get_midpoint_temperature(
        skin_upper=grid.get_node_temperature(idx),
        skin_lower=contact_temperature,
        height=grid.get_node_height(idx),
    )
    idx_lo = idx + 1
    bottom_layer_temperature = get_midpoint_temperature(
        skin_upper=contact_temperature,
        skin_lower=grid.get_node_temperature(idx_lo),
        height=grid.get_node_height(idx_lo),
    )

    grid.set_node_temperature(idx, top_layer_temperature)
    grid.set_node_temperature(idx_lo, bottom_layer_temperature)


def solveHeatEquation_debris(grid, dt: int):
    """Solve heat equation for debris glaciers.

    If debris is present, the grid is separated into three blocks:
    above, within and below the debris layer. The vectorised equation is
    applied to non-debris layers. Reid and Brock (2010) is used for
    debris layers.

    Upper BCs:
    * Debris at surface: BC is set to surface temperature.
    * Debris under surface layer: BC adjusts for a contact
    temperature equal to the temperature of the surface layer.
    * Debris under more than one layer: BC adjusts for a contact
    temperature derived from the layer effusivities.

    Lower BC:
    * Debris above icepack: BC set to freezing temperature. This dampens
      subdebris melt.

    .. note:
        setting lower BC to freezing dampens the subdebris melt.

    TODO: Implement heat transfer for lower BC?
    """

    if grid.get_number_debris_layers() == 0:
        solveHeatEquation(GRID=grid, dt=dt)
    else:
        total_layers = grid.get_number_layers()
        upper_debris_idx, lower_debris_idx = grid.get_debris_extents()
        idx_midpoint = []
        idx_lower = []
        idx_upper = []

        # Lower BC: clamp to freezing only when above ice
        if not lower_debris_idx == grid.number_nodes - 1:
            grid.set_node_temperature(
                lower_debris_idx,
                max(
                    constants.zero_temperature,
                    grid.get_node_temperature(lower_debris_idx + 1),
                ),
            )

        # Reid and Brock iterates upwards, so solve icepack first
        if total_layers - lower_debris_idx > 1:
            # apply bfgs HE to icepack
            idx_midpoint.append(*range(lower_debris_idx + 1, total_layers - 1))
            idx_lower.append(*range(lower_debris_idx + 2, total_layers))
            idx_upper.append(*range(lower_debris_idx, total_layers - 2))

        temperatures = get_new_temperatures_cds2(
            grid=grid,
            lower=idx_lower,
            central=idx_midpoint,
            upper=idx_upper,
            dt=dt,
        )

        for idx in idx_midpoint:
            grid.set_node_temperature(idx, temperatures[idx])

        # Solve debris temperatures
        set_heat_conservation_debris(grid=grid, dt=dt)

        # Solve snow last as it's affected by the debris temperature.
        idx_midpoint = []
        idx_lower = []
        idx_upper = []
        if upper_debris_idx > 2:
            # apply bfgs HE to snowpack
            idx_midpoint.append(*range(1, upper_debris_idx - 1))
            idx_lower.append(*range(2, upper_debris_idx))
            idx_upper.append(*range(0, upper_debris_idx - 2))


@njit
def get_ivhc_tensor(grid, extents: tuple, dt: int = 1) -> np.ndarray:
    """Get the inverse of a layer's volumetric heat capacity.

    Args:
        grid (Grid): Glacier mesh data.
        extents: Indices of debris layer extents.
        dt: Integration time. Default 1.

    Returns:
        Inverse volumetric heat capacity of a layer.
    """

    # Convert to arrays as numba can't handle lists
    heights = np.array(grid.get_height()[extents[0] : extents[1] + 1])
    densities = np.array(grid.get_density()[extents[0] : extents[1] + 1])
    conductivities = np.array(
        grid.get_thermal_conductivity()[extents[0] : extents[1] + 1]
    )
    ivhc_tensor = (dt * conductivities) / (
        2 * constants.spec_heat_debris * np.power(heights, 2) * densities
    )

    return ivhc_tensor


@njit
def get_d_tensor(
    grid, ivhc_tensor: np.ndarray, skin_temperature: float, extents: tuple
) -> np.ndarray:
    """Get Crank-Nicholson d-tensor.

    Corresponds to equation A9 in Reid & Brock (2010).

    Args:
        grid (Grid): Glacier mesh data.
        ivhc_tensor: Inverse volumetric heat capacity of a layer.
        skin_temperature: Previous temperature of uppermost debris
            layer.
        extents: Indices of debris layer extents.

    Returns:
        Crank-Nicholson d-tensor.
    """

    d_tensor = np.zeros_like(ivhc_tensor)
    grid_idx = extents[0]
    d_tensor[1] = (
        ivhc_tensor[1] * skin_temperature
        + ivhc_tensor[1] * grid.get_node_temperature(grid_idx)
        + (1 - 2 * ivhc_tensor[1]) * grid.get_node_temperature(grid_idx + 1)
        + ivhc_tensor[1] * grid.get_node_temperature(grid_idx + 2)
    )
    c_idx = ivhc_tensor.shape[0] - 2  # N-1
    d_tensor[c_idx] = (
        2 * ivhc_tensor[c_idx] * constants.zero_temperature
        + ivhc_tensor[c_idx - 2] * grid.get_node_temperature(extents[1] - 2)
        + (1 - 2 * ivhc_tensor[c_idx])
        * grid.get_node_temperature(extents[0] - 1)
    )

    for idx in range(2, c_idx):  # N-2
        grid_idx = extents[0] + idx
        d_tensor[idx] = (
            ivhc_tensor[idx] * grid.get_node_temperature(grid_idx - 1)
            + (1 - 2 * ivhc_tensor[idx]) * grid.get_node_temperature(grid_idx)
            + ivhc_tensor[idx] * grid.get_node_temperature(grid_idx + 1)
        )

    return d_tensor


@njit
def get_a_tensor(ivhc_tensor: np.ndarray) -> np.ndarray:
    """Get denominator tensor from inverse volumetric heat capacity.

    Corresponds to equation A10 in Reid and Brock (2010). Remains
    constant throughout iteration.

    Args:
        ivhc_tensor: Inverse volumetric heat capacity of a layer.

    Returns:
        Crank-Nicholson denominator A-tensor.
    """

    b_tensor = 2 * ivhc_tensor + 1
    tensor_size = b_tensor.shape[0] - 1
    a_tensor = np.zeros_like(b_tensor)

    a_tensor[1] = b_tensor[1]
    for idx in range(2, tensor_size):
        a_tensor[idx] = b_tensor[idx] - (
            (ivhc_tensor[idx] / a_tensor[idx - 1]) * ivhc_tensor[idx - 1]
        )

    return a_tensor


@njit
def get_s_tensor(
    grid,
    a_tensor: np.ndarray,
    ivhc_tensor: np.ndarray,
    previous_temperature: float,
    extents: tuple,
) -> np.ndarray:
    """Get numerator tensor for Crank-Nicholson scheme.

    Corresponds to equation A11 in Reid & Brock (2010).

    Args:
        grid (Grid): Glacier mesh data.
        a_tensor: Crank-Nicholson denominator A-tensor.
        ivhc_tensor: Inverse volumetric heat capacity of a layer.
        previous_temperature: Previous temperature of uppermost debris
            layer.
        extents: Indices of debris layer extents.

    Returns:
        Crank-Nicholson numerator S-tensor.
    """

    s_tensor = np.zeros_like(a_tensor)
    d_tensor = get_d_tensor(
        grid=grid,
        ivhc_tensor=ivhc_tensor,
        skin_temperature=previous_temperature,
        extents=extents,
    )
    s_tensor[1] = d_tensor[1]
    tensor_size = s_tensor.shape[0]
    for idx in range(2, tensor_size):
        s_tensor[idx] = d_tensor[idx] + (
            (ivhc_tensor[idx] / a_tensor[idx - 1]) * s_tensor[idx - 1]
        )

    return s_tensor


@njit
def set_upper_boundary_condition(grid, top_idx: int):
    """Set upper boundary condition for Reid and Brock (2010).

    Upper BCs:
    * Debris at surface: BC is set to surface temperature.
    * Debris under surface layer: BC adjusts for a contact
    temperature equal to the temperature of the surface layer.
    * Debris under more than one layer: BC adjusts for a contact
    temperature derived from the layer effusivities.

    .. note:
        TODO: Clamp UBC to freezing point?

    Args:
        grid (Grid): Glacier mesh data.
        top_idx: Top index of debris layer.
    """

    if top_idx == 0:
        pass
    elif top_idx == 1:
        contact_temperature = get_contact_temperature(
            grid=grid, idx=top_idx - 1
        )
        skin_temperature = get_skin_temperature(
            grid=grid, idx=1, contact_temperature=contact_temperature
        )
        grid.set_node_temperature(1, skin_temperature)
    else:
        # debris is buried under at least two layers
        contact_temperature = get_contact_temperature(
            grid=grid, idx=top_idx - 1
        )
        set_midpoint_temperatures(
            grid=grid,
            contact_temperature=contact_temperature,
            idx=top_idx - 1,
        )


@njit
def get_internal_debris_temperature(grid, dt: int = 1) -> np.ndarray:
    """Get new debris layer temperatures.

    Corresponds to equations A8-A12 in Reid & Brock (2010). Decoupled
    from surface so the surface temperature remains unchanged. This
    requires a minimum of 5 debris layers.

    .. note:
        TODO: This changes the surface temperature if only one snow
        layer is present above the debris.

    Args:
        grid (Grid): Glacier mesh data.
        dt: Integration time. Default 1.

    Returns:
        New debris layer temperatures.
    """

    deb_layers = grid.get_number_debris_layers()
    if deb_layers < 5:
        error_message = " ".join(
            (
                "Reid & Brock (2010) requires 5+ debris layers.",
                f"There are {deb_layers}.",
            )
        )
        raise ValueError(error_message)

    debris_extents = grid.get_debris_extents()
    top_debris_idx = debris_extents[0]
    current_top_temperature = grid.get_node_temperature(top_debris_idx)

    # Set upper BC
    set_upper_boundary_condition(grid=grid, top_idx=top_debris_idx)

    # A8
    ivhc_tensor = get_ivhc_tensor(grid=grid, extents=debris_extents, dt=dt)
    a_tensor = get_a_tensor(ivhc_tensor=ivhc_tensor)  # A10
    s_tensor = get_s_tensor(  # A11
        grid=grid,
        ivhc_tensor=ivhc_tensor,
        a_tensor=a_tensor,
        previous_temperature=current_top_temperature,
        extents=debris_extents,
    )

    # Iterate upwards
    temperatures = np.empty_like(s_tensor)
    temperatures_idx = temperatures.shape[0] - 1  # corresponds to N

    temperatures[temperatures_idx - 1] = s_tensor[-2] / a_tensor[-2]  # A12
    for idx in range(temperatures_idx - 2, 0, -1):
        temperatures[idx] = (1 / a_tensor[idx]) * (
            s_tensor[idx] + ivhc_tensor[idx] * temperatures[idx + 1]
        )

    return temperatures


@njit
def set_internal_debris_temperature(grid, temperatures: np.ndarray):
    """Set internal debris temperatures.

    Args:
        grid (Grid): Glacier mesh data.
        temperatures: Internal debris temperatures.
    """

    min_idx, max_idx = grid.get_debris_extents()
    for idx in range(min_idx + 1, max_idx):
        grid.set_node_temperature(idx, temperatures[idx - min_idx])


@njit
def set_heat_conservation_debris(grid, dt: int):
    """Set temperatures using the heat equation for englacial debris.

    Adapted from Reid and Brock, (2010).

    Args:
        grid (Grid): Glacier data mesh.
        dt: Model timestep [s].
    """

    debris_temperature = get_internal_debris_temperature(grid=grid, dt=dt)
    set_internal_debris_temperature(grid=grid, temperatures=debris_temperature)
