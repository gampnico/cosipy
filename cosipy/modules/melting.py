"""Parametrises surface melting."""
from numba import njit

import constants
from config import use_debris


@njit
def surface_melting(
    grid,
    sw_net: float,
    lw_in: float,
    lw_out: float,
    ghf: float,
    rhf: float,
    shf: float,
    lhf: float,
    dt: int,
) -> tuple:
    """Get surface melt energy and melt rate.

    Args:
        grid (Grid): Glacier data mesh.
        sw_net: Net shortwave radiation.
        lw_in: Incoming longwave radiation.
        lw_out: Outgoing longwave radiation.
        ghf: Ground heat flux.
        rhf: Rain heat flux.
        shf: Sensible heat flux.
        lhf: Latent heat flux.
        dt: Integration time.

    Returns:
    tuple[float, float]: Surface melt energy and melt rate.
    """
    if not use_debris:
        melt_energy = get_surface_melt_energy_sum(
            sw_net, lw_in, lw_out, ghf, rhf, shf, lhf
        )
    elif grid.get_node_ntype(0) == 1:
        melt_energy = get_debris_melt_energy_reid(grid)
    else:
        melt_energy = get_surface_melt_energy_sum(
            sw_net, lw_in, lw_out, ghf, rhf, shf, lhf
        )

    melt_rate = get_surface_melt_rate(melt_energy, dt)

    return melt_energy, melt_rate


@njit
def get_surface_melt_energy_sum(
    sw_net: float,
    lw_in: float,
    lw_out: float,
    ghf: float,
    rhf: float,
    shf: float,
    lhf: float,
) -> float:
    """Get surface melt energy.

    Args:
        sw_net: Net shortwave radiation.
        lw_in: Incoming longwave radiation.
        lw_out: Outgoing longwave radiation.
        ghf: Ground heat flux.
        rhf: Rain heat flux.
        shf: Sensible heat flux.
        lhf: Latent heat flux.

    Returns:
        Conductive heat flux through the surface layer.
    """
    sum_flux = sum((sw_net, lw_in, lw_out, ghf, rhf, shf, lhf))
    return max(0.0, sum_flux)


@njit
def get_debris_melt_energy_reid(grid) -> float:
    """Get melt energy below a surface debris layer.

    Adapted from Reid et al., (2012). Only valid for debris layers.

    Args:
        grid (Grid): Glacier data mesh.

    Returns:
        Conductive heat flux through the surface debris layer.
    """
    _, debris_ice_idx = grid.get_debris_extents(0)
    delta_T = grid.get_node_temperature(0) - grid.get_node_temperature(
        debris_ice_idx - 1
    )
    debris_layer_heights = grid.get_total_debris_height()

    conductive_heat_flux = -grid.get_node_thermal_conductivity(0) * (
        delta_T / debris_layer_heights
    )
    if conductive_heat_flux < 0.0:
        conductive_heat_flux = 0.0

    return conductive_heat_flux


@njit
def get_surface_melt_rate(
    melt_energy: float, dt: int, density: float = 1000.0
) -> float:
    """Get the surface melt rate.

    Args:
        melt_energy: Available melt energy.
        dt: Timestep resolution.
        density: Surface layer density.

    Returns:
        Melt rate (per timestep).
    """
    # Convert melt energy to m w.e.q.
    melt_rate = melt_energy * dt / (density * constants.lat_heat_melting)

    return melt_rate
