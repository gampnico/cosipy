import numpy as np
from numba import njit

import constants as cn


@njit
def refreezing(GRID):
    # refrozen water
    water_refrozen = 0.0
    LWCref = 0.0

    # Irreducible water when refrozen
    theta_r = 0.0

    # numba expects to sum numpty types
    total_start = np.nansum(np.array(GRID.get_liquid_water_content()))
    # Compute conversion factor A (1/K)
    A = (cn.spec_heat_ice * cn.ice_density) / (
        cn.water_density * cn.lat_heat_melting
    )

    # Loop over all internal grid points for percolation
    # TODO: Vectorise!
    for idxNode in range(0, GRID.number_nodes - 1, 1):
        if (
            (GRID.get_node_temperature(idxNode) - cn.zero_temperature < 1e-3)
            & (GRID.get_node_liquid_water_content(idxNode) > theta_r)
            & (GRID.get_node_ice_fraction(idxNode) > 0.0)
            & (GRID.get_node_ntype(idxNode) != 1)
        ):
            # Temperature difference between layer and freezing temperature, cold content in temperature
            dT = GRID.get_node_temperature(idxNode) - cn.zero_temperature

            # Changes in volumetric contents, maximum amount of water that can refreeze from cold content
            dtheta_w = A * dT * GRID.get_node_ice_fraction(idxNode)

            # Check if enough water to refreeze, if less water than potential energy from cold content, only available water is refrozen
            if (
                GRID.get_node_liquid_water_content(idxNode) + dtheta_w
            ) < theta_r:
                dtheta_w = theta_r - GRID.get_node_liquid_water_content(
                    idxNode
                )

            dtheta_i = (cn.water_density / cn.ice_density) * -dtheta_w
            dT = dtheta_i / (A * GRID.get_node_ice_fraction(idxNode))
            GRID.set_node_temperature(
                idxNode, GRID.get_node_temperature(idxNode) + dT
            )

            if (
                GRID.get_node_ice_fraction(idxNode) + dtheta_i + theta_r
            ) >= 1.0:
                GRID.set_node_liquid_water_content(idxNode, theta_r)
                GRID.set_node_ice_fraction(idxNode, 1.0)
            else:
                GRID.set_node_liquid_water_content(
                    idxNode,
                    GRID.get_node_liquid_water_content(idxNode) + dtheta_w,
                )
                GRID.set_node_ice_fraction(
                    idxNode, GRID.get_node_ice_fraction(idxNode) + dtheta_i
                )

        else:
            dtheta_i = 0
            dtheta_w = 0

        GRID.set_node_refreeze(
            idxNode, dtheta_i * GRID.get_node_height(idxNode)
        )
        water_refrozen -= dtheta_w * GRID.get_node_height(idxNode)

    total_end = np.nansum(np.array(GRID.get_liquid_water_content()))

    return water_refrozen
