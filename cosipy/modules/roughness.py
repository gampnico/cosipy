import constants
from config import use_debris


def updateRoughness(GRID):
    roughness_allowed = ["Moelg12"]
    if constants.roughness_method == "Moelg12":
        if not use_debris:
            sigma = method_Moelg(GRID)
        else:
            sigma = method_Moelg_debris(GRID)
    else:
        error_message = (
            f'Roughness method = "{constants.roughness_method}" is not allowed,',
            f'must be one of {", ".join(roughness_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return sigma


def method_Moelg(GRID):
    """This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Check whether snow or ice
    if GRID.get_node_density(0) <= constants.snow_ice_threshold:
        # Get hours since the last snowfall
        # First get fresh snow properties (height and timestamp)
        _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

        # Get time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp) / 3600.0
        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(
            constants.roughness_fresh_snow
            + constants.aging_factor_roughness * hours_since_snowfall,
            constants.roughness_firn,
        )
    else:
        # Roughness length, set to ice
        sigma = constants.roughness_ice

    return sigma / 1000


def method_Moelg_debris(GRID) -> float:
    """Update the roughness length for debris-covered glaciers.

    Adapted from Moelg et al. (2009), J.Clim. The roughness length of
    snow linearly increases from 0.24 (fresh snow) to 4 (firn) in 60
    days (1440 hours) i.e. (4-0.24)/1440 = 0.0026.

    Returns:
        Surface roughness length, [mm]
    """

    # Check whether snow or ice
    if GRID.get_node_ntype(0) == 1:
        sigma = constants.roughness_debris
    elif GRID.get_node_density(0) <= constants.snow_ice_threshold:
        # First get fresh snow properties (height and timestamp)
        _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

        # Get time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

        sigma = min(
            constants.roughness_fresh_snow
            + constants.aging_factor_roughness * hours_since_snowfall,
            constants.roughness_firn,
        )
    else:
        # Roughness length, set to ice
        sigma = constants.roughness_ice

    return sigma / 1000
