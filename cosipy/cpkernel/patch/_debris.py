"""Defines class methods for `DebrisNode` objects.

To modify an existing method, make your changes here. That's it.

To add a new method:
1. Define an njitted function in the relevant subsection here.
2. Go to `cpkernel.patch._ctors` and follow the instructions there.
3. Go to `cpkernel.patch.proxies` and follow the instructions there.
"""
from numba import float64, int64, njit

import constants

"""Overload get/set methods for attributes."""


@njit(cache=False)
def DebrisNode_get_layer_height(self) -> float64:
    return self.height


@njit(cache=False)
def DebrisNode_get_layer_temperature(self) -> float64:
    return self.temperature


@njit(cache=False)
def DebrisNode_get_layer_ice_fraction(self) -> float64:
    return self.ice_fraction


@njit(cache=False)
def DebrisNode_get_layer_liquid_water_content(self) -> float64:
    return self.liquid_water_content


@njit(cache=False)
def DebrisNode_get_layer_refreeze(self) -> float64:
    return self.refreeze


@njit(cache=False)
def DebrisNode_get_layer_ntype(self) -> int64:
    return self.ntype


"""Overload get/set methods for derived state variables.

1. Define an njitted function. Class properties (`self.foo`) are
   supported, but class methods (`self.get_foo()`) raise a signature
   error.
2. Link the class method to use the njitted function.
3. Overload the method with the njitted function.
"""


@njit(cache=False)
def DebrisNode_get_layer_air_porosity(self) -> float64:
    """Get the node's volumetrically-weighted interstitial void porosity.

    The function's name is kept as `get_layer_air_porosity` for
    cross-compatibility with other Node objects.

    Does NOT include the debris material's porosity, and assumes no
    liquid water content. Note that the packing and void porosities are
    volumetrically-weighted!

    Returns:
        Volumetrically-weighted interstitial void porosity [-].
    """

    if constants.debris_void_porosity >= 1.0:  # filled with air
        porosity = constants.debris_packing_porosity
    else:
        porosity = (
            constants.debris_packing_porosity * constants.debris_void_porosity
        )

    return max(0.0, porosity)


@njit(cache=False)
def DebrisNode_get_layer_porosity(self) -> float:
    """Get the node's porosity.

    Returns:
        Total volumetrically-weighted debris porosity and its
        interstitial void porosity. [-].
    """

    porosity = (
        1 - constants.debris_packing_porosity
    ) * constants.debris_porosity + DebrisNode_get_layer_air_porosity(self)

    return porosity


@njit(cache=False)
def DebrisNode_get_layer_density(self) -> float64:
    """Get the node's mean density including air and interstices.

    The density includes the clast material, the material filling
    the void between clasts, and the air in both clast and void
    filler pores.

    Returns:
        Debris density [:math:`kg~m^{-3}`].
    """

    if constants.debris_void_porosity >= 1.0:
        density = (
            DebrisNode_get_layer_porosity(self) * constants.air_density
            + (1 - constants.debris_packing_porosity)
            * constants.debris_porosity
            * constants.debris_density  # clast density
            + (1 - DebrisNode_get_layer_air_porosity(self))
            * constants.debris_void_density  # void filler density
        )
    else:
        density = (
            DebrisNode_get_layer_porosity(self) * constants.air_density
            + (1 - DebrisNode_get_layer_porosity(self))
            * constants.debris_density
        )
    return density


@njit(cache=False)
def DebrisNode_get_layer_specific_heat(self) -> float64:
    """Get the node's volumetrically-averaged specific heat capacity.

    Returns:
        Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
    """

    if constants.debris_void_porosity >= 1.0:
        cp = DebrisNode_get_layer_air_porosity(
            self
        ) * constants.spec_heat_air + (
            1 - DebrisNode_get_layer_air_porosity(self)
        ) * (
            constants.spec_heat_debris
        )

    else:
        cp = DebrisNode_get_layer_air_porosity(self) * constants.spec_heat_air
        +(
            1 - DebrisNode_get_layer_air_porosity(self)
        ) * constants.spec_heat_debris

    return cp


@njit(cache=False)
def DebrisNode_get_layer_cold_content(self) -> float64:
    """Get the node's cold content.

    Returns:
        Cold content [:math:`J~m^{-2}`].
    """
    return (
        -DebrisNode_get_layer_specific_heat(self)
        * DebrisNode_get_layer_density(self)
        * DebrisNode_get_layer_height(self)
        * (DebrisNode_get_layer_temperature(self) - constants.zero_temperature)
    )


@njit(cache=False)
def DebrisNode_get_layer_thermal_conductivity(self) -> float64:
    """Gets the node's thermal conductivity at ambient temperature.

    The debris' thermal conductivity at 273.15 K should be set in
    constants.py, as it varies between lithologies.

    Equation is from Vosteen & Schellschmidt (2003).

    Returns:
        Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
    """

    methods_allowed = ["sedimentary", "crystalline"]

    if constants.debris_structure not in methods_allowed:
        message = (
            f'"{constants.debris_structure}" debris structure',
            f"is not allowed, must be one of",
            f'{", ".join(methods_allowed)}',
        )
        raise ValueError(message)
    elif constants.debris_structure == "sedimentary":
        a = 0.0034
        b = 0.0039
    elif constants.debris_structure == "crystalline":
        a = 0.0030
        b = 0.0042

    porosity = DebrisNode_get_layer_air_porosity(self)
    conductivity = (1 - porosity) * (
        constants.thermal_conductivity_debris * 0.99
        + DebrisNode_get_layer_temperature(self)
        * (a - (b / constants.thermal_conductivity_debris))
    ) + (porosity * constants.k_a)

    return conductivity


@njit(cache=False)
def DebrisNode_get_layer_thermal_diffusivity(self) -> float64:
    """Get the node's thermal diffusivity.

    Returns:
        Thermal diffusivity [:math:`m^{2}~s^{-1}`].
    """

    if constants.debris_structure == "crystalline":
        # Vosteen & Schellschmidt (2003)
        K = 0.45 * DebrisNode_get_layer_thermal_conductivity(self)
    else:
        K = DebrisNode_get_layer_thermal_conductivity(self) / (
            DebrisNode_get_layer_density(self)
            * DebrisNode_get_layer_specific_heat(self)
        )
    return K
