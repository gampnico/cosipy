"""Defines class methods for `Node` objects.

To modify an existing method, make your changes here. That's it.

To add a new method:
1. Define an njitted function in the relevant subsection here.
2. Go to `cpkernel.patch._clean` and follow the instructions there.
3. OPTIONAL: if you want to use your new method in debris-covered
   simulations, go to `cpkernel.patch._ctors` and follow the
   instructions there.
4. Go to `cpkernel.patch.proxies` and follow the instructions there.
"""

import math

from numba import float64, int64, njit, optional

import constants

"""Overload get/set methods for attributes."""


@njit(cache=False)
def Node_init_ice_fraction(
    ice_fraction: optional(float64), density: float64
) -> float64:
    """Initialise node ice fraction.

    Args:
        ice_fraction: Volumetric ice fraction [-].
        density: Layer density [:math:`kg~m^{-3}`].

    Returns:
        Volumetric ice fraction [-].
    """
    if ice_fraction is None:
        # Remove weight of air from density
        a = (
            density
            - (1 - (density / constants.ice_density)) * constants.air_density
        )
        ice_fraction = a / constants.ice_density
    else:
        ice_fraction = ice_fraction

    return ice_fraction


@njit(cache=False)
def Node_get_layer_height(self) -> float64:
    return self.height


@njit(cache=False)
def Node_get_layer_temperature(self) -> float64:
    return self.temperature


@njit(cache=False)
def Node_get_layer_ice_fraction(self) -> float64:
    return self.ice_fraction


@njit(cache=False)
def Node_get_layer_liquid_water_content(self) -> float64:
    return self.liquid_water_content


@njit(cache=False)
def Node_get_layer_refreeze(self) -> float64:
    return self.refreeze


@njit(cache=False)
def Node_get_layer_ntype(self) -> int64:
    return self.ntype


"""Overload get/set methods for derived state variables.

1. Define an njitted function. Class properties (`self.foo`) are
   supported, but class methods (`self.get_foo()`) raise a signature
   error.
2. Link the class method to use the njitted function.
3. Overload the method with the njitted function.
"""


@njit(cache=False)
def Node_get_layer_air_porosity(self) -> float64:
    """Get the fraction of air in the node.

    Returns:
        Air porosity [:math:`m`].
    """
    porosity = max(0.0, 1 - self.liquid_water_content - self.ice_fraction)

    return porosity


@njit(cache=False)
def Node_get_layer_porosity(self) -> float64:
    """Get the node's porosity.

    Returns:
        Air porosity [-].
    """
    return 1 - self.ice_fraction - self.liquid_water_content


@njit(cache=False)
def Node_get_layer_density(self) -> float64:
    """Get the node's mean density including ice and liquid.

    Returns:
        Snow density [:math:`kg~m^{-3}`].
    """
    return (
        self.ice_fraction * constants.ice_density
        + self.liquid_water_content * constants.water_density
        + Node_get_layer_air_porosity(self) * constants.air_density
    )


@njit(cache=False)
def Node_get_layer_specific_heat(self):
    """Get the node's volumetrically averaged specific heat capacity.

    Returns:
        Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
    """
    return (
        self.ice_fraction * constants.spec_heat_ice
        + Node_get_layer_air_porosity(self) * constants.spec_heat_air
        + self.liquid_water_content * constants.spec_heat_water
    )


@njit(cache=False)
def Node_get_layer_irreducible_water_content(self) -> float64:
    """Get the node's irreducible water content.

    Returns:
        Irreducible water content [-].
    """
    if self.ice_fraction <= 0.23:
        theta_e = 0.0264 + 0.0099 * (
            (1 - self.ice_fraction) / self.ice_fraction
        )
    elif (self.ice_fraction > 0.23) & (self.ice_fraction <= 0.812):
        theta_e = 0.08 - 0.1023 * (self.ice_fraction - 0.03)
    else:
        theta_e = 0.0
    return theta_e


@njit(cache=False)
def Node_get_layer_cold_content(self) -> float64:
    """Get the node's cold content.

    Returns:
        Cold content [:math:`J~m^{-2}`].
    """
    return (
        -Node_get_layer_specific_heat(self)
        * Node_get_layer_density(self)
        * self.height
        * (self.temperature - constants.zero_temperature)
    )


@njit(cache=False)
def Node_get_layer_thermal_conductivity(self) -> float64:
    """Get the node's volumetrically weighted thermal conductivity.

    Returns:
        Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
    """
    methods_allowed = ["bulk", "empirical"]
    if constants.thermal_conductivity_method == "bulk":
        lam = (
            self.ice_fraction * constants.k_i
            + Node_get_layer_air_porosity(self) * constants.k_a
            + self.liquid_water_content * constants.k_w
        )
    elif constants.thermal_conductivity_method == "empirical":
        lam = 0.021 + 2.5 * math.pow((Node_get_layer_density(self) / 1000), 2)
    else:
        message = (
            "Thermal conductivity method =",
            f"{constants.thermal_conductivity_method}",
            f"is not allowed, must be one of",
            f"{', '.join(methods_allowed)}",
        )
        raise ValueError(" ".join(message))
    return lam


@njit(cache=False)
def Node_get_layer_thermal_diffusivity(self) -> float64:
    """Get the node's thermal diffusivity.

    Returns:
        Thermal diffusivity [:math:`m^{2}~s^{-1}`].
    """
    k = Node_get_layer_thermal_conductivity(self) / (
        Node_get_layer_density(self) * Node_get_layer_specific_heat(self)
    )
    return k


@njit(cache=False)
def Node_get_layer_thermal_effusivity(self) -> float64:
    """Get the node's thermal effusivity.

    Returns:
        Thermal effusivity [:math:`W~s^{0.5}~m^{-2}~K`].
    """
    e = math.sqrt(
        Node_get_layer_thermal_diffusivity(self)
        * Node_get_layer_density(self)
        * Node_get_layer_specific_heat(self)
    )
    return e
