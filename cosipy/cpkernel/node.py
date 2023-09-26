import numpy as np
from numba import float64, njit, optional
from numba.core import types
from numba.experimental import structref
from numba.extending import overload, overload_method, register_jitable

import config
import constants


@structref.register
class NodeTypeRef(types.StructRef):
    """Defines the type of reference structure used for `Node`."""

    def preprocess_fields(self, fields: dict) -> tuple:
        """Preprocess fields.

        Called by the type constructor for additional preprocessing on
        the fields. Struct shouldn't take Literal types.

        Args:
            fields: Attribute names and corresponding Literal types.

        Returns:
            Attribute names and corresponding base types.
        """

        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class Node(structref.StructRefProxy):
    """A `Node` class stores a layer's state variables.

    The numerical grid consists of a list of nodes which store the
    information of individual layers. The class provides various
    setter/getter functions to read or overwrite the state of these
    individual layers.

    If you add any new attributes or methods, ensure these are correctly
    defined or overloaded in their respective sections below
    ("Caching/Overload" and "Numba-Python Interfacing").

    Attributes:
        height (float): Layer height [:math:`m`].
        density (float): Layer snow density [:math:`kg~m^{-3}`].
        temperatur (float): Layer temperature [:math:`K`].
        liquid_water_content (float): Liquid water content
            [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
        refreeze (float): Amount of refrozen liquid water
            [:math:`m~w.e.`].
    """

    def __new__(
        cls,
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64,
        ice_fraction: optional(float64),
    ):
        """Overrides __new__.

        Required for implementing mutable reference structures. Should
        not override __init__.
        """

        self = Node_ctor(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )

        return self

    def __init__(
        self,
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64,
        ice_fraction: optional(float64) = None,
    ):
        """Initialise state & dynamic variables.

        Ensure these are not overwritten by the constructor.
        """
        self.set_layer_ice_fraction(_init_ice_fraction(ice_fraction, density))
        self.set_layer_refreeze(0.0)

    # Define attributes
    @property
    def height(self) -> float64:
        return self.get_layer_height()

    @height.setter
    def height(self, value: float64) -> None:
        self.set_layer_height(value)

    @property
    def temperature(self) -> float64:
        return self.get_layer_temperature()

    @temperature.setter
    def temperature(self, value: float64) -> None:
        self._temperature = value

    @property
    def liquid_water_content(self) -> float64:
        return self.get_layer_liquid_water_content()

    @liquid_water_content.setter
    def liquid_water_content(self, value: float64) -> None:
        self._liquid_water_content = value

    @property
    def ice_fraction(self) -> optional(float64):
        return self.get_layer_ice_fraction()

    @ice_fraction.setter
    def ice_fraction(self, value: optional(float64)) -> None:
        self._ice_fraction = value

    @property
    def refreeze(self) -> float64:
        return self.get_layer_refreeze()

    @refreeze.setter
    def refreeze(self, value: float64) -> None:
        self.set_layer_refreeze(value)

    """GETTER FUNCTIONS

    Njit and overload methods to maintain jitclass-compatibility.
    """

    # -----------------------------------------
    # Getter-functions for state variables
    # -----------------------------------------

    @njit(cache=True)
    def get_layer_height(self) -> float64:
        """Get the node's layer height.

        Returns:
            Snow layer height [:math:`m`].
        """
        return Node_get_layer_height(self)

    @njit(cache=True)
    def get_layer_temperature(self) -> float64:
        """Get the node's snow layer temperature.

        Returns:
            Snow layer temperature [:math:`K`].
        """
        return Node_get_layer_temperature(self)

    @njit(cache=True)
    def get_layer_ice_fraction(self) -> optional(float64):
        """Get the node's volumetric ice fraction.

        Returns:
            The volumetric ice fraction [-].
        """
        return Node_get_layer_ice_fraction(self)

    @njit(cache=True)
    def get_layer_liquid_water_content(self) -> float64:
        """Get the node's liquid water content.

        Returns:
            Liquid water content [-].
        """
        return Node_get_layer_liquid_water_content(self)

    @njit(cache=True)
    def get_layer_refreeze(self) -> float64:
        """Get the amount of refrozen water in the node.

        Returns:
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return Node_get_layer_refreeze(self)

    # ---------------------------------------------
    # Getter-functions for derived state variables
    # ---------------------------------------------

    @njit(cache=True)
    def get_layer_density(self) -> float64:
        """Get the node's mean density including ice and liquid.

        Returns:
            Snow density [:math:`kg~m^{-3}`].
        """
        return Node_get_layer_density(self)

    @njit(cache=True)
    def get_layer_air_porosity(self) -> float64:
        """Get the fraction of air in the node.

        Returns:
            Air porosity [:math:`m`].
        """
        return Node_get_layer_air_porosity(self)

    @njit(cache=True)
    def get_layer_specific_heat(self) -> float64:
        """Get the node's volumetrically averaged specific heat capacity.

        Returns:
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """
        return Node_get_layer_specific_heat(self)

    @njit(cache=True)
    def get_layer_irreducible_water_content(self) -> float64:
        """Get the node's irreducible water content.

        Returns:
            Irreducible water content [-].
        """
        return Node_get_layer_irreducible_water_content(self)

    @njit(cache=True)
    def get_layer_cold_content(self) -> float64:
        """Get the node's cold content.

        Returns:
            Cold content [:math:`J~m^{-2}`].
        """
        return Node_get_layer_cold_content(self)

    @njit(cache=True)
    def get_layer_porosity(self) -> float64:
        """Get the node's porosity.

        Returns:
            Air porosity [-].
        """
        return Node_get_layer_porosity(self)

    def get_layer_thermal_conductivity(self) -> float64:
        """Get the node's volumetrically weighted thermal conductivity.

        Returns:
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """
        return Node_get_layer_thermal_conductivity(self)

    def get_layer_thermal_diffusivity(self) -> float64:
        """Get the node's thermal diffusivity.

        Returns:
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """
        return Node_get_layer_thermal_diffusivity(self)

    """SETTER FUNCTIONS

    Njit and overload methods to maintain jitclass-compatibility.
    """

    # ---------------------------------------------
    # Setter-functions for derived state variables
    # ---------------------------------------------

    @njit(cache=True)
    def set_layer_height(self, height: float64):
        """Set the node's layer height.

        Args:
            height: Layer height [:math:`m`].
        """
        self.height = height

    @njit(cache=True)
    def set_layer_temperature(self, T: float64):
        """Set the node's mean temperature.

        Args:
            T: Layer temperature [:math:`K`].
        """
        self.temperature = T

    @njit(cache=True)
    def set_layer_liquid_water_content(self, lwc: float64):
        """Set the node's liquid water content.

        Args:
            lwc: Liquid water content [-].
        """
        self.liquid_water_content = lwc

    @njit(cache=True)
    def set_layer_ice_fraction(self, ifr: float64):
        """Set the node's volumetric ice fraction.

        Args:
            ifr: Volumetric ice fraction [-].
        """
        self.ice_fraction = ifr

    @njit(cache=True)
    def set_layer_refreeze(self, refr: float64):
        """Set the amount of refrozen water in the node.

        Args:
            refr: Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr


"""CACHING/OVERLOAD

Define cached or overloaded methods here.
"""


# Internal methods
@njit(cache=True)
def _init_ice_fraction(
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


"""Overload get/set methods for state attributes.

1. Define an njitted function.
2. Link the class method to use the njitted function.
3. Overload the method with the njitted function.
"""


@njit(cache=True)
def Node_get_layer_height(self) -> float64:
    return self.height


@overload_method(NodeTypeRef, "get_layer_height")
def ol_get_layer_height(self):
    def impl(self):
        return Node_get_layer_height(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_height")
def ol_set_layer_height(self, value: float64):
    def impl(self, value: float64):
        self.height = value

    return impl


@njit(cache=True)
def Node_get_layer_temperature(self) -> float64:
    return self.temperature


@overload_method(NodeTypeRef, "get_layer_temperature")
def ol_get_layer_temperature(self):
    def impl(self):
        return Node_get_layer_temperature(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_temperature")
def ol_set_layer_temperature(self, value: float64):
    def impl(self, value: float64):
        self.temperature = value

    return impl


@njit(cache=True)
def Node_get_layer_ice_fraction(self) -> float64:
    return self.ice_fraction


@overload_method(NodeTypeRef, "get_layer_ice_fraction")
def ol_get_layer_ice_fraction(self):
    def impl(self):
        return Node_get_layer_ice_fraction(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_ice_fraction")
def ol_set_layer_ice_fraction(self, value: float64):
    def impl(self, value: float64):
        self.ice_fraction = value

    return impl


@njit(cache=True)
def Node_get_layer_liquid_water_content(self) -> float64:
    return self.liquid_water_content


@overload_method(NodeTypeRef, "get_layer_liquid_water_content")
def ol_get_layer_liquid_water_content(self):
    def impl(self):
        return Node_get_layer_liquid_water_content(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_liquid_water_content")
def ol_set_layer_liquid_water_content(self, value: float64):
    def impl(self, value: float64):
        self.liquid_water_content = value

    return impl


@njit(cache=True)
def Node_get_layer_refreeze(self) -> float64:
    return self.refreeze


@overload_method(NodeTypeRef, "get_layer_refreeze")
def ol_get_layer_refreeze(self):
    def impl(self):
        return Node_get_layer_refreeze(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_refreeze")
def ol_set_layer_refreeze(self, value: float64):
    def impl(self, value: float64):
        self.refreeze = value

    return impl


"""Overload get/set methods for derived state variables.

1. Define an njitted function. Class properties (`self.foo`) are
   supported, but class methods (`self.get_foo()`) raise a signature
   error.
2. Link the class method to use the njitted function.
3. Overload the method with the njitted function.
"""


@njit(cache=True)
def Node_get_layer_air_porosity(self) -> float64:
    """Get the fraction of air in the node.

    Returns:
        Air porosity [:math:`m`].
    """
    porosity = max(0.0, 1 - self.liquid_water_content - self.ice_fraction)

    return porosity


@overload_method(NodeTypeRef, "get_layer_air_porosity")
def ol_get_layer_air_porosity(self):
    def impl(self):
        return Node_get_layer_air_porosity(self)

    return impl


@njit(cache=True)
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


@overload_method(NodeTypeRef, "get_layer_density")
def ol_get_layer_density(self):
    def impl(self):
        return Node_get_layer_density(self)

    return impl


@njit(cache=True)
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


@overload_method(NodeTypeRef, "get_layer_specific_heat")
def ol_get_layer_specific_heat(self) -> float64:
    def impl(self):
        return Node_get_layer_specific_heat(self)

    return impl


@njit(cache=True)
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


@overload_method(NodeTypeRef, "get_layer_irreducible_water_content")
def ol_get_layer_irreducible_water_content(self) -> float64:
    def impl(self):
        return Node_get_layer_irreducible_water_content(self)

    return impl


@njit(cache=True)
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


@overload_method(NodeTypeRef, "get_layer_cold_content")
def ol_get_layer_cold_content(self) -> float64:
    def impl(self):
        return Node_get_layer_cold_content(self)

    return impl


@njit(cache=True)
def Node_get_layer_porosity(self) -> float64:
    """Get the node's porosity.

    Returns:
        Air porosity [-].
    """
    return 1 - self.ice_fraction - self.liquid_water_content


@overload_method(NodeTypeRef, "get_layer_porosity")
def ol_get_layer_porosity(self) -> float64:
    def impl(self):
        return Node_get_layer_porosity(self)

    return impl


@njit(cache=True)
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
        lam = 0.021 + 2.5 * np.power((Node_get_layer_density(self) / 1000), 2)
    else:
        message = (
            "Thermal conductivity method =",
            f"{constants.thermal_conductivity_method}",
            f"is not allowed, must be one of",
            f"{', '.join(methods_allowed)}",
        )
        raise ValueError(" ".join(message))
    return lam


@overload_method(NodeTypeRef, "get_layer_thermal_conductivity")
def ol_get_layer_thermal_conductivity(self) -> float64:
    def impl(self):
        return Node_get_layer_thermal_conductivity(self)

    return impl


@njit(cache=True)
def Node_get_layer_thermal_diffusivity(self) -> float64:
    """Get the node's thermal diffusivity.

    Returns:
        Thermal diffusivity [:math:`m^{2}~s^{-1}`].
    """
    K = Node_get_layer_thermal_conductivity(self) / (
        Node_get_layer_density(self) * Node_get_layer_specific_heat(self)
    )
    return K


@overload_method(NodeTypeRef, "get_layer_thermal_diffusivity")
def ol_get_layer_thermal_diffusivity(self) -> float64:
    def impl(self):
        return Node_get_layer_thermal_diffusivity(self)

    return impl


"""NUMBA-PYTHON INTERFACING

Define attributes and constructors here.
"""

# Class attributes, including dynamic attributes
fields = [
    ("height", float64),
    ("density", float64),
    ("temperature", float64),
    ("liquid_water_content", float64),
    ("ice_fraction", optional(float64)),
    ("refreeze", float64),
]

# register types and bind the proxy
structref.define_proxy(Node, NodeTypeRef, [name[0] for name in fields])
NodeType = NodeTypeRef(fields)


@njit(cache=True)
def Node_ctor(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> Node:
    """Constructor for Node class in clean-ice simulations.

    Declaring dynamic attributes is not necessary.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature  Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].
    """

    self = structref.new(NodeType)
    self.height = height
    self.density = density
    self.temperature = temperature
    self.liquid_water_content = liquid_water_content
    # letting __init__() handle this causes a TypeError as floats are expected
    self.ice_fraction = _init_ice_fraction(ice_fraction, density)

    return self


@overload(Node)
def ol_Node(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
):
    """Override the constructor for Node."""

    def implementation(
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64,
        ice_fraction: optional(float64),
    ) -> Node:
        return Node_ctor(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )

    return implementation


"""SELECT NODE TYPES"""

if config.use_debris:  # override classes when implementing debris cover
    # BaseNodeType = cosipy.cpkernel.proxies.BaseNodeType
    # BaseNode = cosipy.cpkernel.proxies.BaseNode
    # NodeType = cosipy.cpkernel.proxies.NodeType
    # Node = cosipy.cpkernel.proxies.Node
    # DebrisNodeType = cosipy.cpkernel.proxies.DebrisNodeType
    # DebrisNode = cosipy.cpkernel.proxies.DebrisNode
    print(f"\nRunning with debris cover.")
else:
    DebrisNode = None  # avoid dependency import errors

"""HELPER FUNCTIONS

Declare after the node type is selected so these don't refer to the
default `Node`.
"""


@register_jitable
def _init_empty_node() -> Node:
    """Initialise an empty node, with all attributes set to 0 or None.

    Use this in any `jitclass` to force Numba typing.

    Returns:
        An empty node with all attributes set to 0.0. Ice fraction is
        set to None.
    """
    zero_node = Node(
        height=0.0,
        density=0.0,
        temperature=0.0,
        liquid_water_content=0.0,
        ice_fraction=None,
    )

    return zero_node


@njit(cache=True)
@register_jitable
def _create_node(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> Node:
    """Create a node from user data.

    Inherited by `cpkernel.grid`, and can be used in any `jitclass`.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature  Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].
    """
    node = Node(
        height=height,
        density=density,
        temperature=temperature,
        liquid_water_content=liquid_water_content,
        ice_fraction=ice_fraction,
    )

    return node
