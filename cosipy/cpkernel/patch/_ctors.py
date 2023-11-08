"""Defines and registers proxies used for debris-covered simulations.

This is the debris implementation of COSIPY. It is entirely separate
from the clean-ice implementation.

WARNING! `BaseNode` and its subclasses must have the same attributes,
otherwise upcasting may cause segfaults. Methods can be unique between
subclasses, as long as they are also defined in `BaseNode`.

To add an attribute:
1. Go to the "NUMBA-PYTHON INTERFACING" subsection below.
2. Add the attribute to `ctor_fields`.
3. If the attribute is NOT dynamic (i.e. passed by an argument), add it
   to `BaseNode_ctor`, `DebrisNode_ctor` and `Node_ctor`, along with
   their `__new__`, and `__init__` methods.
4. Define the attribute as a property in `BaseNode`, `Node`, and
   `DebrisNode`.
5. Go to `cpkernel.patch._node` AND `cpkernel.patch._debris` and follow
   the instructions there to add any relevant get/set methods.

To add a new method:
1. Follow the instructions in `cpkernel.patch._node` and/or
   `cpkernel.patch._debris` if you haven't already done so. Methods
   don't need to be defined for all subclasses.
2. Add a method in `Node` and/or `DebrisNode` which inherits the new
   njitted function.
3. Add a method to `BaseNode` which calls itself - just copy/paste a
   neighbouring method and change the name.
4. Go to `cpkernel.patch._proxies` and follow the instructions there.

To add a new subclass, use an existing subclass as a template:
1. Define a TypeRef which inherits from BaseNodeTypeRef.
2. Define a subclass which inherits from BaseNode.
3. Define an njitted constructor for your subclass.
4. Overload your new subclass with the constructor.
5. Register and bind the proxy.
6. Create a new module in `cpkernel.patch` which contains all the
methods you wish to use in your subclass.
7. Go to `cpkernel.patch.proxies` and define casting, overloads etc.
"""

from numba import float64, int64, njit, optional
from numba.core import types
from numba.core.extending import overload
from numba.experimental import structref

import cosipy.cpkernel.patch._debris as cpk_debris
import cosipy.cpkernel.patch._node as cpk_node

# register new subclasses here
__all__ = [
    "BaseNodeTypeRef",
    "BaseNodeType",
    "BaseNode",
    "NodeTypeRef",
    "NodeType",
    "Node",
    "DebrisNodeTypeRef",
    "DebrisNodeType",
    "DebrisNode",
]

"""NUMBA-PYTHON INTERFACING

Class attributes, including dynamic attributes. Must be identical across
all subclasses.
"""

ctor_fields = [
    ("height", float64),
    ("density", float64),
    ("temperature", float64),
    ("liquid_water_content", float64),
    ("ice_fraction", optional(float64)),
    ("refreeze", float64),
    ("ntype", int64),
]


@structref.register
class BaseNodeTypeRef(types.StructRef):
    """Defines the type of reference structure used for `BaseNode`."""

    def preprocess_fields(self, fields):
        """Preprocess fields.

        Called by the type constructor for additional preprocessing on
        the fields. Struct shouldn't take Literal types.
        """

        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class BaseNode(structref.StructRefProxy):
    """Parent for all node subtypes.

    Attributes:
        height (float): Height of the debris layer [:math:`m`].
        density (float): Debris density [:math:`kg~m^{-3}`].
        temperature (float): Layer temperature [:math:`K`].
        liquid_water_content (float): Liquid water content
            [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
        refreeze (float): Amount of refrozen water [:math:`m~w.e.`].
        ntype (int): Node subtype:
            * 0: `Node` (snow/ice)
            * 1: `DebrisNode` (debris)
            * -1: `BaseNode` (parent class)
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

        self = BaseNode_ctor(
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
        # Initialises state variables.
        self.set_layer_ntype(int64(-1))

    @property
    def height(self) -> float64:
        return self.get_layer_height()

    @height.setter
    def height(self, value: float64) -> None:
        self._height = value

    @njit(cache=False)
    def get_layer_height(self) -> float64:
        return self.height

    @njit(cache=False)
    def set_layer_height(self, value) -> None:
        self.height = value

    @property
    def temperature(self) -> float64:
        return self.get_layer_temperature()

    @temperature.setter
    def temperature(self, value: float64) -> None:
        self._temperature = value

    @njit(cache=False)
    def get_layer_temperature(self) -> float64:
        return self.temperature

    @njit(cache=False)
    def set_layer_temperature(self, value) -> None:
        self.temperature = value

    @property
    def liquid_water_content(self) -> float64:
        return self.get_layer_liquid_water_content()

    @liquid_water_content.setter
    def liquid_water_content(self, value: float64) -> None:
        self._liquid_water_content = value

    @njit(cache=False)
    def get_layer_liquid_water_content(self) -> float64:
        return self.liquid_water_content

    @njit(cache=False)
    def set_layer_liquid_water_content(self, value) -> None:
        self.liquid_water_content = value

    @property
    def ice_fraction(self) -> optional(float64):
        return self.get_layer_ice_fraction()

    @ice_fraction.setter
    def ice_fraction(self, value: optional(float64)) -> None:
        self._ice_fraction = value

    @njit(cache=False)
    def get_layer_ice_fraction(self) -> optional(float64):
        return self.ice_fraction

    @property
    def refreeze(self) -> float64:
        return self.get_layer_refreeze()

    @refreeze.setter
    def refreeze(self, value: float64) -> None:
        self.set_layer_refreeze(value)

    @property
    def ntype(self) -> int64:
        return self.get_layer_ntype()

    @ntype.setter
    def ntype(self, idx: int64) -> None:
        self.set_layer_ntype(idx)

    @njit(cache=False)
    def get_layer_ntype(self) -> int64:
        return self.ntype

    @njit(cache=False)
    def set_layer_ntype(self, idx: int64) -> None:
        self.ntype = idx

    @njit(cache=False)
    def get_layer_air_porosity(self) -> float64:
        return self.get_layer_air_porosity()

    @njit(cache=False)
    def get_layer_porosity(self) -> float64:
        return self.get_layer_porosity()

    @njit(cache=False)
    def get_layer_density(self) -> float64:
        return self.get_layer_density()

    @njit(cache=False)
    def get_layer_specific_heat(self) -> float64:
        return self.get_layer_specific_heat()

    @njit(cache=False)
    def get_layer_irreducible_water_content(self) -> float64:
        return self.get_layer_irreducible_water_content()

    @njit(cache=False)
    def get_layer_cold_content(self) -> float64:
        return self.get_layer_cold_content()

    @njit(cache=False)
    def get_layer_thermal_conductivity(self) -> float64:
        return self.get_layer_thermal_conductivity()

    @njit(cache=False)
    def get_layer_thermal_diffusivity(self) -> float64:
        return self.get_layer_thermal_diffusivity()


# register types and bind the proxy
structref.define_proxy(
    BaseNode, BaseNodeTypeRef, [name[0] for name in ctor_fields]
)
BaseNodeType = BaseNodeTypeRef(ctor_fields)


@njit(cache=False)
def BaseNode_ctor(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> BaseNode:
    """Constructor for BaseNode class in debris-covered simulations.

    Declaring dynamic attributes is unnecessary.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature: Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].
    """

    self = structref.new(BaseNodeType)
    self.height = height
    self.density = density
    self.temperature = temperature
    self.liquid_water_content = liquid_water_content
    self.ice_fraction = ice_fraction
    self.ntype = int64(-1)

    return self


@overload(BaseNode)
def ol_BaseNode(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
):
    """Override the constructor for BaseNode."""

    def implementation(
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64,
        ice_fraction: optional(float64),
    ):
        return BaseNode_ctor(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )

    return implementation


@structref.register
class NodeTypeRef(BaseNodeTypeRef):
    """Defines the type of reference structure used for `Node`."""

    def preprocess_fields(self, fields):
        """Preprocess fields.

        Called by the type constructor for additional preprocessing on
        the fields. Struct shouldn't take Literal types.
        """

        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class Node(BaseNode):
    """A `Node` class stores a snow layer's state variables.

    The numerical grid consists of a list of nodes which store the
    information of individual layers. The class provides various
    setter/getter functions to read or overwrite the state of these
    individual layers.

    Attributes:
        height (float): Layer height [:math:`m`].
        snow_density (float): Layer snow density [:math:`kg~m^{-3}`].
        temperature (float): Layer temperature [:math:`K`].
        liquid_water_content (float): Liquid water content
            [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
        refreeze (float): Amount of refrozen water [:math:`m~w.e.`].
        ntype (int): Node subtype. Default "0" for snow/ice.
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

        self = NodeSubClass_ctor(
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
        # Initialise state & dynamic variables.
        self.set_layer_ice_fraction(
            cpk_node.Node_init_ice_fraction(ice_fraction, density)
        )
        self.set_layer_refreeze(float64(0.0))
        self.set_layer_ntype(int64(0))

    """GETTER FUNCTIONS"""

    @njit(cache=False)
    def get_layer_height(self) -> float64:
        """Get the node's layer height.

        Returns:
            Snow layer height [:math:`m`].
        """
        return cpk_node.Node_get_layer_height(self)

    @njit(cache=False)
    def get_layer_temperature(self) -> float64:
        """Get the node's snow layer temperature.

        Returns:
            Snow layer temperature [:math:`K`].
        """
        return cpk_node.Node_get_layer_temperature(self)

    @njit(cache=False)
    def get_layer_ice_fraction(self) -> optional(float64):
        """Get the node's volumetric ice fraction.

        Returns:
            The volumetric ice fraction [-].
        """
        return cpk_node.Node_get_layer_ice_fraction(self)

    @njit(cache=False)
    def get_layer_liquid_water_content(self) -> float64:
        """Get the node's liquid water content.

        Returns:
            Liquid water content [-].
        """
        return cpk_node.Node_get_layer_liquid_water_content(self)

    @njit(cache=False)
    def get_layer_refreeze(self) -> float64:
        """Get the amount of refrozen water in the node.

        Returns:
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return cpk_node.Node_get_layer_refreeze(self)

    @njit(cache=False)
    def get_layer_ntype(self) -> int64:
        return cpk_node.Node_get_layer_ntype(self)

    # ---------------------------------------------
    # Getter-functions for derived state variables
    # ---------------------------------------------

    @njit(cache=False)
    def get_layer_air_porosity(self) -> float64:
        """Get the fraction of air in the node.

        Returns:
            Air porosity [:math:`m`].
        """
        return cpk_node.Node_get_layer_air_porosity(self)

    @njit(cache=False)
    def get_layer_porosity(self) -> float64:
        """Get the node's porosity.

        Returns:
            Air porosity [-].
        """
        return cpk_node.Node_get_layer_porosity(self)

    @njit(cache=False)
    def get_layer_density(self) -> float64:
        """Get the node's mean density including ice and liquid.

        Returns:
            Snow density [:math:`kg~m^{-3}`].
        """
        return cpk_node.Node_get_layer_density(self)

    @njit(cache=False)
    def get_layer_specific_heat(self) -> float64:
        """Get the node's volumetrically averaged specific heat capacity.

        Returns:
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """
        return cpk_node.Node_get_layer_specific_heat(self)

    @njit(cache=False)
    def get_layer_irreducible_water_content(self) -> float64:
        """Get the node's irreducible water content.

        Returns:
            Irreducible water content [-].
        """
        return cpk_node.Node_get_layer_irreducible_water_content(self)

    @njit(cache=False)
    def get_layer_cold_content(self) -> float64:
        """Get the node's cold content.

        Returns:
            Cold content [:math:`J~m^{-2}`].
        """
        return cpk_node.Node_get_layer_cold_content(self)

    @njit(cache=False)
    def get_layer_thermal_conductivity(self) -> float64:
        """Get the node's volumetrically weighted thermal conductivity.

        Returns:
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """
        return cpk_node.Node_get_layer_thermal_conductivity(self)

    @njit(cache=False)
    def get_layer_thermal_diffusivity(self) -> float64:
        """Get the node's thermal diffusivity.

        Returns:
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """
        return cpk_node.Node_get_layer_thermal_diffusivity(self)

    """SETTER FUNCTIONS"""

    # ---------------------------------------------
    # Setter-functions for derived state variables
    # ---------------------------------------------

    @njit(cache=False)
    def set_layer_height(self, height: float64):
        """Sets the node's layer height.

        Args:
            height: Layer height [:math:`m`].
        """
        self.height = height

    @njit(cache=False)
    def set_layer_temperature(self, T: float64):
        """Sets the node's mean temperature.

        Args:
            T: Layer temperature [:math:`K`].
        """
        self.temperature = T

    @njit(cache=False)
    def set_layer_liquid_water_content(self, lwc: float64):
        """Sets the node's liquid water content.

        Args:
            lwc: Liquid water content [-].
        """
        self.liquid_water_content = lwc

    @njit(cache=False)
    def set_layer_ice_fraction(self, ifr: float64):
        """Sets the node's volumetric ice fraction.

        Args:
            ifr: Volumetric ice fraction [-].
        """
        self.ice_fraction = ifr

    @njit(cache=False)
    def set_layer_refreeze(self, refr: float64):
        """Sets the amount of refrozen water in the node.

        Args:
            refr: Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr

    @njit(cache=False)
    def set_layer_ntype(self, ntype: int64):
        """Sets node's subtype.

        Args:
            ntype (int): Node subclass type.
        """
        self.ntype = ntype


"""Node Constructor"""


@njit(cache=False)
def NodeSubClass_ctor(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> Node:
    """Constructor for Node class in debris-covered simulations.

    Declaring dynamic attributes is unnecessary.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature: Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].
    """

    self = structref.new(NodeType)
    self.height = height
    self.density = density
    self.temperature = temperature
    self.liquid_water_content = liquid_water_content
    # letting __init__() handle this causes a TypeError as floats are expected
    self.ice_fraction = cpk_node.Node_init_ice_fraction(ice_fraction, density)
    self.ntype = int64(0)

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
    ):
        return NodeSubClass_ctor(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )

    return implementation


# register types and bind the proxy
structref.define_proxy(Node, NodeTypeRef, [name[0] for name in ctor_fields])
NodeType = NodeTypeRef(ctor_fields)
"""DEBRIS NODE"""


@structref.register
class DebrisNodeTypeRef(BaseNodeTypeRef):
    """Defines the type of reference structure used for `Node`."""

    def preprocess_fields(self, fields):
        """Preprocess fields.

        Called by the type constructor for additional preprocessing on
        the fields. Struct shouldn't take Literal types.
        """
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class DebrisNode(BaseNode):
    """Stores the state variables of a debris layer.

    The numerical grid consists of a list of nodes that store the
    information of individual debris layers. The class provides various
    setter/getter functions to read or overwrite the state of an
    individual debris layer.

    Attributes such as liquid water content/ice fraction are retained in
    case of future extensions for wet debris cover.

    Attributes:
        height (float): Height of the debris layer [:math:`m`].
        density (float): Debris density [:math:`kg~m^{-3}`].
        temperature (float): Debris layer temperature [:math:`K`].
        liquid_water_content (float): Liquid water content
            [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
        refreeze (float): Amount of refrozen water [:math:`m~w.e.`].
        ntype (int): Node subclass type. Default "1" for debris.
    """

    def __new__(
        cls,
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64 = 0.0,
        ice_fraction: optional(float64) = None,
    ):
        """Overrides __new__.

        Required for implementing mutable reference structures. Should
        not override __init__.
        """

        if ice_fraction is None:
            ice_fraction = float64(0.0)
        else:
            ice_fraction = float64(ice_fraction)

        self = DebrisNodeSubClass_ctor(
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
        # Initialize state variables
        self.set_layer_ntype(int64(1))
        self.set_layer_ice_fraction(float64(0.0))

    """GETTER FUNCTIONS"""

    @njit(cache=False)
    def get_layer_height(self) -> float64:
        """Get the node's layer height.

        Returns:
            float: Debris layer height [:math:`m`].
        """
        return cpk_debris.DebrisNode_get_layer_height(self)

    @njit(cache=False)
    def get_layer_temperature(self) -> float64:
        """Get the node's layer temperature.

        Returns:
            Debris layer temperature [:math:`K`].
        """
        return cpk_debris.DebrisNode_get_layer_temperature(self)

    @njit(cache=False)
    def get_layer_liquid_water_content(self) -> float64:
        """Get the node's liquid water content.

        Returns:
            Liquid water content [-].
        """
        return cpk_debris.DebrisNode_get_layer_liquid_water_content(self)

    @njit(cache=False)
    def get_layer_ice_fraction(self) -> optional(float64):
        """Get the node's volumetric ice fraction.

        Returns:
            The volumetric ice fraction [-].
        """
        return cpk_debris.DebrisNode_get_layer_ice_fraction(self)

    @njit(cache=False)
    def get_layer_refreeze(self) -> float64:
        """Get the amount of refrozen water in the node.

        Returns:
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return cpk_debris.DebrisNode_get_layer_refreeze(self)

    @njit(cache=False)
    def get_layer_ntype(self) -> int64:
        return cpk_debris.DebrisNode_get_layer_ntype(self)

    # Derived state variables

    @njit(cache=False)
    def get_layer_air_porosity(self) -> float64:
        """Get the node's volumetrically-weighted interstitial void porosity.

        The function's name is kept as `get_layer_air_porosity` for
        cross-compatibility with other Node objects.

        Does NOT include the debris' porosity, and assumes no liquid.
        Note that the packing and void porosities are
        volumetrically-weighted!

        Returns:
            float: Interstitial void porosity [-].
        """

        return cpk_debris.DebrisNode_get_layer_air_porosity(self)

    @njit(cache=False)
    def get_layer_porosity(self) -> float64:
        return cpk_debris.DebrisNode_get_layer_porosity(self)

    @njit(cache=False)
    def get_layer_density(self) -> float64:
        return cpk_debris.DebrisNode_get_layer_density(self)

    @njit(cache=False)
    def get_layer_specific_heat(self) -> float64:
        """Get the node's volumetrically averaged specific heat capacity.

        Returns:
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """
        return cpk_debris.DebrisNode_get_layer_specific_heat(self)

    @njit(cache=False)
    def get_layer_cold_content(self) -> float64:
        """Get the node's cold content.

        Returns:
            Cold content [:math:`J~m^{-2}`].
        """
        return cpk_debris.DebrisNode_get_layer_cold_content(self)

    @njit(cache=False)
    def get_layer_thermal_conductivity(self) -> float64:
        """Get the node's volumetrically weighted thermal conductivity.

        Returns:
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """
        return cpk_debris.DebrisNode_get_layer_thermal_conductivity(self)

    @njit(cache=False)
    def get_layer_thermal_diffusivity(self) -> float64:
        """Get the node's thermal diffusivity.

        Returns:
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """
        return cpk_debris.DebrisNode_get_layer_thermal_diffusivity(self)

    """SETTER FUNCTIONS"""

    # ----------------------------------------------
    # Setter-functions for derived state variables
    # ----------------------------------------------

    @njit(cache=False)
    def set_layer_height(self, height: float64) -> None:
        """Sets the node's layer height.

        Args:
            height: Layer height [:math:`m`].
        """
        self.height = height

    @njit(cache=False)
    def set_layer_ice_fraction(self, ifr: float64) -> None:
        """Sets the node's ice fraction.

        Args:
            ifr: Ice fraction [-].
        """
        self.ice_fraction = ifr

    @njit(cache=False)
    def set_layer_refreeze(self, refr: float64):
        """Sets the amount of refrozen water in the node.

        Args:
            refr: Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr

    @njit(cache=False)
    def set_layer_ntype(self, ntype: int64):
        """Sets node's subtype.

        Args:
            ntype (int): Node subclass type.
        """
        self.ntype = ntype


"""DebrisNode Constructor"""


@njit(cache=False)
def DebrisNodeSubClass_ctor(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> DebrisNode:
    """Constructor for Node class in debris-covered simulations.

    Declaring dynamic attributes is unnecessary.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature: Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].
    """

    self = structref.new(DebrisNodeType)
    self.height = height
    self.density = density
    self.temperature = temperature
    self.liquid_water_content = liquid_water_content
    self.ice_fraction = ice_fraction
    self.ntype = int64(1)

    return self


@overload(DebrisNode)
def ol_DebrisNode(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
):
    """Override the constructor for DebrisNode."""

    def implementation(
        height: float64,
        density: float64,
        temperature: float64,
        liquid_water_content: float64,
        ice_fraction: optional(float64),
    ):
        return DebrisNodeSubClass_ctor(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )

    return implementation


# register types and bind the proxy
structref.define_proxy(
    DebrisNode, DebrisNodeTypeRef, [name[0] for name in ctor_fields]
)
DebrisNodeType = DebrisNodeTypeRef(ctor_fields)
