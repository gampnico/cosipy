"""Defines overloaded methods and upcasting for StructRefProxies.

This only affects the debris implementation of COSIPY.

If you modified a method in an existing node subclass, then no changes
are needed here.

If you created a new method for a subclass, overload it here in the
appropriate section by inheriting your new method.

If you want to change how `BaseNode` handles subclassed methods, modify
the relevant method in "OVERLOAD BASENODE".

If you created a new subclass:
1. Import your new class (`FooNodeTypeRef`, `FooNodeType`, `FooNode`).
2. Create a new subsection in this module and add the relevant
   overloads. The easiest way is to copy/paste an existing subsection
   and change the names.
3. Define a new upcasting template at the bottom of this module.
"""

from numba import float64, njit
from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.core.extending import overload_method, register_jitable
from numba.core.imputils import lower_cast

import cosipy.cpkernel.patch._ctors as cpk_ctors
import cosipy.cpkernel.patch._debris as cpk_debris
import cosipy.cpkernel.patch._node as cpk_node

BaseNodeTypeRef = cpk_ctors.BaseNodeTypeRef
BaseNodeType = cpk_ctors.BaseNodeType
BaseNode = cpk_ctors.BaseNode

NodeTypeRef = cpk_ctors.NodeTypeRef
NodeType = cpk_ctors.NodeType
Node = cpk_ctors.Node

DebrisNodeTypeRef = cpk_ctors.DebrisNodeTypeRef
DebrisNodeType = cpk_ctors.DebrisNodeType
DebrisNode = cpk_ctors.DebrisNode

# @njit(cache=False)
@register_jitable
def verify_node_type(node, node_type):
    if not isinstance(node, node_type):
        raise TypingError(msg=f"{type(node)} not supported.")


"""OVERLOAD NODE"""

"""Overload get/set methods for Node attributes."""


@overload_method(NodeTypeRef, "get_layer_height")
def ol_get_layer_height(self):
    def impl(self):
        return cpk_node.Node_get_layer_height(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_height")
def ol_set_layer_height(self, value: float64):
    def impl(self, value: float64):
        self.height = value

    return impl


@overload_method(NodeTypeRef, "get_layer_temperature")
def ol_get_layer_temperature(self):
    def impl(self):
        return cpk_node.Node_get_layer_temperature(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_temperature")
def ol_set_layer_temperature(self, value: float64):
    def impl(self, value: float64):
        self.temperature = value

    return impl


@overload_method(NodeTypeRef, "get_layer_ice_fraction")
def ol_get_layer_ice_fraction(self):
    def impl(self):
        return cpk_node.Node_get_layer_ice_fraction(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_ice_fraction")
def ol_set_layer_ice_fraction(self, value: float64):
    def impl(self, value: float64):
        self.ice_fraction = value

    return impl


@overload_method(NodeTypeRef, "get_layer_liquid_water_content")
def ol_get_layer_liquid_water_content(self):
    def impl(self):
        return cpk_node.Node_get_layer_liquid_water_content(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_liquid_water_content")
def ol_set_layer_liquid_water_content(self, value: float64):
    def impl(self, value: float64):
        self.liquid_water_content = value

    return impl


@overload_method(NodeTypeRef, "get_layer_refreeze")
def ol_get_layer_refreeze(self):
    def impl(self):
        return cpk_node.Node_get_layer_refreeze(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_refreeze")
def ol_set_layer_refreeze(self, value: float64):
    def impl(self, value: float64):
        self.refreeze = value

    return impl


@overload_method(NodeTypeRef, "get_layer_ntype")
def ol_get_layer_ntype(self):
    def impl(self):
        return cpk_node.Node_get_layer_ntype(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_ntype")
def ol_set_layer_ntype(self, value: float64):
    def impl(self, value: float64):
        self.ntype = value

    return impl


"""Overload get/set methods for derived state variables."""


@overload_method(NodeTypeRef, "get_layer_air_porosity")
def ol_get_layer_air_porosity(self):
    def impl(self):
        return cpk_node.Node_get_layer_air_porosity(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_density")
def ol_get_layer_density(self):
    def impl(self):
        return cpk_node.Node_get_layer_density(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_specific_heat")
def ol_get_layer_specific_heat(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_specific_heat(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_irreducible_water_content")
def ol_get_layer_irreducible_water_content(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_irreducible_water_content(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_cold_content")
def ol_get_layer_cold_content(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_cold_content(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_porosity")
def ol_get_layer_porosity(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_porosity(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_thermal_conductivity")
def ol_get_layer_thermal_conductivity(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_thermal_conductivity(self)

    return impl


@overload_method(NodeTypeRef, "get_layer_thermal_diffusivity")
def ol_get_layer_thermal_diffusivity(self) -> float64:
    def impl(self):
        return cpk_node.Node_get_layer_thermal_diffusivity(self)

    return impl


"""OVERLOAD DEBRIS NODE"""

"""Overload get/set methods for `DebrisNode` attributes."""


@overload_method(DebrisNodeTypeRef, "get_layer_height")
def ol_get_layer_height(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_height(self)

    return impl


@overload_method(DebrisNodeTypeRef, "set_layer_height")
def ol_set_layer_height(self, value: float64):
    def impl(self, value: float64):
        self.height = value

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_temperature")
def ol_get_layer_temperature(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_temperature(self)

    return impl


@overload_method(DebrisNodeTypeRef, "set_layer_temperature")
def ol_set_layer_temperature(self, value: float64):
    def impl(self, value: float64):
        self.temperature = value

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_ice_fraction")
def ol_get_layer_ice_fraction(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_ice_fraction(self)

    return impl


@overload_method(DebrisNodeTypeRef, "set_layer_ice_fraction")
def ol_set_layer_ice_fraction(self, value: float64):
    def impl(self, value: float64):
        self.ice_fraction = value

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_liquid_water_content")
def ol_get_layer_liquid_water_content(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_liquid_water_content(self)

    return impl


@overload_method(DebrisNodeTypeRef, "set_layer_liquid_water_content")
def ol_set_layer_liquid_water_content(self, value: float64):
    def impl(self, value: float64):
        self.liquid_water_content = value

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_refreeze")
def ol_get_layer_refreeze(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_refreeze(self)

    return impl


@overload_method(DebrisNodeTypeRef, "set_layer_refreeze")
def ol_set_layer_refreeze(self, value: float64):
    def impl(self, value: float64):
        self.refreeze = value

    return impl


@overload_method(NodeTypeRef, "get_layer_ntype")
def ol_get_layer_ntype(self):
    def impl(self):
        return cpk_node.Node_get_layer_ntype(self)

    return impl


@overload_method(NodeTypeRef, "set_layer_ntype")
def ol_set_layer_ntype(self, value: float64):
    def impl(self, value: float64):
        self.ntype = value

    return impl


"""Overload get/set methods for derived state variables."""


@overload_method(DebrisNodeTypeRef, "get_layer_air_porosity")
def ol_get_layer_air_porosity(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_air_porosity(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_density")
def ol_get_layer_density(self):
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_density(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_specific_heat")
def ol_get_layer_specific_heat(self) -> float64:
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_specific_heat(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_cold_content")
def ol_get_layer_cold_content(self) -> float64:
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_cold_content(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_porosity")
def ol_get_layer_porosity(self) -> float64:
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_porosity(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_thermal_conductivity")
def ol_get_layer_thermal_conductivity(self) -> float64:
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_thermal_conductivity(self)

    return impl


@overload_method(DebrisNodeTypeRef, "get_layer_thermal_diffusivity")
def ol_get_layer_thermal_diffusivity(self) -> float64:
    def impl(self):
        return cpk_debris.DebrisNode_get_layer_thermal_diffusivity(self)

    return impl


"""OVERLOAD BASENODE

Patch subclass methods using conditionals.
"""

"""Overload get/set methods for `BaseNode` attributes."""


@overload_method(BaseNodeTypeRef, "get_layer_height")
def ol_get_layer_height(self):
    def impl(self):
        return cpk_node.Node_get_layer_height(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_height")
def ol_set_layer_height(self, value: float64):
    def impl(self, value: float64):
        self.height = value

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_temperature")
def ol_get_layer_temperature(self):
    def impl(self):
        return cpk_node.Node_get_layer_temperature(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_temperature")
def ol_set_layer_temperature(self, value: float64):
    def impl(self, value: float64):
        self.temperature = value

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_ice_fraction")
def ol_get_layer_ice_fraction(self):
    def impl(self):
        if self.ntype == 1:
            return cpk_debris.DebrisNode_get_layer_ice_fraction(self)
        else:
            return cpk_node.Node_get_layer_ice_fraction(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_ice_fraction")
def ol_set_layer_ice_fraction(self, value: float64):
    def impl(self, value: float64):
        self.ice_fraction = value

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_liquid_water_content")
def ol_get_layer_liquid_water_content(self):
    def impl(self):
        return cpk_node.Node_get_layer_liquid_water_content(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_liquid_water_content")
def ol_set_layer_liquid_water_content(self, value: float64):
    def impl(self, value: float64):
        self.liquid_water_content = value

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_refreeze")
def ol_get_layer_refreeze(self):
    def impl(self):
        if self.ntype == 1:
            return cpk_debris.DebrisNode_get_layer_refreeze(self)
        else:
            return cpk_node.Node_get_layer_refreeze(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_refreeze")
def ol_set_layer_refreeze(self, value: float64):
    def impl(self, value: float64):
        self.refreeze = value

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_ntype")
def ol_get_layer_ntype(self):
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        return cpk_node.Node_get_layer_ntype(self)

    return impl


@overload_method(BaseNodeTypeRef, "set_layer_ntype")
def ol_set_layer_ntype(self, value: float64):
    def impl(self, value: float64):
        self.ntype = value

    return impl


"""Overload get/set methods for derived state variables."""


@overload_method(BaseNodeTypeRef, "get_layer_air_porosity")
def ol_get_layer_air_porosity(self):
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            porosity = cpk_debris.DebrisNode_get_layer_air_porosity(self)
        else:
            porosity = cpk_node.Node_get_layer_air_porosity(self)
        return porosity

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_porosity")
def ol_get_layer_porosity(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            porosity = cpk_debris.DebrisNode_get_layer_porosity(self)
        else:
            porosity = cpk_node.Node_get_layer_porosity(self)
        return porosity

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_density")
def ol_get_layer_density(self):
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            density = cpk_debris.DebrisNode_get_layer_density(self)
        else:
            density = cpk_node.Node_get_layer_density(self)
        return density

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_specific_heat")
def ol_get_layer_specific_heat(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            heat = cpk_debris.DebrisNode_get_layer_specific_heat(self)
        else:
            heat = cpk_node.Node_get_layer_specific_heat(self)
        return heat

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_irreducible_water_content")
def ol_get_layer_irreducible_water_content(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            iwc = 0.0  # no irreducible water content in dry debris
        else:
            iwc = cpk_node.Node_get_layer_irreducible_water_content(self)
        return iwc

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_cold_content")
def ol_get_layer_cold_content(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            c_content = cpk_debris.DebrisNode_get_layer_cold_content(self)
        else:
            c_content = cpk_node.Node_get_layer_cold_content(self)
        return c_content

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_thermal_conductivity")
def ol_get_layer_thermal_conductivity(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            x = cpk_debris.DebrisNode_get_layer_thermal_conductivity(self)
        else:
            x = cpk_node.Node_get_layer_thermal_conductivity(self)
        return x

    return impl


@overload_method(BaseNodeTypeRef, "get_layer_thermal_diffusivity")
def ol_get_layer_thermal_diffusivity(self) -> float64:
    # verify_node_type(self, BaseNodeTypeRef)
    def impl(self):
        if self.ntype == 1:
            x = cpk_debris.DebrisNode_get_layer_thermal_diffusivity(self)
        else:
            x = cpk_node.Node_get_layer_thermal_diffusivity(self)
        return x

    return impl


"""HERE BE DRAGONS!

Deals with upcasting and numba's backend. Please modify with caution!

Ensure typed Lists are initialised with the `BaseNode` type, otherwise
`Node` can be cast to `DebrisNode` and vice versa.
"""


def _cast_object(
    context, builder, from_type, to_type, val, incref: bool = True
):
    """Casts an object from one type to another.

    Custom objects should have the same attributes to avoid a segfault.

    Args:
        context (numba.core.cpu.CPUContext): Context switch.
        builder (llvmlite.ir.builder.IRBuilder): Fills in LLVM
            instructions.
        from_type: Original type reference.
        to_type: Target type reference.
        val (llvmlite.ir.instructions.InsertValue): Inserts values into
            member fields.
        incref (bool): Increment reference count. Default True.
    """

    constructor = cgutils.create_struct_proxy(from_type)
    destructor = constructor(context, builder, value=val)
    meminfo = destructor.meminfo

    if incref and context.enable_nrt:
        context.nrt.incref(
            builder, types.MemInfoPointer(types.voidptr), meminfo
        )

    structure = cgutils.create_struct_proxy(to_type)(context, builder)
    structure.meminfo = meminfo

    return structure._getvalue()


"""DEFINE CASTING"""


@lower_cast(NodeType, BaseNodeType)
def upcast_Node(context, builder, fromty, toty, val):
    return _cast_object(context, builder, fromty, toty, val)


@lower_cast(DebrisNodeType, BaseNodeType)
def upcast_DebrisNode(context, builder, fromty, toty, val):
    return _cast_object(context, builder, fromty, toty, val)


@lower_cast(BaseNodeType, NodeType)
def upcast_BaseNode_to_Node(context, builder, fromty, toty, val):
    return _cast_object(context, builder, fromty, toty, val)


@lower_cast(BaseNodeType, DebrisNodeType)
def upcast_BaseNode_to_DebrisNode(context, builder, fromty, toty, val):
    return _cast_object(context, builder, fromty, toty, val)


@lower_cast(NodeType, DebrisNodeType)
def upcast_Node_to_DebrisNode(context, builder, fromty, toty, val):
    raise TypingError("Casting from Node to DebrisNode is forbidden.")


@lower_cast(DebrisNodeType, NodeType)
def upcast_DebrisNode_to_Node(context, builder, fromty, toty, val):
    raise TypingError("Casting from DebrisNode to Node is forbidden.")
