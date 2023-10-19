"""Selects Node type from user configuration.

To modify/add methods/attributes in `Node`, follow the instructions in
`cpkernel.patch._node.py`.

To modify/add methods/attributes in `DebrisNode`, follow the
instructions in cpkernel.patch._debris.py`.

To add a new node subclass to the debris implementation or change how
`BaseNode` handles subclasses, follow the instructions in
`cpkernel.patch._ctors`.
"""

from numba import float64, njit, optional
from numba.extending import register_jitable

import config

__all__ = [
    "NodeTypeRef",
    "NodeType",
    "Node",
    "BaseNodeTypeRef",
    "BaseNodeType",
    "BaseNode",
    "DebrisNodeTypeRef",
    "DebrisNodeType",
    "DebrisNode",
    "_init_node_type",
    "_create_node",
]

"""SELECT NODE TYPES"""

if not config.use_debris:  # override classes when implementing debris cover
    import cosipy.cpkernel.patch._clean

    NodeTypeRef = cosipy.cpkernel.patch._clean.NodeTypeRef
    NodeType = cosipy.cpkernel.patch._clean.NodeType
    Node = cosipy.cpkernel.patch._clean.Node
    # avoid dependency import errors
    BaseNodeTypeRef = None
    BaseNodeType = None
    BaseNode = None
    DebrisNodeTypeRef = None
    DebrisNodeType = None
    DebrisNode = None

else:
    import cosipy.cpkernel.patch.proxies

    BaseNodeTypeRef = cosipy.cpkernel.patch.proxies.BaseNodeTypeRef
    BaseNodeType = cosipy.cpkernel.patch.proxies.BaseNodeType
    BaseNode = cosipy.cpkernel.patch.proxies.BaseNode
    NodeTypeRef = cosipy.cpkernel.patch.proxies.NodeTypeRef
    NodeType = cosipy.cpkernel.patch.proxies.NodeType
    Node = cosipy.cpkernel.patch.proxies.Node
    DebrisNodeTypeRef = cosipy.cpkernel.patch.proxies.DebrisNodeTypeRef
    DebrisNodeType = cosipy.cpkernel.patch.proxies.DebrisNodeType
    DebrisNode = cosipy.cpkernel.patch.proxies.DebrisNode

"""HELPER FUNCTIONS

Declare after the node type is selected so these don't refer to the
default `Node`.
"""


@njit((float64, float64, float64, float64, optional(float64)), cache=False)
@register_jitable
def _create_node(
    height: float64,
    density: float64,
    temperature: float64,
    liquid_water_content: float64,
    ice_fraction: optional(float64),
) -> Node:
    """Create a node from user data.

    Can be used in any `jitclass`.

    Args:
        height: Layer height [:math:`m`].
        density: Layer snow density [:math:`kg~m^{-3}`].
        temperature  Layer temperature [:math:`K`].
        liquid_water_content: Liquid water content [:math:`m~w.e.`].
        ice_fraction: Volumetric ice fraction [-].

    Returns:
        A snowpack node with user-defined properties.
    """
    node = Node(
        height=height,
        density=density,
        temperature=temperature,
        liquid_water_content=liquid_water_content,
        ice_fraction=ice_fraction,
    )

    return node


@register_jitable
def _init_node_type():
    """Initialises the node types used by Grid.grid.

    Node type selection is automatically handled when importing
    `cpkernel.node`.

    Returns:
        The base type for Node objects.
    """

    if not config.use_debris:
        node_type = NodeType
    else:
        node_type = BaseNodeType

    return node_type
