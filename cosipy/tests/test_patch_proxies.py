import pytest
from numba.typed import List

from cosipy.cpkernel.node import BaseNode, DebrisNode, Node


@pytest.mark.PatchNode
class TestPatchProxiesBuilder:
    """Fixtures and helper functions for testing proxies.

    Attributes:
        height (float): Layer height [:math:`m`].
        density (float): Snow density [:math:`kg~m^{-3}`].
        ice_fraction (float): Volumetric ice fraction [-].
        ntype (int): Type of Node [-].
    """

    height = 0.1
    density = 200.0
    temperature = 270.0
    lwc = 0.2
    ice_fraction = 0.4
    fields = [
        "height",
        "temperature",
        "liquid_water_content",
        "ice_fraction",
        "ntype",
    ]

    def create_base_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        liquid_water_content: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> BaseNode:
        """Instantiate a Node."""

        node = BaseNode(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )
        assert isinstance(node, BaseNode)

        return node

    def test_create_base_node(self, conftest_boilerplate):
        node = self.create_base_node()
        assert isinstance(node, BaseNode)
        for attribute in self.fields:
            assert hasattr(node, attribute)
        conftest_boilerplate.check_output(node.ntype, int, -1)

    def create_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        liquid_water_content: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> Node:
        """Instantiate a Node."""

        node = Node(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )
        assert isinstance(node, Node)

        return node

    def test_create_node(self, conftest_boilerplate):
        node = self.create_node()
        assert isinstance(node, Node)
        for attribute in self.fields:
            assert hasattr(node, attribute)
        conftest_boilerplate.check_output(node.ntype, int, 0)

    def create_debris_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        liquid_water_content: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> Node:
        """Instantiate a Node."""

        node = DebrisNode(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=liquid_water_content,
            ice_fraction=ice_fraction,
        )
        assert isinstance(node, DebrisNode)

        return node

    def test_create_debris_node(self, conftest_boilerplate):
        node = self.create_debris_node()
        assert isinstance(node, DebrisNode)
        for attribute in self.fields:
            assert hasattr(node, attribute)
        conftest_boilerplate.check_output(node.ntype, int, 1)

    def check_node_list(self, node_list: List, x_class, x_ntypes: list):
        """Checks objects are correctly cast to the same type in a List.

        Args:
            node_list: Numba typed List containing casted objects.
            x_class: Expected type for cast objects.
            x_ntypes: Sorted list of expected values for `ntype`
                attributes.

        .. todo:: maybe move to `conftest_boilerplate`?
        """

        for n in node_list:
            assert isinstance(n, x_class)  # correct casting
            for attribute in self.fields:
                assert hasattr(n, attribute)  # attributes preserved

        for idx, ntype in enumerate(x_ntypes):  # ntype preserved
            assert isinstance(node_list[idx].ntype, int)
            assert node_list[idx].ntype == ntype

        for n in node_list:
            assert n.get_layer_height() == self.height
            assert n.get_layer_temperature() == self.temperature
            assert n.get_layer_liquid_water_content() == self.lwc
            if not n.ntype == 1:
                assert n.get_layer_ice_fraction() == self.ice_fraction
            else:
                assert n.get_layer_ice_fraction() == 0.0

    def create_typed_list(self, nodes: list):
        typed_list = List()
        for node in nodes:
            typed_list.append(node)
        return typed_list


@pytest.mark.PatchNode
class TestPatchProxiesCast(TestPatchProxiesBuilder):
    """Tests casting between different node classes."""

    def test_upcast_Node(self):
        base_node = self.create_base_node()
        node = self.create_node()
        test_list = self.create_typed_list([base_node, node])

        self.check_node_list(test_list, BaseNode, [-1, 0])

    def test_upcast_DebrisNode(self):
        base_node = self.create_base_node()
        debris_node = self.create_debris_node()
        test_list = self.create_typed_list([base_node, debris_node])

        self.check_node_list(test_list, BaseNode, [-1, 1])

    def test_cast_BaseNode_to_Node(self):
        node = self.create_node()
        base_node = self.create_base_node()

        # force upcasting
        test_list = self.create_typed_list([base_node, node])
        self.check_node_list(test_list, BaseNode, [-1, 0])
        upcast_node = test_list[1]

        compare_list = self.create_typed_list(
            [self.create_node(), upcast_node]
        )

        self.check_node_list(compare_list, Node, [0, 0])

    def test_cast_BaseNode_to_DebrisNode(self):
        debris_node = self.create_debris_node()
        base_node = self.create_base_node()

        # force upcasting
        test_list = self.create_typed_list([base_node, debris_node])
        self.check_node_list(test_list, BaseNode, [-1, 1])
        upcast_node = test_list[1]

        compare_list = self.create_typed_list(
            [self.create_debris_node(), upcast_node]
        )

        self.check_node_list(compare_list, DebrisNode, [1, 1])

    def test_upcast_any_to_BaseNode(self):
        base_node = self.create_base_node()
        node = self.create_node()
        debris_node = self.create_debris_node()
        test_list = self.create_typed_list([base_node, node, debris_node])

        self.check_node_list(test_list, BaseNode, [-1, 0, 1])

    def test_cast_BaseNode_to_any(self):
        debris_node = self.create_debris_node()
        node = self.create_node()
        base_node = self.create_base_node()

        # force upcasting
        test_list = self.create_typed_list([base_node, node, debris_node])
        self.check_node_list(test_list, BaseNode, [-1, 0, 1])
        upcast_node = test_list[1]
        upcast_debris_node = test_list[2]

        node_list = self.create_typed_list(
            [self.create_node(), upcast_node, upcast_debris_node]
        )

        self.check_node_list(node_list, Node, [0, 0, 1])

        debris_list = self.create_typed_list(
            [self.create_debris_node(), upcast_debris_node, upcast_node]
        )

        self.check_node_list(debris_list, DebrisNode, [1, 1, 0])
