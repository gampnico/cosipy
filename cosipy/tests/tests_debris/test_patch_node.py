import math

import pytest

import config
import constants
import cosipy.cpkernel.patch._node as _node
from cosipy.cpkernel.patch._ctors import Node


class TestPatchNodeGet:
    """Tests Node subtypes for snow/ice layer.

    Attributes:
        height (float): Layer height [:math:`m`]
        density (float): Snow density [:math:`kg~m^{-3}`]
        temperature (int): Layer temperature [:math:`K`]
        lwc (float): Liquid water content [:math:`m~w.e.`]
        ice_fraction (float): Volumetric ice fraction [-]
    """

    height = 0.1
    density = 200.0
    temperature = 270.0
    lwc = 0.2
    ice_fraction = 0.4

    def test_node_init(self, conftest_boilerplate):
        """Inherit methods from parent."""

        test_node = Node(
            height=self.height,
            density=self.density,
            temperature=self.temperature,
            liquid_water_content=self.lwc,
            ice_fraction=self.ice_fraction,
        )
        assert test_node
        assert isinstance(test_node, Node)

        conftest_boilerplate.check_output(
            test_node.temperature, float, self.temperature
        )
        conftest_boilerplate.check_output(test_node.height, float, self.height)
        conftest_boilerplate.check_output(
            test_node.liquid_water_content, float, self.lwc
        )
        conftest_boilerplate.check_output(test_node.refreeze, float, 0.0)
        # conftest_boilerplate.check_output(
        #     test_node.density, float, self.density
        # )

    def create_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        lwc: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> Node:
        """Instantiate a Node."""

        node = Node(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=lwc,
            ice_fraction=ice_fraction,
        )
        assert isinstance(node, Node)

        return node

    def test_create_node(self):
        node = self.create_node()
        assert isinstance(node, Node)

    @pytest.fixture(name="node", autouse=False, scope="function")
    def fixture_node(self):
        return self.create_node()

    @pytest.mark.parametrize("arg_ice_fraction", [None, 0.4])
    def test_node_init_ice_fraction(
        self, conftest_boilerplate, arg_ice_fraction
    ):
        if arg_ice_fraction is None:
            a = (
                self.density
                - (1 - (self.density / constants.ice_density))
                * constants.air_density
            )
            test_ice = a / constants.ice_density
        else:
            test_ice = arg_ice_fraction
        compare_ice = _node.Node_init_ice_fraction(
            ice_fraction=test_ice, density=self.density
        )
        conftest_boilerplate.check_output(compare_ice, float, test_ice)

    def test_node_get_layer_height(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            _node.Node_get_layer_height(node), float, self.height
        )

    def test_node_get_layer_temperature(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            _node.Node_get_layer_temperature(node), float, self.temperature
        )

    def test_node_get_layer_liquid_water_content(
        self, node, conftest_boilerplate
    ):
        assert conftest_boilerplate.check_output(
            _node.Node_get_layer_liquid_water_content(node), float, self.lwc
        )

    def test_node_get_layer_refreeze(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            _node.Node_get_layer_refreeze(node), float, 0.0
        )

    @pytest.mark.parametrize("arg_ice_fraction", [0.0, 0.1, 0.9, None])
    def test_node_get_layer_ice_fraction(
        self, conftest_boilerplate, arg_ice_fraction
    ):
        node = self.create_node(ice_fraction=arg_ice_fraction)
        if arg_ice_fraction is None:
            test_ice_fraction = (
                self.density
                - (1 - (self.density / constants.ice_density))
                * constants.air_density
            ) / constants.ice_density
        else:
            test_ice_fraction = arg_ice_fraction
        compare_ice_fraction = _node.Node_get_layer_ice_fraction(node)
        conftest_boilerplate.check_output(
            compare_ice_fraction, float, test_ice_fraction
        )
        conftest_boilerplate.check_output(
            compare_ice_fraction, float, node.ice_fraction
        )

    @pytest.mark.skipif(not config.use_debris, reason="Debris is disabled.")
    def test_node_get_ntype(self, node, conftest_boilerplate):
        conftest_boilerplate.check_output(
            _node.Node_get_layer_ntype(node), int, 0
        )

    def test_node_get_layer_air_porosity(self, node, conftest_boilerplate):
        test_porosity = max(0.0, 1 - self.lwc - self.ice_fraction)
        conftest_boilerplate.check_output(
            _node.Node_get_layer_air_porosity(node), float, test_porosity
        )

    def test_node_get_layer_density(self, node, conftest_boilerplate):
        test_density = (
            self.ice_fraction * constants.ice_density
            + self.lwc * constants.water_density
            + _node.Node_get_layer_air_porosity(node) * constants.air_density
        )
        assert conftest_boilerplate.check_output(
            _node.Node_get_layer_density(node), float, test_density
        )

    def test_node_get_layer_porosity(self, node, conftest_boilerplate):
        test_porosity = 1 - self.lwc - self.ice_fraction
        conftest_boilerplate.check_output(
            _node.Node_get_layer_porosity(node), float, test_porosity
        )

    def test_node_get_layer_specific_heat(self, node, conftest_boilerplate):
        test_specific_heat = (
            (1 - self.lwc - self.ice_fraction) * constants.spec_heat_air
            + self.ice_fraction * constants.spec_heat_ice
            + self.lwc * constants.spec_heat_water
        )
        conftest_boilerplate.check_output(
            _node.Node_get_layer_specific_heat(node), float, test_specific_heat
        )

    def test_node_get_layer_cold_content(self, node, conftest_boilerplate):
        test_cold_content = (
            -_node.Node_get_layer_specific_heat(node)
            * _node.Node_get_layer_density(node)
            * self.height
            * (self.temperature - constants.zero_temperature)
        )
        conftest_boilerplate.check_output(
            _node.Node_get_layer_cold_content(node), float, test_cold_content
        )

    def test_node_get_layer_thermal_conductivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_conductivity = (
            self.ice_fraction * constants.k_i
            + _node.Node_get_layer_porosity(node) * constants.k_a
            + self.lwc * constants.k_w
        )
        conftest_boilerplate.check_output(
            _node.Node_get_layer_thermal_conductivity(node),
            float,
            test_thermal_conductivity,
        )

    def test_node_get_layer_thermal_diffusivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_diffusivity = _node.Node_get_layer_thermal_conductivity(
            node
        ) / (
            _node.Node_get_layer_density(node)
            * _node.Node_get_layer_specific_heat(node)
        )
        conftest_boilerplate.check_output(
            _node.Node_get_layer_thermal_diffusivity(node),
            float,
            test_thermal_diffusivity,
        )

    def test_node_get_layer_thermal_effusivity(
        self, node, conftest_boilerplate
    ):
        effusivity_product = math.sqrt(
            _node.Node_get_layer_thermal_conductivity(node)
            * _node.Node_get_layer_density(node)
            * _node.Node_get_layer_specific_heat(node)
        )
        effusivity_ratio = _node.Node_get_layer_thermal_conductivity(
            node
        ) / math.sqrt(_node.Node_get_layer_thermal_diffusivity(node))
        for test_effusivity in (effusivity_product, effusivity_ratio):
            conftest_boilerplate.check_output(
                _node.Node_get_layer_thermal_effusivity(node),
                float,
                test_effusivity,
            )

    @pytest.mark.parametrize("arg_ice_fraction", [0.1, 0.9])
    def test_node_get_layer_irreducible_water_content(
        self, conftest_boilerplate, arg_ice_fraction
    ):
        node = self.create_node(ice_fraction=arg_ice_fraction)

        test_irreducible_water_content = (
            conftest_boilerplate.calculate_irreducible_water_content(
                _node.Node_get_layer_ice_fraction(node)
            )
        )
        conftest_boilerplate.check_output(
            _node.Node_get_layer_irreducible_water_content(node),
            float,
            test_irreducible_water_content,
        )
