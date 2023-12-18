import math

import pytest

import constants
import cosipy.cpkernel.node
import cosipy.cpkernel.patch._debris
from cosipy.cpkernel.node import DebrisNode


class TestDebrisNodeGet:
    """Tests get methods for `DebrisNode`.

    Attributes:
        height (float): Layer height [:math:`m`].
        density (float): Snow density [:math:`kg~m^{-3}`].
        temperature (int): Layer temperature [:math:`K`].
        lwc (float): Liquid water content [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
    """

    height = 0.1
    density = 2840.0  # dolomite
    temperature = 270.0
    lwc = 0.0
    ice_fraction = 0.0

    def create_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        lwc: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> DebrisNode:
        """Instantiate a Node."""
        node = cosipy.cpkernel.node.DebrisNode(
            height=height,
            density=density,
            temperature=temperature,
            liquid_water_content=lwc,
            ice_fraction=ice_fraction,
        )
        assert isinstance(node, DebrisNode)

        return node

    def test_create_node(self):
        node = self.create_node()
        assert isinstance(node, DebrisNode)

    @pytest.fixture(name="node", autouse=False, scope="function")
    def fixture_node(self):
        return self.create_node()

    def test_node_get_layer_height(self, node, conftest_boilerplate):
        for attr in (node.get_layer_height(), node.height):
            conftest_boilerplate.check_output(attr, float, self.height)

    def test_node_get_layer_temperature(self, node, conftest_boilerplate):
        for attr in (node.get_layer_temperature(), node.temperature):
            conftest_boilerplate.check_output(attr, float, self.temperature)

    def test_node_get_layer_liquid_water_content(
        self, node, conftest_boilerplate
    ):
        for attr in (
            node.get_layer_liquid_water_content(),
            node.liquid_water_content,
        ):
            conftest_boilerplate.check_output(attr, float, self.lwc)

    def test_node_get_layer_ice_fraction(self, node, conftest_boilerplate):
        for attr in (node.get_layer_ice_fraction(), node.ice_fraction):
            conftest_boilerplate.check_output(attr, float, self.ice_fraction)

    def test_node_get_layer_refreeze(self, node, conftest_boilerplate):
        for attr in (node.get_layer_refreeze(), node.refreeze):
            conftest_boilerplate.check_output(attr, float, 0.0)

    def test_node_get_layer_ntype(self, node, conftest_boilerplate):
        for attr in (node.get_layer_ntype(), node.ntype):
            conftest_boilerplate.check_output(attr, int, 1)

    @pytest.mark.parametrize("arg_void_porosity", [1.0])
    @pytest.mark.parametrize("arg_packing_porosity", [0.2596])
    def test_node_get_layer_air_porosity(
        self,
        monkeypatch,
        conftest_boilerplate,
        arg_void_porosity,
        arg_packing_porosity,
    ):
        patches = {
            "debris_void_porosity": arg_void_porosity,
            "debris_packing_porosity": arg_packing_porosity,
        }
        conftest_boilerplate.patch_variable(
            monkeypatch, cosipy.cpkernel.patch._debris.constants, patches
        )

        node = self.create_node()
        compare_porosity = node.get_layer_air_porosity()
        conftest_boilerplate.check_output(
            compare_porosity,
            float,
            constants.debris_packing_porosity * arg_void_porosity,
        )

    @pytest.mark.parametrize("arg_void_porosity", [1.0])
    @pytest.mark.parametrize("arg_packing_porosity", [0.2596])
    def test_node_get_layer_porosity(
        self,
        monkeypatch,
        conftest_boilerplate,
        arg_void_porosity,
        arg_packing_porosity,
    ):
        patches = {
            "debris_void_porosity": arg_void_porosity,
            "debris_packing_porosity": arg_packing_porosity,
        }
        conftest_boilerplate.patch_variable(
            monkeypatch, cosipy.cpkernel.patch._debris.constants, patches
        )

        test_porosity = (
            (1 - constants.debris_packing_porosity) * constants.debris_porosity
            + constants.debris_packing_porosity * arg_void_porosity
        )
        node = self.create_node()
        conftest_boilerplate.check_output(
            node.get_layer_porosity(), float, test_porosity
        )

    @pytest.mark.parametrize("arg_void_porosity", [1.0])
    @pytest.mark.parametrize("arg_packing_porosity", [0.2596])
    def test_node_get_layer_density(
        self,
        monkeypatch,
        conftest_boilerplate,
        arg_void_porosity,
        arg_packing_porosity,
    ):
        patches = {
            "debris_void_porosity": arg_void_porosity,
            "debris_packing_porosity": arg_packing_porosity,
        }
        conftest_boilerplate.patch_variable(
            monkeypatch, cosipy.cpkernel.patch._debris.constants, patches
        )

        test_air_porosity = arg_packing_porosity * arg_void_porosity
        test_porosity = (
            1 - arg_packing_porosity
        ) * constants.debris_porosity + test_air_porosity

        if arg_void_porosity >= 1.0:
            test_density = (
                test_porosity * constants.air_density
                + (1 - arg_packing_porosity)
                * constants.debris_porosity
                * constants.debris_density  # clast density
                + (1 - test_air_porosity)
                * constants.debris_void_density  # void filler density
            )
        else:
            test_density = (
                test_porosity * constants.air_density
                + (1 - test_porosity) * constants.debris_density
            )

        node = self.create_node()
        conftest_boilerplate.check_output(
            node.get_layer_density(), float, test_density
        )

    def test_node_get_layer_specific_heat(self, node, conftest_boilerplate):
        test_specific_heat = (
            node.get_layer_air_porosity() * constants.spec_heat_air
            + (1 - node.get_layer_air_porosity())
            * (constants.spec_heat_debris)
        )

        conftest_boilerplate.check_output(
            node.get_layer_specific_heat(), float, test_specific_heat
        )

    def test_node_get_layer_cold_content(self, node, conftest_boilerplate):
        test_cold_content = (
            -node.get_layer_specific_heat()
            * node.get_layer_density()
            * self.height
            * (self.temperature - constants.zero_temperature)
        )
        assert conftest_boilerplate.check_output(
            node.get_layer_cold_content(), float, test_cold_content
        )

    @pytest.mark.parametrize(
        "arg_structure", [("sedimentary", 0.0034, 0.0039)]
    )
    @pytest.mark.parametrize("arg_temperature", [270.0])
    def test_node_get_layer_thermal_conductivity(
        self, monkeypatch, conftest_boilerplate, arg_structure, arg_temperature
    ):
        patches = {"debris_structure": arg_structure[0]}
        conftest_boilerplate.patch_variable(
            monkeypatch, cosipy.cpkernel.patch._debris.constants, patches
        )

        node = self.create_node(temperature=arg_temperature)
        test_porosity = node.get_layer_air_porosity()
        test_conductivity = (1 - test_porosity) * (
            constants.thermal_conductivity_debris * 0.99
            + node.get_layer_temperature()
            * (
                arg_structure[1]
                - (arg_structure[2] / constants.thermal_conductivity_debris)
            )
        ) + (test_porosity * constants.k_a)
        assert conftest_boilerplate.check_output(
            node.get_layer_thermal_conductivity(),
            float,
            test_conductivity,
        )

    def test_node_get_layer_thermal_diffusivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_diffusivity = node.get_layer_thermal_conductivity() / (
            node.get_layer_density() * node.get_layer_specific_heat()
        )
        assert conftest_boilerplate.check_output(
            node.get_layer_thermal_diffusivity(),
            float,
            test_thermal_diffusivity,
        )

    def test_node_get_layer_thermal_effusivity(
        self, node, conftest_boilerplate
    ):
        effusivity_from_product = math.sqrt(
            node.get_layer_thermal_conductivity()
            * node.get_layer_density()
            * node.get_layer_specific_heat()
        )

        effusivity_from_ratio = (
            node.get_layer_thermal_conductivity()
            / math.sqrt(node.get_layer_thermal_diffusivity())
        )

        for test_effusivity in (
            effusivity_from_product,
            effusivity_from_ratio,
        ):
            conftest_boilerplate.check_output(
                node.get_layer_thermal_effusivity(),
                float,
                test_effusivity,
            )

    def test_node_check_thermal_relationship(self, node, conftest_boilerplate):
        """Check thermal relationship.

        Thermal conductivity/diffusivity and specific heat capacity
        should follow the relation Cp = TC / (TD * rho).
        """
        test_thermal_conductivity = (
            node.get_layer_thermal_diffusivity()
            * node.get_layer_density()
            * node.get_layer_specific_heat()
        )
        assert conftest_boilerplate.check_output(
            node.get_layer_thermal_conductivity(),
            float,
            test_thermal_conductivity,
        )
