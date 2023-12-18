import math

import pytest

import config
import constants
import cosipy.cpkernel.node
import cosipy.cpkernel.patch._debris as _debris
from cosipy.cpkernel.patch._ctors import DebrisNode


class TestPatch_debrisGet:
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
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_height(node), float, self.height
        )

    def test_node_get_layer_temperature(self, node, conftest_boilerplate):
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_temperature(node),
            float,
            self.temperature,
        )

    def test_node_get_layer_liquid_water_content(
        self, node, conftest_boilerplate
    ):
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_liquid_water_content(node),
            float,
            self.lwc,
        )

    def test_node_get_layer_ice_fraction(self, node, conftest_boilerplate):
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_ice_fraction(node),
            float,
            self.ice_fraction,
        )

    def test_node_get_layer_refreeze(self, node, conftest_boilerplate):
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_refreeze(node), float, 0.0
        )

    def test_node_get_layer_ntype(self, node, conftest_boilerplate):
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_ntype(node), int, 1
        )

    @pytest.mark.parametrize("arg_void_porosity", [1.0, 0.3])
    @pytest.mark.parametrize("arg_packing_porosity", [0.2596, 0.4764])
    def test_node_get_layer_air_porosity(
        self,
        monkeypatch,
        node,
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
        njit_porosity = _debris.DebrisNode_get_layer_air_porosity(node)
        conftest_boilerplate.check_output(njit_porosity, float, 0.2596)

        compare_porosity = _debris.DebrisNode_get_layer_air_porosity.py_func(
            node
        )
        conftest_boilerplate.check_output(
            compare_porosity,
            float,
            arg_packing_porosity * arg_void_porosity,
        )

    @pytest.mark.parametrize("arg_void_porosity", [1.0, 0.3])
    @pytest.mark.parametrize("arg_packing_porosity", [0.2596, 0.4764])
    def test_node_get_layer_porosity(
        self,
        monkeypatch,
        node,
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
        test_porosity = (
            1 - arg_packing_porosity
        ) * constants.debris_porosity + _debris.DebrisNode_get_layer_air_porosity(
            node
        )
        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_porosity.py_func(node),
            float,
            test_porosity,
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
            _debris.DebrisNode_get_layer_density(node), float, test_density
        )

    def test_node_get_layer_specific_heat(self, node, conftest_boilerplate):
        test_specific_heat = _debris.DebrisNode_get_layer_air_porosity(
            node
        ) * constants.spec_heat_air + (
            1 - _debris.DebrisNode_get_layer_air_porosity(node)
        ) * (
            constants.spec_heat_debris
        )

        conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_specific_heat(node),
            float,
            test_specific_heat,
        )

    def test_node_get_layer_cold_content(self, node, conftest_boilerplate):
        test_cold_content = (
            -_debris.DebrisNode_get_layer_specific_heat(node)
            * _debris.DebrisNode_get_layer_density(node)
            * self.height
            * (self.temperature - constants.zero_temperature)
        )
        assert conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_cold_content(node),
            float,
            test_cold_content,
        )

    @pytest.mark.parametrize(
        "arg_structure",
        [("sedimentary", 0.0034, 0.0039), ("crystalline", 0.0030, 0.0042)],
    )
    @pytest.mark.parametrize("arg_temperature", [270.0, 273.15, 280.0])
    def test_node_get_layer_thermal_conductivity(
        self,
        monkeypatch,
        node,
        conftest_boilerplate,
        arg_structure,
        arg_temperature,
    ):
        patches = {"debris_structure": arg_structure[0]}
        conftest_boilerplate.patch_variable(
            monkeypatch, cosipy.cpkernel.patch._debris.constants, patches
        )

        node = self.create_node(temperature=arg_temperature)
        test_porosity = _debris.DebrisNode_get_layer_air_porosity(node)
        test_conductivity = (1 - test_porosity) * (
            constants.thermal_conductivity_debris * 0.99
            + _debris.DebrisNode_get_layer_temperature(node)
            * (
                arg_structure[1]
                - (arg_structure[2] / constants.thermal_conductivity_debris)
            )
        ) + (test_porosity * constants.k_a)
        assert conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_thermal_conductivity.py_func(node),
            float,
            test_conductivity,
        )

    def test_node_get_layer_thermal_diffusivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_diffusivity = (
            _debris.DebrisNode_get_layer_thermal_conductivity(node)
            / (
                _debris.DebrisNode_get_layer_density(node)
                * _debris.DebrisNode_get_layer_specific_heat(node)
            )
        )
        assert conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_thermal_diffusivity(node),
            float,
            test_thermal_diffusivity,
        )

    def test_node_get_layer_thermal_effusivity(
        self, node, conftest_boilerplate
    ):
        effusivity_product = math.sqrt(
            _debris.DebrisNode.DebrisNode_get_layer_thermal_conductivity(node)
            * _debris.DebrisNode.DebrisNode_get_layer_density(node)
            * _debris.DebrisNode.DebrisNode_get_layer_specific_heat(node)
        )
        effusivity_ratio = (
            _debris.DebrisNode.DebrisNode_get_layer_thermal_conductivity(node)
            / math.sqrt(
                _debris.DebrisNode.DebrisNode_get_layer_thermal_diffusivity(
                    node
                )
            )
        )
        for test_effusivity in (effusivity_product, effusivity_ratio):
            conftest_boilerplate.check_output(
                _debris.DebrisNode_get_layer_thermal_effusivity(node),
                float,
                test_effusivity,
            )

    def test_node_check_thermal_relationship(self, node, conftest_boilerplate):
        """Check thermal relationship.

        Thermal conductivity/diffusivity and specific heat capacity
        should follow the relation Cp = TC / (TD * rho).
        """
        test_thermal_conductivity = (
            _debris.DebrisNode_get_layer_thermal_diffusivity(node)
            * _debris.DebrisNode_get_layer_density(node)
            * _debris.DebrisNode_get_layer_specific_heat(node)
        )
        assert conftest_boilerplate.check_output(
            _debris.DebrisNode_get_layer_thermal_conductivity(node),
            float,
            test_thermal_conductivity,
        )
