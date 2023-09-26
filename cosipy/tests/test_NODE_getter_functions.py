import pytest

import constants
from cosipy.cpkernel.node import Node, _init_empty_node, _create_node


class TestNodeGetter:
    """Tests get methods for Node objects.

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

    def calculate_irreducible_water_content(
        self, current_ice_fraction: float
    ) -> float:
        """Calculate irreducible water content."""
        if current_ice_fraction <= 0.23:
            theta_e = 0.0264 + 0.0099 * (
                (1 - current_ice_fraction) / current_ice_fraction
            )
        elif (current_ice_fraction > 0.23) & (current_ice_fraction <= 0.812):
            theta_e = 0.08 - 0.1023 * (current_ice_fraction - 0.03)
        else:
            theta_e = 0.0

        return theta_e

    @pytest.mark.parametrize("arg_ice_fraction", [0.2, 0.5, 0.9])
    def test_calculate_irreducible_water_content(self, arg_ice_fraction):
        theta_e = self.calculate_irreducible_water_content(arg_ice_fraction)
        assert isinstance(theta_e, float)

    @pytest.fixture(name="node", autouse=False, scope="function")
    def fixture_node(self):
        return self.create_node()

    def test_node_get_layer_height(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            node.get_layer_height(), float, self.height
        )

    def test_node_get_layer_temperature(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            node.get_layer_temperature(), float, self.temperature
        )

    def test_node_get_layer_liquid_water_content(
        self, node, conftest_boilerplate
    ):
        assert conftest_boilerplate.check_output(
            node.get_layer_liquid_water_content(), float, self.lwc
        )

    def test_node_get_layer_refreeze(self, node, conftest_boilerplate):
        assert conftest_boilerplate.check_output(
            node.get_layer_refreeze(), float, 0.0
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
        compare_ice_fraction = node.get_layer_ice_fraction()
        conftest_boilerplate.check_output(
            compare_ice_fraction, float, test_ice_fraction
        )
        conftest_boilerplate.check_output(
            compare_ice_fraction, float, node.ice_fraction
        )

    def test_node_get_layer_air_porosity(self, node, conftest_boilerplate):
        test_porosity = 1 - self.lwc - self.ice_fraction
        conftest_boilerplate.check_output(
            node.get_layer_air_porosity(), float, test_porosity
        )

    def test_node_get_layer_density(self, node, conftest_boilerplate):
        test_density = (
            self.ice_fraction * constants.ice_density
            + self.lwc * constants.water_density
            + node.get_layer_air_porosity() * constants.air_density
        )
        assert conftest_boilerplate.check_output(
            node.get_layer_density(), float, test_density
        )

    def test_node_get_layer_porosity(self, node, conftest_boilerplate):
        test_porosity = 1 - self.lwc - self.ice_fraction
        conftest_boilerplate.check_output(
            node.get_layer_porosity(), float, test_porosity
        )

    def test_node_get_layer_specific_heat(self, node, conftest_boilerplate):
        test_specific_heat = (
            (1 - self.lwc - self.ice_fraction) * constants.spec_heat_air
            + self.ice_fraction * constants.spec_heat_ice
            + self.lwc * constants.spec_heat_water
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
        conftest_boilerplate.check_output(
            node.get_layer_cold_content(), float, test_cold_content
        )

    def test_node_get_layer_thermal_conductivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_conductivity = (
            self.ice_fraction * constants.k_i
            + node.get_layer_porosity() * constants.k_a
            + self.lwc * constants.k_w
        )
        conftest_boilerplate.check_output(
            node.get_layer_thermal_conductivity(),
            float,
            test_thermal_conductivity,
        )

    def test_node_get_layer_thermal_diffusivity(
        self, node, conftest_boilerplate
    ):
        test_thermal_diffusivity = node.get_layer_thermal_conductivity() / (
            node.get_layer_density() * node.get_layer_specific_heat()
        )
        conftest_boilerplate.check_output(
            node.get_layer_thermal_diffusivity(),
            float,
            test_thermal_diffusivity,
        )

    @pytest.mark.parametrize("arg_ice_fraction", [0.1, 0.9])
    def test_node_get_layer_irreducible_water_content(
        self, conftest_boilerplate, arg_ice_fraction
    ):
        node = self.create_node(ice_fraction=arg_ice_fraction)

        test_irreducible_water_content = (
            self.calculate_irreducible_water_content(
                node.get_layer_ice_fraction()
            )
        )
        conftest_boilerplate.check_output(
            node.get_layer_irreducible_water_content(),
            float,
            test_irreducible_water_content,
        )

    def test_init_empty_node(self, conftest_boilerplate):
        test_node = self.create_node()
        compare_node = _init_empty_node()

        assert isinstance(compare_node, Node)
        conftest_boilerplate.check_output(
            variable=test_node.temperature,
            x_type=float,
            x_value=self.temperature,
        )
        conftest_boilerplate.check_output(test_node.height, float, self.height)
        conftest_boilerplate.check_output(
            test_node.liquid_water_content, float, self.lwc
        )
        conftest_boilerplate.check_output(test_node.refreeze, float, 0.0)

    def test_node_create_node(self, conftest_boilerplate):
        test_node = self.create_node()
        compare_node = _create_node(
            height=self.height,
            density=self.density,
            temperature=self.temperature,
            liquid_water_content=self.lwc,
            ice_fraction=self.ice_fraction,
        )

        assert isinstance(compare_node, Node)
        conftest_boilerplate.check_output(
            test_node.temperature, float, self.temperature
        )
        conftest_boilerplate.check_output(test_node.height, float, self.height)
        conftest_boilerplate.check_output(
            test_node.liquid_water_content, float, self.lwc
        )
        conftest_boilerplate.check_output(test_node.refreeze, float, 0.0)
