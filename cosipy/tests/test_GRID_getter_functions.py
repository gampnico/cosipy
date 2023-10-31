from collections import OrderedDict

import numpy as np
import pytest
from numba import float64, intp, optional, types

import cosipy.cpkernel.grid as cpgrid
from cosipy.cpkernel.node import BaseNodeType, NodeType
from cosipy.cpkernel.patch._node import Node_init_ice_fraction


class TestGridSpecs:
    """Tests compilation of Grid JIT specs.

    Returned types should point to the same object to avoid duplicate
    caching.
    """

    @pytest.mark.parametrize("arg_debris", [True, False])
    def test_init_node_type(self, conftest_boilerplate, arg_debris):
        conftest_boilerplate.patch_debris(arg_debris)
        if not arg_debris:
            class_type = NodeType
        else:
            class_type = BaseNodeType
        compare_type = cpgrid._init_node_type()
        assert isinstance(compare_type, type(class_type))

    def test_init_grid_type(self):
        class_type = NodeType
        test_type = types.ListType(class_type)
        compare_type = cpgrid._init_grid_type(node_type=class_type)
        assert compare_type == test_type
        assert compare_type is test_type  # cache points to same object

    @pytest.mark.parametrize("arg_debris", [True, False])
    def test_init_grid_jit_types(
        self, monkeypatch, conftest_boilerplate, arg_debris
    ):
        conftest_boilerplate.patch_debris(arg_debris)
        node_type = cpgrid._init_node_type()
        conftest_boilerplate.patch_variable(
            monkeypatch, cpgrid, {"_NODE_TYPE": node_type}
        )

        grid_type = types.ListType(node_type)
        fields = [
            ("layer_heights", float64[:]),
            ("layer_densities", float64[:]),
            ("layer_temperatures", float64[:]),
            ("layer_liquid_water_content", float64[:]),
            ("layer_ice_fraction", optional(float64[:])),
            ("number_nodes", intp),
            ("new_snow_height", float64),
            ("new_snow_timestamp", float64),
            ("old_snow_timestamp", float64),
            ("grid", grid_type),
        ]
        test_spec = OrderedDict()
        test_spec.update(fields)

        compare_spec = cpgrid._init_grid_jit_types()

        assert isinstance(compare_spec, OrderedDict)
        assert test_spec == compare_spec


class TestGridSetup:
    """Tests initialisation methods for Grid objects."""

    node_type = cpgrid._init_node_type()
    grid_type = cpgrid._init_grid_type(node_type)
    fields = [
        ("layer_heights", float64[:]),
        ("layer_densities", float64[:]),
        ("layer_temperatures", float64[:]),
        ("layer_liquid_water_content", float64[:]),
        ("layer_ice_fraction", optional(float64[:])),
        ("number_nodes", intp),
        ("new_snow_height", float64),
        ("new_snow_timestamp", float64),
        ("old_snow_timestamp", float64),
        ("grid", grid_type),
    ]

    @pytest.mark.parametrize("arg_debris", [True, False])
    def test_grid_init(self, monkeypatch, conftest_boilerplate, arg_debris):
        data = {
            "layer_heights": [0.1, 0.2, 0.3],
            "layer_densities": [200.0, 210.0, 220.0],
            "layer_temperatures": [270.0, 260.0, 250.0],
            "layer_liquid_water_content": [0.0, 0.0, 0.0],
        }
        conftest_boilerplate.patch_debris(arg_debris)
        node_type = cpgrid._init_node_type()
        conftest_boilerplate.patch_variable(
            monkeypatch, cpgrid, {"_NODE_TYPE": node_type}
        )

        test_grid = cpgrid.Grid(
            layer_heights=float64(data["layer_heights"]),
            layer_densities=float64(data["layer_densities"]),
            layer_temperatures=float64(data["layer_temperatures"]),
            layer_liquid_water_content=float64(
                data["layer_liquid_water_content"]
            ),
        )
        assert isinstance(test_grid, cpgrid.Grid)
        conftest_boilerplate.check_output(
            test_grid.number_nodes, int, len(data["layer_heights"])
        )
        assert test_grid.layer_ice_fraction is None
        for i in range(test_grid.number_nodes):
            assert isinstance(test_grid.get_node_ice_fraction(i), float)

    def test_grid_init_grid(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_number_nodes = test_grid.number_nodes
        test_grid.init_grid()

        assert test_grid.number_nodes == test_number_nodes
        assert test_grid.layer_ice_fraction is None
        for i in range(test_grid.number_nodes):
            assert isinstance(test_grid.get_node_ice_fraction(i), float)


class TestGridGetter:
    """Tests get methods for Grid objects.

    ..
        Pytest documentation recommends `np.allclose` instead of
        `pytest.approx`.

    Attributes:
        data (dict[float64]): Dummy grid data.
    """

    data = {
        "layer_heights": float64([0.1, 0.2, 0.3, 0.5, 0.5]),
        "layer_densities": float64([250, 250, 250, 917, 917]),
        "layer_temperatures": float64([260, 270, 271, 271.5, 272]),
        "layer_liquid_water_content": float64([0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    def create_grid(self):
        grid_object = cpgrid.Grid(
            layer_heights=self.data["layer_heights"],
            layer_densities=self.data["layer_densities"],
            layer_temperatures=self.data["layer_temperatures"],
            layer_liquid_water_content=self.data["layer_liquid_water_content"],
        )
        return grid_object

    @pytest.fixture(name="grid", autouse=False, scope="function")
    def fixture_grid(self):
        return self.create_grid()

    def test_create_grid(self):
        test_grid = self.create_grid()
        assert isinstance(test_grid, cpgrid.Grid)
        assert test_grid.number_nodes == len(self.data["layer_heights"])

    def test_grid_get_height(self, grid, conftest_boilerplate):
        assert np.allclose(grid.get_height(), self.data["layer_heights"])
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_height(i), float, self.data["layer_heights"][i]
            )

    def test_grid_get_temperature(self, grid, conftest_boilerplate):
        assert np.allclose(
            grid.get_temperature(), self.data["layer_temperatures"]
        )
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_temperature(i),
                float,
                self.data["layer_temperatures"][i],
            )

    def test_grid_get_liquid_water_content(self, grid, conftest_boilerplate):
        assert np.allclose(
            grid.get_liquid_water_content(),
            self.data["layer_liquid_water_content"],
        )
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_liquid_water_content(i),
                float,
                self.data["layer_liquid_water_content"][i],
            )

    def test_grid_get_density(self, grid, conftest_boilerplate):
        assert np.allclose(grid.get_density(), self.data["layer_densities"])
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_density(i),
                float,
                self.data["layer_densities"][i],
            )

    def test_grid_get_ice_fraction(self, grid, conftest_boilerplate):
        ice_fractions = [
            Node_init_ice_fraction(None, density)
            for density in self.data["layer_densities"]
        ]
        assert np.allclose(grid.get_ice_fraction(), ice_fractions)

        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_ice_fraction(i), float, ice_fractions[i]
            )

    def test_grid_get_refreeze(self, grid, conftest_boilerplate):
        refrozen = [0.0 for i in range(grid.number_nodes)]
        assert np.allclose(grid.get_refreeze(), refrozen)

        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_refreeze(i), float, refrozen[i]
            )

    def test_grid_get_snow_ice_heights(
        self, conftest_mock_grid_values, conftest_mock_grid
    ):
        data = conftest_mock_grid_values.copy()
        GRID = conftest_mock_grid

        assert np.allclose(GRID.get_snow_heights(), data["layer_heights"][0:3])
        assert np.allclose(GRID.get_ice_heights(), data["layer_heights"][3:5])
        assert np.allclose(GRID.get_node_height(0), data["layer_heights"][0])
