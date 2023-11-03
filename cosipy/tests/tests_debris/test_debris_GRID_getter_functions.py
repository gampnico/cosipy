from collections import OrderedDict

import numpy as np
import pytest
from numba import float64, intp, optional, types

import constants
import cosipy.cpkernel.grid as cpgrid
from cosipy.cpkernel.node import BaseNodeType
from cosipy.cpkernel.patch._node import Node_init_ice_fraction

# import constants

class TestGridSpecs:
    """Tests compilation of Grid JIT specs.

    Returned types should point to the same object to avoid duplicate
    caching.
    """

    def test_init_grid_type(self):
        class_type = BaseNodeType
        test_type = types.ListType(class_type)
        compare_type = cpgrid._init_grid_type(node_type=class_type)
        assert compare_type == test_type
        assert compare_type is test_type  # cache points to same object

    def test_init_grid_jit_types(self):
        test_spec = OrderedDict()
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

    def test_grid_init(self, conftest_boilerplate):
        data = {
            "layer_heights": [0.1, 0.2, 0.3],
            "layer_densities": [200.0, 210.0, 220.0],
            "layer_temperatures": [270.0, 260.0, 250.0],
            "layer_liquid_water_content": [0.0, 0.0, 0.0],
        }

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

    def test_grid_get_node_ntype(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid

        for i in range(test_grid.number_nodes):
            conftest_boilerplate.check_output(
                test_grid.get_node_ntype(i), int, 0
            )

    def test_njit_check_node_ntype(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        assert cpgrid._check_node_ntype(test_grid, 0, 1)
        for i in range(1, test_grid.number_nodes):
            assert cpgrid._check_node_ntype(test_grid, i, 0)

    def test_njit_get_number_ntype_layers(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_snow_ice = test_grid.get_number_layers() - 1

        compare_snow_ice = cpgrid._get_number_ntype_layers(test_grid, ntype=0)
        compare_debris = cpgrid._get_number_ntype_layers(test_grid, ntype=1)
        compare_base = cpgrid._get_number_ntype_layers(test_grid, ntype=-1)

        conftest_boilerplate.check_output(compare_snow_ice, int, test_snow_ice)
        conftest_boilerplate.check_output(compare_debris, int, 1)
        assert compare_base == 0

    def test_grid_get_debris_heights(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        for i in range(test_grid.number_nodes - 1):
            test_grid.grid[i].set_layer_ntype(1)
            assert test_grid.get_node_ntype(i) == 1

        compare_heights = test_grid.get_debris_heights()
        assert isinstance(compare_heights, list)
        assert len(compare_heights) == test_grid.number_nodes - 1
        assert all(
            compare_heights[i] == self.data["layer_heights"][i]
            for i in range(test_grid.number_nodes - 1)
        )

    def test_get_number_debris_layers(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        for i in range(test_grid.number_nodes - 1):
            test_grid.grid[i].set_layer_ntype(1)
            assert test_grid.get_node_ntype(i) == 1

        compare_nlayers = test_grid.get_number_debris_layers()
        assert isinstance(compare_nlayers, int)
        assert compare_nlayers == test_grid.number_nodes - 1

    def test_njit_check_node_is_snow(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_snow = test_grid.get_number_snow_layers()
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        test_nodes = test_grid.number_nodes
        assert not cpgrid._check_node_is_snow(test_grid, 0)
        for i in range(1, test_snow):
            assert cpgrid._check_node_is_snow(test_grid, i)
        for i in range(test_snow + 1, test_nodes):
            assert not cpgrid._check_node_is_snow(test_grid, i)

    def test_get_number_snow_layers(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid

        test_nlayers = [
            1
            for idx in range(test_grid.number_nodes)
            if (test_grid.get_node_density(idx) < constants.snow_ice_threshold)
            & (test_grid.get_node_ntype(idx) != 1)
        ]

        compare_nlayers = test_grid.get_number_snow_layers()
        conftest_boilerplate.check_output(
            compare_nlayers, int, sum(test_nlayers)
        )

    def test_get_total_debris_height(
        self, conftest_boilerplate, conftest_mock_grid
    ):
        test_grid = conftest_mock_grid
        test_nlayers = [
            test_grid.grid[idx].get_layer_height()
            for idx in range(test_grid.number_nodes)
            if test_grid.get_node_ntype(idx) == 1
        ]
        compare_nlayers = test_grid.get_total_debris_height()

        conftest_boilerplate.check_output(
            compare_nlayers, float, sum(test_nlayers)
        )
