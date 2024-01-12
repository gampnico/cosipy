"""Provides additional shared fixtures and methods for debris tests.

Use these to replace duplicated code. These fixtures are only accessible
by tests within the `tests_debris` folder. Fixtures from conftest in
parent folders are still accessible.

For generating objects within a test function's scope, call a fixture
directly:

    .. code-block:: python
        
        def test_foobar(self, conftest_mock_grid):
            grid_object = conftest_mock_grid
            grid_object.set_foo(foo=bar)
            ...
"""

import numpy as np
import pytest
from numba import float64, intp, optional

import cosipy.cpkernel.grid as cpgrid


@pytest.fixture(
    name="conftest_debris_mock_fields", scope="function", autouse=False
)
def fixture_conftest_debris_mock_fields() -> list:
    """Construct the fields used to define JIT types.

    Returns:
        List of fields and corresponding numba type.
    """

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

    yield fields


class TestDebrisBoilerplate:
    """Provides boilerplate methods for serialising tests on debris.

    The class is instantiated via the `conftest_debris_boilerplate`
    fixture. The fixture is autoused, and can be called directly within
    a test::

    ..code-block:: python

        def test_foo(self, conftest_debris_boilerplate):

            foobar = [...]
            conftest_debris_boilerplate.bar(foobar)

    Methods are arranged with their appropriate test::

    .. code-block:: python

        def foo(self, ...):
            pass

        def test_foo(self ...):
            pass
    """

    def add_debris_to_grid(self, grid_obj: cpgrid.Grid):
        """Add a debris layer."""
        grid_obj.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)

    def test_add_debris_to_grid(self):
        """Add debris layer using fixture."""
        data = {}
        data["layer_heights"] = float64([0.1, 0.2, 0.5])
        data["layer_densities"] = float64([250, 250, 917])
        data["layer_temperatures"] = float64([260, 270, 272])
        data["layer_liquid_water_content"] = float64([0.0, 0.0, 0.0])

        assert isinstance(data, dict)
        for array in data.values():
            assert isinstance(array, np.ndarray)
            assert len(array) == 3

        grid = cpgrid.Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )

        test_nodes = grid.number_nodes
        self.add_debris_to_grid(grid_obj=grid)
        assert grid.get_node_ntype(0) == 1
        for i in range(1, grid.number_nodes):
            assert grid.get_node_ntype(i) == 0
        assert grid.number_nodes == test_nodes + 1
        assert grid.get_number_debris_layers() == 1

    def test_boilerplate_integration(self):
        """Integration test for boilerplate methods."""

        self.test_add_debris_to_grid()


@pytest.fixture(
    name="conftest_debris_boilerplate", scope="function", autouse=False
)
def conftest_boilerplate():
    """Yields class containing methods for common tests."""

    test_boilerplate = TestDebrisBoilerplate()
    test_boilerplate.test_boilerplate_integration()

    yield TestDebrisBoilerplate()
