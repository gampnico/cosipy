import numba
import numpy as np

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid
from cosipy.modules.densification import densification


class TestParamDensification:
    """Tests methods for parametrising densification.

    Attributes:
        heights (np.ndarray[float]): Snowpack heights for each layer
            [:math:`m`].
        densities (np.ndarray[float]): Snow densities for each layer
            [:math:`kg~m^{-3}`].
        temperatures (np.ndarray[float]): Temperatures for each layer
            [:math:`K`].
        liquid_water_contents (np.ndarray[float]): Liquid water content
            for each layer [:math:`m~w.e.`].
    """

    # values are different to fixture
    heights = numba.float64([10.0, 10.0, 10.0, 0.2, 0.2])
    densities = numba.float64([450, 450, 450, 100, 100])
    temperatures = numba.float64([260, 270, 271, 271.5, 272])
    liquid_water_contents = numba.float64([0, 0, 0, 0, 0])

    def test_densification(self, conftest_debris_boilerplate):
        test_grid = Grid(  # values are different to fixture
            layer_heights=self.heights,
            layer_densities=self.densities,
            layer_temperatures=self.temperatures,
            layer_liquid_water_content=self.liquid_water_contents,
        )
        conftest_debris_boilerplate.add_debris_to_grid(
            grid_obj=test_grid, debris=5, snow=2
        )
        test_snow_layers = test_grid.get_number_snow_layers()
        test_layers = test_grid.number_nodes
        test_debris_layers = test_grid.get_number_debris_layers()

        SWE_before = np.array(test_grid.get_height()) / np.array(
            test_grid.get_density()
        )
        SWE_before_sum = np.nansum(SWE_before)

        densification(GRID=test_grid, SLOPE=0.0, dt=3600)

        SWE_after = np.array(test_grid.get_height()) / np.array(
            test_grid.get_density()
        )
        SWE_after_sum = np.nansum(SWE_after)
        assert np.isclose(SWE_before_sum, SWE_after_sum, atol=1e-3)

        assert test_grid.get_number_snow_layers() == test_snow_layers
        assert test_grid.number_nodes == test_layers
        assert test_grid.get_number_debris_layers() == test_debris_layers

        for idx in range(test_grid.number_nodes):
            if test_grid.get_node_ntype(idx) == 1:
                assert test_grid.get_node_ice_fraction(idx) == 0.0
            else:
                assert test_grid.get_node_ice_fraction(idx) > 0.0
