import numba

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid
from cosipy.modules.refreezing import refreezing


class TestParamRefreezing:
    """Tests methods for parametrising refreezing.

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
    heights = numba.float64([0.1, 0.2, 0.3, 0.5, 0.5])
    densities = numba.float64([250, 250, 250, 917, 917])
    temperatures = numba.float64([260, 270, 271, 271.5, 272])
    liquid_water_contents = numba.float64([0.01, 0.01, 0.01, 0.01, 0.01])

    def test_refreezing(self, conftest_boilerplate):
        test_grid = Grid(
            layer_heights=self.heights,
            layer_densities=self.densities,
            layer_temperatures=self.temperatures,
            layer_liquid_water_content=self.liquid_water_contents,
        )
        test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)
        test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)
        compare_refreeze = refreezing(test_grid)
        conftest_boilerplate.check_output(
            test_grid.get_node_liquid_water_content(0), float, 0.0
        )
        conftest_boilerplate.check_output(
            test_grid.get_node_liquid_water_content(1), float, 0.0
        )
