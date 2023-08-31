from collections import OrderedDict

import numpy as np
from numba import float64
from numba.experimental import jitclass

import constants

spec = OrderedDict()
spec["height"] = float64
spec["temperature"] = float64
spec["liquid_water_content"] = float64
spec["ice_fraction"] = float64
spec["refreeze"] = float64
spec["grain_size"] = float64

@jitclass(spec)
class Node:
    """A `Node` class stores a layer's state variables.
    
    The numerical grid consists of a list of nodes which store the
    information of individual layers. The class provides various
    setter/getter functions to read or overwrite the state of these
    individual layers. 

    Attributes
    ----------
    height : float
        Layer height [:math:`m`].
    snow_density : float
        Layer snow density [:math:`kg~m^{-3}`].
    temperature: float
        Layer temperature [:math:`K`].
    liquid_water_content : float
        Liquid water content [:math:`m~w.e.`].
    ice_fraction : float
        Volumetric ice fraction [-].
    refreeze : float
        Amount of refrozen liquid water [:math:`m~w.e.`].

    Returns
    -------
    Node : :py:class:`cosipy.cpkernel.node` object.
    """

    def __init__(
        self,
        height: float,
        snow_density: float,
        temperature: float,
        liquid_water_content: float,
        ice_fraction: float = None,
    ):
        # Initialises state variables.
        self.height = height
        self.temperature = temperature
        self.liquid_water_content = liquid_water_content

        if ice_fraction is None:
            # Remove weight of air from density
            a = snow_density - (1 - (snow_density / constants.ice_density)) * constants.air_density
            self.ice_fraction = a / constants.ice_density
        else:
            self.ice_fraction = ice_fraction

        self.refreeze = 0.0


    """GETTER FUNCTIONS"""

    # -----------------------------------------
    # Getter-functions for state variables
    # -----------------------------------------
    def get_layer_height(self) -> float:
        """Gets the node's layer height.
        
        Returns
        -------
        height : float
            Snow layer height [:math:`m`].
        """
        return self.height

    def get_layer_temperature(self) -> float:
        """Gets the node's snow layer temperature.
        
        Returns
        -------
        T : float
            Snow layer temperature [:math:`K`].
        """
        return self.temperature

    def get_layer_ice_fraction(self) -> float:
        """Gets the node's volumetric ice fraction.
        
        Returns
        -------
        ice_fraction : float
            The volumetric ice fraction [-].
        """
        return self.ice_fraction

    def get_layer_refreeze(self) -> float:
        """Gets the amount of refrozen water in the node.
        
        Returns
        -------
        refreeze : float
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return self.refreeze

    # ---------------------------------------------
    # Getter-functions for derived state variables
    # ---------------------------------------------
    def get_layer_density(self) -> float:
        """Gets the node's mean density including ice and liquid.

        Returns
        -------
        rho : float
            Snow density [:math:`kg~m^{-3}`].
        """
        return (
            self.get_layer_ice_fraction() * constants.ice_density
            + self.get_layer_liquid_water_content() * constants.water_density
            + self.get_layer_air_porosity() * constants.air_density
        )

    def get_layer_air_porosity(self) -> float:
        """Gets the fraction of air in the node.

        Returns
        -------
        porosity : float
            Air porosity [:math:`m`].
        """
        return max(0.0, 1 - self.get_layer_liquid_water_content() - self.get_layer_ice_fraction())

    def get_layer_specific_heat(self) -> float:
        """Gets the node's volumetric averaged specific heat capacity.

        Returns
        -------
        cp : float
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """
        return self.get_layer_ice_fraction()*constants.spec_heat_ice + self.get_layer_air_porosity()*constants.spec_heat_air + self.get_layer_liquid_water_content()*constants.spec_heat_water

    def get_layer_liquid_water_content(self) -> float:
        """Gets the node's liquid water content.

        Returns
        -------
        lwc : float
            Liquid water content [-].
        """
        return self.liquid_water_content

    def get_layer_irreducible_water_content(self) -> float:
        """Gets the node's irreducible water content.

        Returns
        -------
        ret : float
            Irreducible water content [-].
        """
        if self.get_layer_ice_fraction() <= 0.23:
            theta_e = 0.0264 + 0.0099*((1-self.get_layer_ice_fraction())/self.get_layer_ice_fraction())
        elif (self.get_layer_ice_fraction() > 0.23) & (self.get_layer_ice_fraction() <= 0.812):
            theta_e = 0.08 - 0.1023*(self.get_layer_ice_fraction()-0.03)
        else:
            theta_e = 0.0
        return theta_e

    def get_layer_cold_content(self) -> float:
        """Gets the node's cold content.

        Returns
        -------
        cc : float
            Cold content [:math:`J~m^{-2}`].
        """
        return -self.get_layer_specific_heat() * self.get_layer_density() * self.get_layer_height() * (self.get_layer_temperature()-constants.zero_temperature)

    def get_layer_porosity(self) -> float:
        """Gets the node's porosity.

        Returns
        -------
        porosity : float
            Air porosity [-].
        """
        return 1-self.get_layer_ice_fraction()-self.get_layer_liquid_water_content()

    def get_layer_thermal_conductivity(self) -> float:
        """Gets the node's volumetric weighted thermal conductivity.

        Returns
        -------
        lam : float
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """
        methods_allowed = ['bulk', 'empirical']
        if constants.thermal_conductivity_method == 'bulk':
            lam = self.get_layer_ice_fraction()*constants.k_i + self.get_layer_air_porosity()*constants.k_a + self.get_layer_liquid_water_content()*constants.k_w
        elif constants.thermal_conductivity_method == 'empirical':
            lam = 0.021 + 2.5 * np.power((self.get_layer_density()/1000),2)
        else:
            message = ("Thermal conductivity method =",
                       f"{constants.thermal_conductivity_method}",
                       f"is not allowed, must be one of",
                       f"{', '.join(methods_allowed)}")
            raise ValueError(" ".join(message))
        return lam

    def get_layer_thermal_diffusivity(self) -> float:
        """Gets the node's thermal diffusivity.

        Returns
        -------
        K : float
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """
        K = self.get_layer_thermal_conductivity()/(self.get_layer_density()*self.get_layer_specific_heat())
        return K

    """SETTER FUNCTIONS"""

    # ---------------------------------------------
    # Setter-functions for derived state variables
    # ---------------------------------------------
    def set_layer_height(self, height: float):
        """Sets the node's layer height.
        
        Parameters
        ----------
        height : float
            Layer height [:math:`m`].
        """
        self.height = height

    def set_layer_temperature(self, T: float):
        """Sets the node's mean temperature.

        Parameters
        ----------
        T : float
            Layer temperature [:math:`K`].
        """
        self.temperature = T

    def set_layer_liquid_water_content(self, lwc: float):
        """Sets the node's liquid water content.

        Parameters
        ----------
        lwc : float
            Liquid water content [-].
        """
        self.liquid_water_content = lwc

    def set_layer_ice_fraction(self, ifr: float):
        """Sets the node's volumetric ice fraction.

        Parameters
        ----------
        ifr : float
            Volumetric ice fraction [-].
        """
        self.ice_fraction = ifr

    def set_layer_refreeze(self, refr: float):
        """Sets the amount of refrozen water in the node.

        Parameters
        ----------
        refr : float
            Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr


# disable numba compilation until workaround found for pytest parametrization
# @jitclass(spec)
class DebrisNode:
    """Stores the state variables of a debris layer.

    The numerical grid consists of a list of nodes that store the
    information of individual debris layers. The class provides various
    setter/getter functions to read or overwrite the state of an
    individual debris layer.

    Attributes such as liquid water content/ice fraction are retained in
    case of future extensions for wet debris cover.

    Attributes
    ----------
    height : float
        Height of the debris layer [:math:`m`].
    density : float
        Debris density [:math:`kg~m^{-3}`].
    temperature : float
        Debris layer temperature [:math:`K`].
    liquid_water_content : float
        Liquid water content [:math:`m~w.e.`].
    """

    def __init__(
        self,
        height: float,
        debris_density: float,
        temperature: float,
        grain_size: float,
        liquid_water_content: float = 0.0,
        ice_fraction: float = None,
    ):
        # Initialize state variables
        self.height = height
        self.temperature = temperature
        self.grain_size = grain_size
        self.liquid_water_content = liquid_water_content

        if ice_fraction is None:
            # Remove weight of air from density
            # a = (
            #     debris_density
            #     - (1 - (debris_density / constants.debris_density))
            #     * constants.air_density
            # )
            # self.ice_fraction = a / constants.ice_density
            self.ice_fraction = 0.0
        else:
            self.ice_fraction = ice_fraction

        self.refreeze = 0.0

    """GETTER FUNCTIONS"""

    # ------------------------------------------
    # Getter-functions for state variables
    # ------------------------------------------
    def get_layer_height(self) -> float:
        """Get the node's layer height.

        Returns
        -------
        float
            Debris layer height [:math:`m`].
        """
        return self.height

    def get_layer_temperature(self) -> float:
        """Get the node's layer temperature.

        Returns
        -------
        float
            Debris layer temperature [:math:`K`].
        """
        return self.temperature

    def get_layer_grain_size(self) -> float:
        """Get the node's debris grain size.

        Returns
        -------
        float
            Debris grain size [:math:`mm`].
        """
        return self.grain_size

    def get_layer_ice_fraction(self) -> float:
        """Get the node's volumetric ice fraction.

        Returns
        -------
        float
            Volumetric ice fraction [-].
        """
        return self.ice_fraction

    def get_layer_refreeze(self) -> float:
        """Get the amount of refrozen water in the node.

        Returns
        -------
        float
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return self.refreeze

    # ----------------------------------------------
    # Getter-functions for derived state variables
    # ----------------------------------------------
    def get_layer_density(self) -> float:
        """Get the node's mean density including air and interstices.

        The density includes the clast material, the material filling
        the void between clasts, and the air in both clast and void
        filler pores.

        Returns
        -------
        float
            Debris density [:math:`kg~m^{-3}`].
        """

        if constants.debris_void_porosity >= 1.0:
            density = (
                self.get_layer_porosity() * constants.air_density
                + (1 - constants.debris_packing_porosity)
                * constants.debris_porosity
                * constants.debris_density  # clast density
                + (1 - self.get_layer_air_porosity())
                * constants.debris_void_density  # void filler density
            )
        else:
            density = (
                self.get_layer_porosity() * constants.air_density
                + (1 - self.get_layer_porosity()) * constants.debris_density
            )
        return density

    def get_layer_air_porosity(self) -> float:
        """Get the node's volumetrically-weighted interstitial void porosity.

        The function's name is kept as `get_layer_air_porosity` for
        cross-compatibility with other Node objects.

        Does NOT include the debris' porosity, and assumes no liquid.
        Note that the packing and void porosities are
        volumetrically-weighted!

        Returns
        -------
        float
            Interstitial void porosity [-].
        """

        if constants.debris_void_porosity >= 1.0:  # filled with air
            porosity = constants.debris_packing_porosity
        else:
            porosity = (
                constants.debris_packing_porosity * constants.debris_void_porosity
            )

        return max(0.0, porosity)

    def get_layer_specific_heat(self) -> float:
        """Get the node's volumetric averaged specific heat capacity.

        Returns
        -------
        float
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """

        if constants.debris_void_porosity >= 1.0:
            cp = self.get_layer_air_porosity() * constants.spec_heat_air + (
                1 - self.get_layer_air_porosity()
            ) * (constants.spec_heat_debris)

        else:
            cp = self.get_layer_air_porosity() * constants.spec_heat_air
            +(1 - self.get_layer_air_porosity()) * constants.spec_heat_debris

        return cp

    def get_layer_liquid_water_content(self) -> float:
        """Get the node's liquid water content.

        Returns
        -------
        float
            Liquid water content [-].
        """
        return self.liquid_water_content

    def get_layer_irreducible_water_content(self) -> float:
        """Get the node's irreducible water content.

        Returns
        -------
        float
            Irreducible water content [-].
        """
        if self.get_layer_ice_fraction() <= 0.23:
            theta_e = 0.0264 + 0.0099 * (
                (1 - self.get_layer_ice_fraction()) / self.get_layer_ice_fraction()
            )
        elif (self.get_layer_ice_fraction() > 0.23) & (
            self.get_layer_ice_fraction() <= 0.812
        ):
            theta_e = 0.08 - 0.1023 * (self.get_layer_ice_fraction() - 0.03)
        else:
            theta_e = 0.0
        return theta_e

    def get_layer_cold_content(self) -> float:
        """Get the node's cold content.

        Returns
        -------
        float
            Cold content [:math:`J~m^{-2}`].
        """
        return (
            -self.get_layer_specific_heat()
            * self.get_layer_density()
            * self.get_layer_height()
            * (self.get_layer_temperature() - constants.zero_temperature)
        )

    def get_layer_porosity(self) -> float:
        """Get the node's porosity.

        Volumetrically-weighted average of the debris porosity and its
        interstitial void porosity.

        Returns
        -------
        float
            Porosity [-].
        """

        porosity = (
            1 - constants.debris_packing_porosity
        ) * constants.debris_porosity + self.get_layer_air_porosity()

        return porosity

    def get_layer_thermal_conductivity(self) -> float:
        """Gets the node's thermal conductivity at ambient temperature.

        The debris' thermal conductivity at 273.15 K should be set in
        constants.py, as it varies between lithologies.

        Equation is from Vosteen & Schellschmidt (2003).

        Returns
        -------
        float
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """

        methods_allowed = ["sedimentary", "crystalline"]
        porosity = self.get_layer_air_porosity()

        if constants.debris_structure not in methods_allowed:
            message = (
                f'"{constants.debris_structure}" debris structure',
                f"is not allowed, must be one of",
                f'{", ".join(methods_allowed)}',
            )
            raise ValueError(message)
        elif constants.debris_structure == "sedimentary":
            a = 0.0034
            b = 0.0039
        elif constants.debris_structure == "crystalline":
            a = 0.0030
            b = 0.0042

        conductivity = (1 - porosity) * (
            constants.thermal_conductivity_debris * 0.99
            + self.get_layer_temperature()
            * (a - (b / constants.thermal_conductivity_debris))
        ) + (porosity * constants.k_a)

        return conductivity

    def get_layer_thermal_diffusivity(self) -> float:
        """Get the node's thermal diffusivity.

        Returns
        -------
        float
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """

        if constants.debris_structure == "crystalline":
            # Vosteen & Schellschmidt (2003)
            K = 0.45 * self.get_layer_thermal_conductivity()
        else:
            K = self.get_layer_thermal_conductivity() / (
                self.get_layer_density() * self.get_layer_specific_heat()
            )
        return K

    """SETTER FUNCTIONS"""

    # ----------------------------------------------
    # Setter-functions for derived state variables
    # ----------------------------------------------
    def set_layer_height(self, height) -> None:
        """Sets the node's layer height.

        Parameters
        ----------
        float
            Layer height [:math:`m`].
        """
        self.height = height

    def set_layer_temperature(self, T) -> None:
        """Sets the node's mean temperature.

        Parameters
        ----------
        float
            Layer temperature [:math:`K`].
        """
        self.temperature = T

    def set_layer_liquid_water_content(self, lwc) -> None:
        """Sets the node's liquid water content.

        Parameters
        ----------
        float
            Liquid water content [-].
        """
        self.liquid_water_content = lwc

    def set_layer_ice_fraction(self, ifr) -> None:
        """Sets the node's ice fraction.

        Parameters
        ----------
        float
            Ice fraction [-].
        """
        self.ice_fraction = ifr

    def set_layer_refreeze(self, refr) -> None:
        """Sets the amount of refrozen water in the node.

        Parameters
        ----------
        float
            Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr
