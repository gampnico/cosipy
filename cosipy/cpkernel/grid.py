import os
from collections import OrderedDict

import numpy as np
from numba import float64, int64, intp, njit, optional, typed, types
from numba.experimental import jitclass
from numba.extending import register_jitable

import constants
from config import use_debris
from cosipy.cpkernel.node import *  # contains node.__all__

__all__ = ["Grid"]
_NODE_TYPE = _init_node_type()  # else _init_grid can't access the correct type


@register_jitable
def _init_grid_type(node_type) -> types.ListType:
    """Initialises typed List used by Grid.grid.

    Args:
        node_type: A registered StructRefProxy type.
    Returns:
        The base List type for Grid objects.
    """

    grid_type = types.ListType(node_type)
    return grid_type


@register_jitable
def _init_grid_jit_types() -> OrderedDict:
    """Initialise numba types for JIT-compiled Grid objects.

    Returns:
        Numba types for Grid objects, including node type.
    """

    spec = OrderedDict()  # Using `spec.update` is slower than adding keys.
    spec["layer_heights"] = float64[:]
    spec["layer_densities"] = float64[:]
    spec["layer_temperatures"] = float64[:]
    spec["layer_liquid_water_content"] = float64[:]
    spec["layer_ice_fraction"] = optional(float64[:])
    spec["number_nodes"] = intp
    spec["new_snow_height"] = float64
    spec["new_snow_timestamp"] = float64
    spec["old_snow_timestamp"] = float64

    grid_type = _init_grid_type(_NODE_TYPE)
    spec["grid"] = grid_type

    return spec


spec = _init_grid_jit_types()


@jitclass(spec)
class Grid:
    """The Grid class controls the numerical mesh.

    The grid attribute consists of a list of nodes that each store
    information on an individual layer. The class provides various
    setter/getter functions to add, read, overwrite, merge, split,
    update or re-mesh the layers.

    Attributes
    ----------
    layer_heights : np.ndarray
        Height of the snowpack layers [:math:`m`].
    layer_densities : np.ndarray
        Snow density of the snowpack layers [:math:`kg~m^{-3}`].
    layer_temperatures : np.ndarray
        Layer temperatures [:math:`K`].
    layer_liquid_water_content : np.ndarray
        Liquid water content of the layers [:math:`m~w.e.`].
    layer_ice_fraction : np.ndarray
        Volumetric ice fraction  of the layers [-]. Default None.
    new_snow_height : float
        Height of the fresh snow layer [:math:`m`]. Default None.
    new_snow_timestamp : float
        Time elapsed since the last snowfall [s]. Default None.
    old_snow_timestamp : float
        Time elapsed between the last and penultimate snowfalls [s].
            Default None.
    grid : typed.List
        Numerical mesh for glacier data.
    number_nodes : int
        Number of layers in the numerical mesh.
    """

    def __init__(
        self,
        layer_heights: float64[:],
        layer_densities: float64[:],
        layer_temperatures: float64[:],
        layer_liquid_water_content: float64[:],
        layer_ice_fraction: optional(float64[:]) = None,
        new_snow_height: float64 = None,
        new_snow_timestamp: float64 = None,
        old_snow_timestamp: float64 = None,
    ):
        # Set class variables
        self.layer_heights = layer_heights
        self.layer_densities = layer_densities
        self.layer_temperatures = layer_temperatures
        self.layer_liquid_water_content = layer_liquid_water_content
        self.layer_ice_fraction = layer_ice_fraction

        # Number of total nodes
        self.number_nodes = len(layer_heights)

        # Track the fresh snow layer (new_snow_height, new_snow_timestamp) as well as the old
        # snow layer age (old_snow_timestamp)
        if (
            (new_snow_height is not None)
            and (new_snow_timestamp is not None)
            and (old_snow_timestamp is not None)
        ):
            self.new_snow_height = new_snow_height  # meter snow accumulation
            self.new_snow_timestamp = (
                new_snow_timestamp  # seconds since snowfall
            )
            self.old_snow_timestamp = (
                old_snow_timestamp  # snow age below fresh snow layer
            )
        else:
            # TO DO: pick better initialization values
            self.new_snow_height = 0.0
            self.new_snow_timestamp = 0.0
            self.old_snow_timestamp = 0.0

        # Initialise the grid
        self.init_grid()

    def init_grid(self):
        """Initialize the grid with the input data."""
        _init_grid(self)

    def add_fresh_snow(
        self, height, density, temperature, liquid_water_content
    ):
        """Add a fresh snow layer (node).

        Adds a fresh snow layer to the beginning of the node list (upper
        layer).

        Parameters
        ----------
        height : float
            Layer height [:math:`m`].
        density : float
            Layer density [:math:`kg~m^{-3}`].
        temperature : float
            Layer temperature [:math:`K`].
        liquid_water_content : float
            Liquid water content of the layer [:math:`m~w.e.`].
        """

        # Add new node
        self.grid.insert(
            0, Node(height, density, temperature, liquid_water_content, None)
        )

        # Increase node counter
        self.number_nodes += 1

        # Set fresh snow properties for albedo calculation (height and
        # timestamp)
        self.set_fresh_snow_props(height)

    def add_fresh_debris(
        self, height, density, temperature, liquid_water_content
    ):
        """Add a debris layer (DebrisNode).

        Adds a debris layer to the beginning of the node list (upper
        layer). This buries any snow.

        Parameters
        ----------
        height : float
            Layer height [:math:`m`].
        density : float
            Layer density [:math:`kg~m^{-3}`].
        temperature : float
            Layer temperature [:math:`K`].
        liquid_water_content : float
            Liquid water content of the layer [:math:`m~w.e.`].
        """

        self.grid.insert(
            0,
            DebrisNode(height, density, temperature, 0.0, 0.0),
        )
        self.number_nodes += 1

        self.set_fresh_snow_props(0.0)  # snow is buried under debris

    def remove_node(self, idx: list = None):
        """Remove a layer (node) from the grid (node list).

        Parameters
        ----------
        idx: list
            Indices of the node to be removed. If empty or None, the
            first node is removed. Default None.
        """

        # Remove node from list when there is at least one node
        if self.grid:
            if idx is None or not idx:
                self.grid.pop(0)
                self.number_nodes -= 1  # Decrease node counter
            else:
                for index in sorted(idx, reverse=True):
                    del self.grid[index]
                self.number_nodes -= len(idx)

    def merge_nodes(self, idx):
        """Merge two subsequent nodes.

        Merges the two nodes at location `idx` and `idx+1`.
        The node at `idx` is updated with the new properties (height,
        liquid water content, ice fraction, temperature). The node
        at `idx+1` is deleted after merging.

        Parameters
        ----------
        idx : int
            Index of the node to be removed. The first node is removed
            if no index is provided.
        """
        # Get overburden pressure for consistency check
        # w0 = self.get_node_height(idx) * self.get_node_density(
        #     idx
        # ) + self.get_node_height(idx + 1) * self.get_node_density(idx + 1)

        # New layer height by adding up the height of the two layers
        new_height = self.get_node_height(idx) + self.get_node_height(idx + 1)

        # Update liquid water
        # new_liquid_water_content = self.get_node_liquid_water_content(idx) + self.get_node_liquid_water_content(idx+1)
        new_liquid_water_content = (
            self.get_node_liquid_water_content(idx) * self.get_node_height(idx)
            + self.get_node_liquid_water_content(idx + 1)
            * self.get_node_height(idx + 1)
        ) / new_height

        # Update ice fraction
        new_ice_fraction = (
            self.get_node_ice_fraction(idx) * self.get_node_height(idx)
            + self.get_node_ice_fraction(idx + 1)
            * self.get_node_height(idx + 1)
        ) / new_height

        # New air porosity
        new_air_porosity = 1 - new_liquid_water_content - new_ice_fraction

        if (
            abs(
                1
                - new_ice_fraction
                - new_air_porosity
                - new_liquid_water_content
            )
            > 1e-8
        ):
            print(
                "Merging is not mass consistent",
                (
                    new_ice_fraction
                    + new_air_porosity
                    + new_liquid_water_content
                ),
            )

        # Calc new temperature
        new_temperature = (
            self.get_node_height(idx) / new_height
        ) * self.get_node_temperature(idx) + (
            self.get_node_height(idx + 1) / new_height
        ) * self.get_node_temperature(
            idx + 1
        )

        # Update node properties
        self.update_node(
            idx,
            new_height,
            new_temperature,
            new_ice_fraction,
            new_liquid_water_content,
        )

        # Remove the second layer
        self.remove_node([idx + 1])

    def correct_layer_selector(self, idx, min_height):
        """Restrict layer correction to matching layer types.

        Used by the debris implementation. Prevents debris merging with
        snow/ice.

        Parameters
        ----------
        idx : int
            Index of the node to be removed. The first node is removed
            if no index is provided.
        min_height : float
            New layer height [:math:`m`].
        """

        if _check_node_ntype(
            self, idx=idx + 1, ntype=self.get_node_ntype(idx)
        ):
            self.correct_layer(idx, min_height)

    def correct_layer(self, idx, min_height):
        """Adjust the height of a given layer.

        Adjusts the height of the layer at index `idx` to the given
        height `min_height`. First the layers below are merged until the
        height is sufficiently large to allow for the adjustment. Then
        the layer is merged with the subsequent layer.

        Parameters
        ----------
        idx : int
            Index of the node to be removed. The first node is removed
            if no index is provided.
        min_height : float
            New layer height [:math:`m`].

        """
        # New layer height by adding up the height of the two layers
        total_height = self.get_node_height(idx)

        # Merge subsequent layer with underlying layers until height of the layer is greater than the given height
        while (total_height < min_height) & (
            idx + 1 < self.get_number_layers()
        ):
            if (self.get_node_density(idx) < constants.snow_ice_threshold) & (
                self.get_node_density(idx + 1) < constants.snow_ice_threshold
            ):
                self.merge_nodes(idx)
            elif (self.get_node_density(idx) >= constants.snow_ice_threshold) & (
                self.get_node_density(idx + 1) >= constants.snow_ice_threshold
            ):
                self.merge_nodes(idx)
            else:
                break

            # Recalculate total height
            total_height = self.get_node_height(idx)

        # Only merge snow-snow or glacier-glacier, and if the height is greater than the minimum height
        if total_height > min_height:
            # Get new heights for layer 0 and 1
            h0 = min_height
            h1 = total_height - min_height

            """Update liquid water content.
            Fills the upper layer with water until maximum retention.
            The remaining water is assigned to the second layer.
            If LWC exceeds the irreducible water content of the first
            layer, then the first layer is filled and the rest assigned
            to the second layer.
            """
            if (
                self.get_node_liquid_water_content(idx)
                - self.get_node_irreducible_water_content(idx)
            ) > 0:
                lw0 = self.get_node_irreducible_water_content(
                    idx
                ) * self.get_node_height(idx)
                lw1 = self.get_node_liquid_water_content(
                    idx
                ) * self.get_node_height(
                    idx
                ) - self.get_node_irreducible_water_content(
                    idx
                ) * self.get_node_height(
                    idx
                )
            # if LWC<WC_irr, then assign all water to the first layer
            else:
                lw0 = self.get_node_liquid_water_content(
                    idx
                ) * self.get_node_height(idx)
                lw1 = 0.0

            # Update ice fraction
            if0 = self.get_node_ice_fraction(idx)
            if1 = self.get_node_ice_fraction(idx)

            # Temperature
            T0 = self.get_node_temperature(idx)
            T1 = self.get_node_temperature(idx)

            # New volume fractions and density
            lwc0 = lw0 / h0
            lwc1 = lw1 / h1
            por0 = 1 - lwc0 - if0
            por1 = 1 - lwc1 - if1

            # Check for consistency
            if (abs(1 - if0 - por0 - lwc0) > 1e-8) | (
                abs(1 - if1 - por1 - lwc1) > 1e-8
            ):
                print(
                    "Correct layer is not mass consistent [Layer 0]",
                    (if0, por0, lwc0),
                )
                print(
                    "Correct layer is not mass consistent [Layer 1]",
                    (if0, por0, lwc0),
                )

            # Update node properties
            self.update_node(idx, h0, T0, if0, lwc0)
            self.grid.insert(
                idx + 1, Node(h1, self.get_node_density(idx), T1, lwc1, if1)
            )

            # Update node counter
            self.number_nodes += 1

    def set_next_height(
        self, remesh_depth: float, current_height: float
    ) -> tuple:
        """Set next layer's height and the amount of snow/ice to remesh.

        Args:
            remesh_depth: The amount of snow or ice left to remesh [m].
            current_height: The current layer's height [m].

        Returns:
            tuple[float, float]: Updated remesh depth and the next
            layer's height.
        """
        remesh_depth = remesh_depth - current_height
        # Height for the next layer
        next_height = constants.layer_stretching * current_height

        return remesh_depth, next_height

    def log_profile(self):
        """Remesh the layer heights logarithmically.

        This algorithm remeshes the layer heights (numerical
        grid) logarithmically using a given stretching factor and first
        layer height. Both are defined in `constants.py`:

        * The stretching factor is defined by `layer_stretching`.
        * The first layer height is defined by `first_layer_height`.

        E.g. for the stretching factor, a value of 1.1 corresponds to a
        10% stretching from one layer to the next.
        """
        last_layer_height = constants.first_layer_height

        # Total snowheight
        hsnow = self.get_total_snowheight()

        # How much snow is not remeshed
        hrest = hsnow

        # First remesh the snowpack
        idx = 0

        while idx < self.get_number_snow_layers():
            if hrest >= last_layer_height:
                # Correct first layer
                self.correct_layer(idx, last_layer_height)
                hrest, last_layer_height = self.set_next_height(
                    hrest, last_layer_height
                )

            # if the last layer is smaller than the required height,
            # then merge with the previous layer
            elif (hrest < last_layer_height) & (idx > 0):
                self.merge_nodes(idx - 1)

            idx = idx + 1

        # get the glacier depth
        hrest = self.get_total_height() - self.get_total_snowheight()

        # get number of snow layers
        idx = self.get_number_snow_layers()

        # then the glacier
        while idx < self.get_number_layers():
            if hrest >= last_layer_height:
                # Correct first layer
                self.correct_layer(idx, last_layer_height)
                hrest, last_layer_height = self.set_next_height(
                    hrest, last_layer_height
                )

            elif hrest < last_layer_height:
                self.merge_nodes(idx - 1)

            idx = idx + 1

    def log_profile_debris(self):
        """Remesh the layer heights logarithmically (debris-compatible).

        Used by the debris implementation of COSIPY. It remeshes the
        snow layer heights (numerical grid) logarithmically using a
        given stretching factor and first layer height. Both are defined
        in `constants.py`:

        * The stretching factor is defined by `layer_stretching`.
        * The first layer height is defined by `first_layer_height`.

        E.g. for the stretching factor, a value of 1.1 corresponds to a
        10% stretching from one layer to the next.

        This algorithm does not remesh debris layer heights.

        .. note:: A snow/ice layer directly below a debris layer cannot
            be merged even if it is smaller than `last_layer_height`.
        """

        last_layer_height = constants.first_layer_height
        hsnow = self.get_total_snowheight()
        # How much snow is not yet remeshed
        hrest = hsnow

        # First remesh the snowpack
        idx = 0
        n_debris = 0
        while idx < self.get_number_snow_layers() + n_debris:
            if _check_node_is_snow(self, idx):
                if hrest >= last_layer_height:
                    # Correct first layer
                    self.correct_layer_selector(idx, last_layer_height)
                    hrest, last_layer_height = self.set_next_height(
                        hrest, last_layer_height
                    )
                # if the last layer is smaller than the required height,
                # then merge with the previous layer
                elif (
                    (hrest < last_layer_height)
                    & (idx > 0)
                    & (_check_node_is_snow(self, idx - 1))
                ):
                    self.merge_nodes(idx - 1)
            else:
                n_debris = n_debris + 1
            idx = idx + 1

        # get the glacier depth
        # TODO: only subtract heights of skipped debris layers
        hrest = (
            self.get_total_height()
            - self.get_total_snowheight()
            - self.get_total_debris_height()
        )

        idx = self.get_number_snow_layers() + n_debris
        n_debris = 0  # already processed debris layers in snowpack
        while idx < self.get_number_layers():
            if not _check_node_ntype(self, idx, 1):
                if (hrest >= last_layer_height) & (
                    idx + 1 < self.get_number_layers()
                ):
                    self.correct_layer_selector(idx, last_layer_height)
                    hrest, last_layer_height = self.set_next_height(
                        hrest, last_layer_height
                    )
                elif (hrest < last_layer_height) & (
                    not _check_node_ntype(self, idx - 1, 1)
                ):
                    self.merge_nodes(idx - 1)
            else:
                n_debris = n_debris + 1
            idx = idx + 1

    def adaptive_profile(self):
        """Remesh according to certain layer state criteria.

        This algorithm is an alternative to logarithmic remeshing.
        It checks the similarity of two subsequent layers. Layers are
        merged if:

        (1) the density difference between the layer and the subsequent
        layer is smaller than the user defined threshold.
        (2) the temperature difference is smaller than the user defined
        threshold.
        (3) the number of merges per time step does not exceed the user
        defined threshold.

        The thresholds are defined by `temperature_threshold_merging`,
        `density_threshold_merging`, and `merge_max` in `constants.py`.
        """
        # First remesh the snowpack
        idx = 0
        merge_counter = 0
        while idx < self.get_number_snow_layers() - 1:
            dT = np.abs(
                self.get_node_temperature(idx)
                - self.get_node_temperature(idx + 1)
            )
            dRho = np.abs(
                self.get_node_density(idx) - self.get_node_density(idx + 1)
            )

            if (
                (dT <= constants.temperature_threshold_merging)
                & (dRho <= constants.density_threshold_merging)
                & (self.get_node_height(idx) <= 0.1)
                & (merge_counter <= constants.merge_max)
            ):
                self.merge_nodes(idx)
                merge_counter = merge_counter + 1
            # elif ((self.get_node_height(idx)<=minimum_snow_layer_height) & (dRho<=density_threshold_merging)):
            elif self.get_node_height(idx) <= constants.minimum_snow_layer_height:
                self.remove_node([idx])
            else:
                idx += 1

        # Correct first layer
        self.correct_layer(0, constants.first_layer_height)

    def adaptive_profile_debris(self):
        """Remesh according to certain layer state criteria.

        This algorithm is an alternative to logarithmic remeshing.
        It checks the similarity of two subsequent layers. Layers are
        merged if:

        (1) the density difference between the layer and the subsequent
        layer is smaller than the user defined threshold.
        (2) the temperature difference is smaller than the user defined
        threshold.
        (3) the number of merges per time step does not exceed the user
        defined threshold.

        The thresholds are defined by `temperature_threshold_merging`,
        `density_threshold_merging`, and `merge_max` in `constants.py`.
        """

        idx = 0
        merge_counter = 0
        n_debris = 0

        # First remesh the snowpack
        while idx < self.get_number_snow_layers() + n_debris - 1:
            if _check_node_is_snow(self, idx):
                dT = np.abs(
                    self.get_node_temperature(idx)
                    - self.get_node_temperature(idx + 1)
                )
                dRho = np.abs(
                    self.get_node_density(idx) - self.get_node_density(idx + 1)
                )
                if (
                    _check_node_is_snow(self, idx + 1)
                    & (dT <= constants.temperature_threshold_merging)
                    & (dRho <= constants.density_threshold_merging)
                    & (self.get_node_height(idx) <= 0.1)
                    & (merge_counter <= constants.merge_max)
                ):
                    self.merge_nodes(idx)
                    merge_counter += 1
                # elif ((self.get_node_height(idx)<=minimum_snow_layer_height) & (dRho<=density_threshold_merging)):
                elif (
                    self.get_node_height(idx)
                    <= constants.minimum_snow_layer_height
                ):
                    self.remove_node([idx])
                else:
                    idx += 1  # only step when all other conditions exhausted
            else:
                n_debris += 1
                idx += 1  # skip current debris layer

        # Correct first layer
        if not _check_node_ntype(self, 0, 1):
            self.correct_layer(0, constants.first_layer_height)

    def split_node(self, pos):
        """Split node at position.

        Splits a node at a location index `pos` into two similar nodes.
        The new nodes at location `pos` and `pos+1` will have the same
        properties (height, liquid water content, ice fraction,
        temperature).

        Parameters
        ----------
        pos : int
            Index of the node to split.
        """
        self.grid.insert(
            pos + 1,
            Node(
                self.get_node_height(pos) / 2.0,
                self.get_node_density(pos),
                self.get_node_temperature(pos),
                self.get_node_liquid_water_content(pos) / 2.0,
                self.get_node_ice_fraction(pos),
            ),
        )
        self.update_node(
            pos,
            self.get_node_height(pos) / 2.0,
            self.get_node_temperature(pos),
            self.get_node_ice_fraction(pos),
            self.get_node_liquid_water_content(pos) / 2.0,
        )

        self.number_nodes += 1

    def update_node(
        self, idx, height, temperature, ice_fraction, liquid_water_content
    ):
        """Update properties of a specific node.

        This function updates a layer's attributes for `height`,
        `temperature`, `ice_fraction`, and `liquid_water_content`.
        The density cannot be updated as it is derived from air
        porosity, liquid water content, and ice fraction.

        Parameters
        ----------
        idx : int
            Index of the layer to be updated.
        height : float
            Layer's new snowpack height [:math:`m`].
        temperature : float
            Layer's new temperature [:math:`K`].
        ice_fraction : float
            Layer's new ice fraction [:math:`-`].
        liquid_water_content : float
            Layer's new liquid water content [:math:`m~w.e.`].
        """
        self.set_node_height(idx, height)
        self.set_node_temperature(idx, temperature)
        self.set_node_ice_fraction(idx, ice_fraction)
        self.set_node_liquid_water_content(idx, liquid_water_content)

    def check(self, name):
        """Check layer temperature and height are within a valid range."""
        if np.min(self.get_height()) < 0.01:
            print(name)
            print(
                "Layer height is smaller than the user defined minimum new_height"
            )
            print(self.get_height())
            print(self.get_density())
        if np.max(self.get_temperature()) > 273.2:
            print(name)
            print("Layer temperature exceeds 273.16 K")
            print(self.get_temperature())
            print(self.get_density())
        if np.max(self.get_height()) > 1.0:
            print(name)
            print("Layer height exceeds 1.0 m")
            print(self.get_height())
            print(self.get_density())

    def update_grid_debris(self):
        """Remesh layers (numerical grid) for debris-covered glaciers.

        Two algorithms are currently implemented to remesh layers:

            (i)  log_profile_debris
            (ii) adaptive_profile_debris

        (i)  The log-profile algorithm arranges the mesh
             logarithmically. The user specifies the stretching factor
             `layer_stretching` in `constants.py` to determine the
             increase in layer heights. Debris layers are skipped and
             englacial debris will prevent adjacent snow/ice nodes from
             merging.

        (ii) Profile adjustment uses layer similarity. Layers with very
             similar temperature and density states are joined.
             Similarity is determined from the user-specified threshold
             values `temperature_threshold_merging` and
             `density_threshold_merging` in `constants.py`. The maximum
             number of merging steps per time step is specified by
             `merge_max`.
        """
        if constants.remesh_method == "log_profile":
            self.log_profile_debris()
        elif constants.remesh_method == "adaptive_profile":
            self.adaptive_profile_debris()
        else:
            error_msg = f"{constants.remesh_method} is not implemented."
            raise NotImplementedError(error_msg)

        # first layer is too small and not debris
        if (not _check_node_ntype(self, 0, 1)) & (
            self.get_node_height(0) < constants.minimum_snow_layer_height
        ):
            self.remove_node([0])

    def update_grid(self):
        """Remesh the layers (numerical grid).

        Two algorithms are currently implemented to remesh layers:

            (i)  log_profile
            (ii) adaptive_profile

        (i)  The log-profile algorithm arranges the mesh
             logarithmically. The user specifies the stretching factor
             `layer_stretching` in `constants.py` to determine the
             increase in layer heights.

        (ii) Profile adjustment is done using layer similarity. Layers
             with very similar temperature and density states are
             joined. Similarity is determined from the user-specified
             threshold values `temperature_threshold_merging` and
             `density_threshold_merging` in `constants.py`. The maximum
             number of merging steps per time step is specified by
             `merge_max`.
        """
        # -------------------------------------------------------------------------
        # Remeshing options
        # -------------------------------------------------------------------------
        if not use_debris:
            if constants.remesh_method == "log_profile":
                self.log_profile()
            elif constants.remesh_method == "adaptive_profile":
                self.adaptive_profile()
            else:
                error_msg = f"{constants.remesh_method} is not implemented."
                raise NotImplementedError(error_msg)

            # if first layer becomes very small, remove it
            if self.get_node_height(0) < constants.minimum_snow_layer_height:
                self.remove_node([0])
        else:
            self.update_grid_debris()

    def merge_snow_with_glacier(self, idx):
        """Merge a snow layer with an ice layer.

        Merges a snow layer at location `idx` (density smaller than the
        `snow_ice_threshold` value in `constants.py`) with an ice layer
        at location `idx+1`.

        Parameters
        ----------
        idx : int
            Index of the snow layer.
        """
        if (self.get_node_density(idx) < constants.snow_ice_threshold) & (
            self.get_node_density(idx + 1) >= constants.snow_ice_threshold
        ):
            # Update node properties
            first_layer_height = self.get_node_height(idx) * (
                self.get_node_density(idx) / constants.ice_density
            )
            self.update_node(
                idx + 1,
                self.get_node_height(idx + 1) + first_layer_height,
                self.get_node_temperature(idx + 1),
                self.get_node_ice_fraction(idx + 1),
                0.0,
            )

            # Remove the second layer
            self.remove_node([idx])

            # self.check('Merge snow with glacier function')

    def remove_melt_weq(self, melt, idx=0):
        """Remove mass from a layer.

        Reduces the mass/height of layer `idx` by the available melt
        energy.

        Parameters
        ----------
        melt : float
            Snow water equivalent of melt [:math:`m~w.e.`].
        idx : int
            Index of the layer. If no value is given, the function acts
            on the first layer.
        """
        lwc_from_layers = 0

        while melt > 0:
            # Get SWE of layer
            SWE = self.get_node_height(idx) * (
                self.get_node_density(idx) / constants.water_density
            )
            # Remove melt from layer and set new snowheight
            if melt < SWE:
                self.set_node_height(
                    idx,
                    (SWE - melt)
                    / (self.get_node_density(idx) / constants.water_density),
                )
                melt = 0.0
            # remove layer otherwise and continue loop
            elif melt >= SWE:
                lwc_from_layers = (
                    lwc_from_layers
                    + self.get_node_liquid_water_content(idx)
                    * self.get_node_height(idx)
                )
                self.remove_node([idx])
                melt = melt - SWE

        # Keep track of the fresh snow layer
        if idx == 0:
            self.set_fresh_snow_props_height(self.new_snow_height - melt)

        return lwc_from_layers

    # ===============================================================================
    # Getter and setter functions
    # ===============================================================================

    def set_fresh_snow_props(self, height):
        """Track the new snowheight.

        Parameters
        ----------
        height : float
            Height of the fresh snow layer [:math:`m`].
        """
        self.new_snow_height = height
        # Keep track of the old snow age
        self.old_snow_timestamp = self.new_snow_timestamp
        # Set the timestamp to zero
        self.new_snow_timestamp = 0

    def set_fresh_snow_props_to_old_props(self):
        """Revert the timestamp of fresh snow properties.

        Reverts the timestamp of fresh snow properties to that of the
        underlying snow layer. This is used internally to track the
        albedo properties of the first snow layer.
        """
        self.new_snow_timestamp = self.old_snow_timestamp

    def set_fresh_snow_props_update_time(self, seconds):
        """Update the timestamp of the snow properties.

        Parameters
        ----------
        seconds : float
            seconds without snowfall [:math:`s`].
        """
        self.old_snow_timestamp = self.old_snow_timestamp + seconds
        # Set the timestamp to zero
        self.new_snow_timestamp = self.new_snow_timestamp + seconds

    def set_fresh_snow_props_height(self, height):
        """Update the fresh snow layer height.

        This is used internally to track the albedo properties of the
        first snow layer.
        """
        self.new_snow_height = height

    def get_fresh_snow_props(self):
        """Get the first snow layer's properties.

        This is used internally to track the albedo properties of the
        first snow layer.
        """
        return (
            self.new_snow_height,
            self.new_snow_timestamp,
            self.old_snow_timestamp,
        )

    def set_node_temperature(self, idx, temperature):
        """Set the temperature of a layer (node) at location `idx`.

        Parameters
        ----------
        idx : int
            Index of the layer.
        temperature : float
            Layer's new temperature [:math:`K`].
        """
        self.grid[idx].set_layer_temperature(temperature)

    def set_temperature(self, temperature):
        """Set all layer temperatures.

        Parameters
        ----------
        temperature : np.ndarray
            New layer temperatures [:math:`K`].
        """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_temperature(temperature[idx])

    def set_node_height(self, idx: int, height: float):
        """Set node height."""
        self.grid[idx].set_layer_height(height)

    def set_height(self, height):
        """Set the height profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_height(height[idx])

    def set_node_liquid_water_content(
        self, idx: int, liquid_water_content: float
    ):
        """Set liquid water content of a node."""
        self.grid[idx].set_layer_liquid_water_content(liquid_water_content)

    def set_liquid_water_content(self, liquid_water_content):
        """Set the liquid water content profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water_content(
                liquid_water_content[idx]
            )

    def set_node_ice_fraction(self, idx: int, ice_fraction: float):
        """Set liquid ice_fraction of a node."""
        self.grid[idx].set_layer_ice_fraction(ice_fraction)

    def set_ice_fraction(self, ice_fraction):
        """Set the ice fraction profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_ice_fraction(ice_fraction[idx])

    def set_node_refreeze(self, idx: int, refreeze: float):
        """Set the refreezing of a node."""
        self.grid[idx].set_layer_refreeze(refreeze)

    def set_refreeze(self, refreeze):
        """Set the refreezing profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_refreeze(refreeze[idx])

    def get_temperature(self):
        """Get the temperature profile."""
        return [
            self.grid[idx].get_layer_temperature()
            for idx in range(self.number_nodes)
        ]

    def get_node_temperature(self, idx: int):
        """Get a node's temperature."""
        return self.grid[idx].get_layer_temperature()

    def get_specific_heat(self):
        """Get the specific heat capacity profile (air+water+ice)."""
        return [
            self.grid[idx].get_layer_specific_heat()
            for idx in range(self.number_nodes)
        ]

    def get_node_specific_heat(self, idx: int):
        """Get a node's specific heat capacity (air+water+ice)."""
        return self.grid[idx].get_layer_specific_heat()

    def get_height(self):
        """Get the heights of all the layers."""
        return [
            self.grid[idx].get_layer_height()
            for idx in range(self.number_nodes)
        ]

    def get_snow_heights(self):
        """Get the heights of the snow layers."""
        if not use_debris:
            snow_heights = [
                self.grid[idx].get_layer_height()
                for idx in range(self.get_number_snow_layers())
            ]
        else:
            snow_heights = [
                self.grid[idx].get_layer_height()
                for idx in range(self.number_nodes)
                if (_check_node_is_snow(self, idx))
            ]
        return snow_heights

    def get_ice_heights(self):
        """Get the heights of the ice layers."""
        if not use_debris:
            ice_heights = [
                self.grid[idx].get_layer_height()
                for idx in range(self.number_nodes)
                if (self.get_node_density(idx) >= constants.snow_ice_threshold)
            ]
        else:
            ice_heights = [
                self.grid[idx].get_layer_height()
                for idx in range(self.number_nodes)
                if (_check_node_is_ice(self, idx))
            ]

        return ice_heights

    def get_node_ntype(self, idx: int) -> int:
        """Get a node's subclass type if available."""
        if not use_debris:
            return 0
        else:
            return self.grid[idx].get_layer_ntype()

    def get_ntype(self) -> list:
        """Get the layer node types."""
        return [
            self.grid[idx].get_layer_ntype()
            for idx in range(self.number_nodes)
        ]

    def get_debris_heights(self) -> list:
        """Get the heights of the debris layers."""
        return [
            self.grid[idx].get_layer_height()
            for idx in range(self.number_nodes)
            if (_check_node_ntype(self, idx, 1))
        ]

    def get_node_height(self, idx: int):
        """Get a node's layer height."""
        return self.grid[idx].get_layer_height()

    def get_node_density(self, idx: int):
        """Get a node's density."""
        return self.grid[idx].get_layer_density()

    def get_density(self):
        """Get the density profile."""
        return [
            self.grid[idx].get_layer_density()
            for idx in range(self.number_nodes)
        ]

    def get_node_liquid_water_content(self, idx: int):
        """Get a node's liquid water content."""
        return self.grid[idx].get_layer_liquid_water_content()

    def get_liquid_water_content(self):
        """Get a profile of the liquid water content."""
        return [
            self.grid[idx].get_layer_liquid_water_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_ice_fraction(self, idx: int):
        """Get a node's ice fraction."""
        return self.grid[idx].get_layer_ice_fraction()

    def get_ice_fraction(self):
        """Get a profile of the ice fraction."""
        return [
            self.grid[idx].get_layer_ice_fraction()
            for idx in range(self.number_nodes)
        ]

    def get_node_irreducible_water_content(self, idx: int):
        """Get a node's irreducible water content."""
        return self.grid[idx].get_layer_irreducible_water_content()

    def get_irreducible_water_content(self):
        """Get a profile of the irreducible water content."""
        return [
            self.grid[idx].get_layer_irreducible_water_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_cold_content(self, idx: int):
        """Get a node's cold content."""
        return self.grid[idx].get_layer_cold_content()

    def get_cold_content(self):
        """Get the cold content profile."""
        return [
            self.grid[idx].get_layer_cold_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_porosity(self, idx: int):
        """Get a node's porosity."""
        return self.grid[idx].get_layer_porosity()

    def get_porosity(self):
        """Get the porosity profile."""
        return [
            self.grid[idx].get_layer_porosity()
            for idx in range(self.number_nodes)
        ]

    def get_node_thermal_conductivity(self, idx: int):
        """Get a node's thermal conductivity."""
        return self.grid[idx].get_layer_thermal_conductivity()

    def get_thermal_conductivity(self):
        """Get the thermal conductivity profile."""
        return [
            self.grid[idx].get_layer_thermal_conductivity()
            for idx in range(self.number_nodes)
        ]

    def get_node_thermal_diffusivity(self, idx: int):
        """Get a node's thermal diffusivity."""
        return self.grid[idx].get_layer_thermal_diffusivity()

    def get_thermal_diffusivity(self):
        """Get the thermal diffusivity profile"""
        return [
            self.grid[idx].get_layer_thermal_diffusivity()
            for idx in range(self.number_nodes)
        ]

    def get_node_refreeze(self, idx: int):
        """Get the amount of refrozen water in a node."""
        return self.grid[idx].get_layer_refreeze()

    def get_refreeze(self):
        """Get the profile of refrozen water."""
        return [
            self.grid[idx].get_layer_refreeze()
            for idx in range(self.number_nodes)
        ]

    def get_node_depth(self, idx: int):
        d = 0
        for i in range(idx + 1):
            if i == 0:
                d = d + self.get_node_height(i) / 2.0
            else:
                d = (
                    d
                    + self.get_node_height(i - 1) / 2.0
                    + self.get_node_height(i) / 2.0
                )
        return d

    def get_depth(self):
        """Returns depth profile."""
        return [self.get_node_depth(idx) for idx in range(self.number_nodes)]

    def get_total_snowheight(self, verbose=False):
        """Get the total snowheight (density<snow_ice_threshold)."""
        return sum(self.get_snow_heights())

    def get_total_debris_height(self):
        """Get the sum of all debris layer heights."""
        return sum(self.get_debris_heights())

    def get_total_height(self, verbose=False):
        """Get the total domain height."""
        return sum(self.get_height())

    def get_number_snow_layers(self):
        """Get the number of snow layers (density<snow_ice_threshold)."""
        if not use_debris:
            nlayers = [
                1
                for idx in range(self.number_nodes)
                if self.get_node_density(idx) < constants.snow_ice_threshold
            ]

        else:
            nlayers = [
                1
                for idx in range(self.number_nodes)
                if (_check_node_is_snow(self, idx))
            ]

        return sum(nlayers)

    def get_number_debris_layers(self) -> int64:
        """Get the number of debris layers (ntype == 1)."""
        return _get_number_ntype_layers(self, 1)

    def get_number_layers(self):
        """Get the number of layers."""
        return self.number_nodes

    def info(self):
        """Print some information on grid."""

        print("******************************")
        print("Number of nodes:", self.number_nodes)
        print("******************************")

        tmp = 0
        for i in range(self.number_nodes):
            tmp = tmp + self.get_node_height(i)

        print("Grid consists of", self.number_nodes, "nodes")
        print("Total domain depth is", tmp, "m")

    def grid_info(self, n=-999):
        """Print the state of the snowpack.

        Parameters
        ----------
        n : int
            Number of nodes to plot from top.
        """
        if n == -999:
            n = self.number_nodes

        print(
            "Node no., Layer height [m], Temperature [K], Density [kg m^-3]",
            "LWC [-], LW [m], CC [J m^-2], Porosity [-], Refreezing [m w.e.]",
            "Irreducible water content [-]",
        )

        for i in range(n):
            print(
                i,
                np.round(self.get_node_height(i), decimals=3),
                np.round(self.get_node_temperature(i), decimals=3),
                np.round(self.get_node_density(i), decimals=3),
                np.round(self.get_node_liquid_water_content(i), decimals=3),
                np.round(self.get_node_cold_content(i), decimals=3),
                np.round(self.get_node_porosity(i), decimals=3),
                np.round(self.get_node_refreeze(i), decimals=3),
                np.round(
                    self.get_node_irreducible_water_content(i), decimals=3
                ),
            )

    def grid_info_screen(self, n=-999):
        """Prints the state of the snowpack.

        Parameters
        ----------
        n : int
            Number of nodes to plot from top.
        """
        if n == -999:
            n = self.number_nodes

        print(
            "Node no., Layer height [m], Temperature [K], Density [kg m^-3], LWC [-], \
               Retention [-], CC [J m^-2], Porosity [-], Refreezing [m w.e.]"
        )

        for i in range(n):
            print(
                i,
                self.get_node_height(i),
                self.get_node_temperature(i),
                self.get_node_density(i),
                self.get_node_liquid_water_content(i),
                self.get_node_irreducible_water_content(i),
                self.get_node_cold_content(i),
                self.get_node_porosity(i),
                self.get_node_refreeze(i),
            )

    def grid_check(self, level=1):
        """Checks the grid.

        Parameters
        ----------
            level : int
                Level number.
        """
        # if level == 1:
        #    self.check_layer_property(self.get_height(), 'thickness', 1.01, -0.001)
        #    self.check_layer_property(self.get_temperature(), 'temperature', 273.2, 100.0)
        #    self.check_layer_property(self.get_density(), 'density', 918, 100)
        #    self.check_layer_property(self.get_liquid_water_content(), 'LWC', 1.0, 0.0)
        #    #self.check_layer_property(self.get_cold_content(), 'CC', 1000, -10**8)
        #    self.check_layer_property(self.get_porosity(), 'Porosity', 0.8, -0.00001)
        #    self.check_layer_property(self.get_refreeze(), 'Refreezing', 0.5, 0.0)
        pass

    def check_layer_property(
        self, property, name, maximum, minimum, n=-999, level=1
    ):
        if np.nanmax(property) > maximum or np.nanmin(property) < minimum:
            print(
                str.capitalize(name),
                "max",
                np.nanmax(property),
                "min",
                np.nanmin(property),
            )
            os._exit()


@njit(cache=False)
def _get_number_ntype_layers(self, ntype: int64 = 0) -> int:
    """Get the number of layers matching a specific subtype.

    Used by the debris implementation.

    Args:
        ntype: Subtype number. Default 0.

    Returns:
        nlayers: Number of layers with the specified subtype number.
    """
    nlayers = 0
    for idx in range(self.number_nodes):
        if _check_node_ntype(self, idx, ntype):
            nlayers += 1
    return nlayers


@njit(cache=False)
def _check_node_ntype(self, idx: int, ntype: int64 = 0) -> bool:
    """Check a layer has a specific subtype number.

    Used by the debris implementation.

    Args:
        idx: Node index in `grid`.
        ntype: Subtype number. Default 0.

    Returns:
        True if layer subtype matches, otherwise returns False.
    """
    return self.get_node_ntype(idx) == ntype


@njit(cache=False)
def _check_node_is_snow(self, idx: int) -> bool:
    """Check if a layer is a snow layer.

    Used by the debris implementation.

    Args:
        idx: Node index in `grid`.

    Returns:
        True if layer is snow, otherwise returns False.
    """
    return (self.get_node_ntype(idx) == 0) & (
        self.get_node_density(idx) < constants.snow_ice_threshold
    )


@njit(cache=False)
def _check_node_is_ice(self, idx: int) -> bool:
    """Check if a layer is an ice layer.

    Used by the debris implementation.

    Args:
        idx: Node index in `grid`.

    Returns:
        True if layer is snow, otherwise returns False.
    """
    return (self.get_node_ntype(idx) == 0) & (
        self.get_node_density(idx) >= constants.snow_ice_threshold
    )


@njit(cache=False)
def _create_node_at_idx(grid_obj: Grid, idx: int) -> Node:
    """Create a node instance with user data at a specific grid index.

    This function automatically hooks the appropriate constructor for
    the node type selected in config.

    Parameters
    ----------
    grid_obj : Grid
        A new Grid instance with an empty `grid` attribute.
    idx : int
        Grid index of node.

    Returns
    -------
    node : Node
        Node containing the data passed by user arguments to Grid.
    """

    if grid_obj.layer_ice_fraction is not None:
        ice_fraction = grid_obj.layer_ice_fraction[idx]
    else:
        ice_fraction = None

    # let cpkernel.node handle node type overrides
    node = _create_node(
        grid_obj.layer_heights[idx],
        grid_obj.layer_densities[idx],
        grid_obj.layer_temperatures[idx],
        grid_obj.layer_liquid_water_content[idx],
        ice_fraction,
    )
    return node


@register_jitable
def _init_grid(grid_obj: Grid):
    """Bind user data to the `grid` attribute.

    Parameters
    ----------
    grid_obj : Grid
        A new Grid instance with an empty `grid` attribute.
    """

    grid_obj.grid = typed.List.empty_list(_NODE_TYPE)
    # Fill the list with node instances containing user defined data
    for idx_node in range(grid_obj.number_nodes):
        fill_node = _create_node_at_idx(grid_obj, idx_node)
        grid_obj.grid.append(fill_node)
