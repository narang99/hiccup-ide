import itertools
from typing import cast, Union
import networkx as nx
from neural_data.types import (
    Coordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    ReLUOutputCoordinate,
    ReLUOutputPatchNode,
    ModelInputCoordinate,
    ModelInputPatchNode,
    TuplifiedInputCoordinates,
)
from .core import Skip, Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .common import must_consume


class Slice2PatchStrategy:
    """Consolidates ReLU outputs or Model inputs into a single UI Patch node.

    This strategy looks for a Conv2dSliceCoordinate followed by
    Conv2dInputCoordinates, each of which must have exactly one
    ReLUOutputCoordinate or ModelInputCoordinate successor.

    Pattern:
    - conv2d slice coord -> [conv2d input coords] -> [relu out coords]
    - conv2d slice coord -> [conv2d input coords] -> [model input coords]
    """

    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        if not isinstance(root, Conv2dSliceCoordinate):
            return Skip()

        successors = list(raw_graph.successors(root))
        if not successors or not all(
            isinstance(s, Conv2dInputCoordinate) for s in successors
        ):
            return Skip()

        # Check if every Conv2dInputCoordinate has exactly one successor
        # and that all these successors are of the same type (either ReLU or Model Input)
        leaf_coords: list[Union[ReLUOutputCoordinate, ModelInputCoordinate]] = []
        target_type = None

        for s in successors:
            grand_successors = list(raw_graph.successors(s))
            if len(grand_successors) != 1:
                return Skip()
            
            gs = grand_successors[0]
            if not isinstance(gs, (ReLUOutputCoordinate, ModelInputCoordinate)):
                return Skip()
            
            if target_type is None:
                target_type = type(gs)
            elif not isinstance(gs, target_type):
                # Mixed types are not supported by this strategy
                return Skip()
                
            leaf_coords.append(gs)

        # Handle caching
        if root in cache:
            return Consumed(nodes=cache[root])

        patch_node, children = _get_patch_and_children(
            root,
            leaf_coords,
            raw_graph,
            cache,
            tfm_graph,
            main_strategy,
        )
        tfm_graph.add_node(patch_node)

        for c in children:
            tfm_graph.add_edge(patch_node, c)

        cache[root] = [patch_node]
        return Consumed(nodes=[patch_node])


def _get_patch_and_children(
    root: Conv2dSliceCoordinate,
    leaf_coords: list[Union[ReLUOutputCoordinate, ModelInputCoordinate]],
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
    main_strategy: RecurseStrategy,
) -> tuple[Union[ReLUOutputPatchNode, ModelInputPatchNode], list[Coordinate]]:
    leaf_coord_by_ui_children = {}
    for leaf_coord in leaf_coords:
        its_children = list(raw_graph.successors(leaf_coord))
        ui_children = must_consume(
            its_children, raw_graph, cache, tfm_graph, main_strategy
        )
        leaf_coord_by_ui_children[leaf_coord] = ui_children

    min_y, min_x, max_y, max_x = _get_patch_boundaries(leaf_coords)
    
    # We assume all leaf_coords belong to the same layer and channel 
    representative = leaf_coords[0]

    if isinstance(representative, ReLUOutputCoordinate):
        patch_node = ReLUOutputPatchNode(
            type="ReLUOutputPatchNode",
            layer_name=representative.layer_name,
            layer_type="relu",
            coordinate_type="output_patch",
            channel=representative.channel,
            patch_min_y=min_y,
            patch_min_x=min_x,
            patch_max_y=max_y,
            patch_max_x=max_x,
            input_coordinates=_tuplify_input_coords(leaf_coord_by_ui_children),
        )
    else:
        patch_node = ModelInputPatchNode(
            type="ModelInputPatchNode",
            layer_name=representative.layer_name,
            layer_type="input",
            coordinate_type="input_patch",
            channel=representative.channel,
            patch_min_y=min_y,
            patch_min_x=min_x,
            patch_max_y=max_y,
            patch_max_x=max_x,
            input_coordinates=_tuplify_input_coords(leaf_coord_by_ui_children),
        )

    all_ui_children = list(
        itertools.chain.from_iterable(leaf_coord_by_ui_children.values())
    )
    return patch_node, all_ui_children


def _tuplify_input_coords(
    ip_coord_by_ui_children: dict[Union[ReLUOutputCoordinate, ModelInputCoordinate], list[Coordinate]],
) -> TuplifiedInputCoordinates:
    list_of_tuples = [(k, tuple(v)) for (k, v) in ip_coord_by_ui_children.items()]
    return tuple(list_of_tuples)


def _get_patch_boundaries(input_coords: list[Union[ReLUOutputCoordinate, ModelInputCoordinate]]):
    min_y = min(coord.y for coord in input_coords)
    max_y = max(coord.y for coord in input_coords)
    min_x = min(coord.x for coord in input_coords)
    max_x = max(coord.x for coord in input_coords)
    return (min_y, min_x, max_y, max_x)
