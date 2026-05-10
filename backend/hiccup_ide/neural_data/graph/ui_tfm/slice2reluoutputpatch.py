import itertools
from typing import cast
import networkx as nx
from neural_data.types import (
    Coordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    ReLUOutputCoordinate,
    ReLUOutputPatchNode,
    TuplifiedInputCoordinates,
)
from .core import Skip, Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .common import must_consume


class Slice2ReLUOutputPatchStrategy:
    """Consolidates ReLU outputs into a single UI Patch node.

    This strategy looks for a Conv2dSliceCoordinate followed by
    Conv2dInputCoordinates, each of which must have exactly one
    ReLUOutputCoordinate successor.

    Pattern:
    - conv2d slice coord -> [conv2d input coords] -> [relu out coords]
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

        # Check if every Conv2dInputCoordinate has exactly one ReLUOutputCoordinate successor
        relu_outputs: list[ReLUOutputCoordinate] = []
        for s in successors:
            grand_successors = list(raw_graph.successors(s))
            if len(grand_successors) != 1 or not isinstance(
                grand_successors[0], ReLUOutputCoordinate
            ):
                return Skip()
            relu_outputs.append(cast(ReLUOutputCoordinate, grand_successors[0]))

        # Handle caching
        if root in cache:
            return Consumed(nodes=cache[root])

        patch_node, children = _get_patch_and_children(
            root,
            relu_outputs,
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
    relu_coords: list[ReLUOutputCoordinate],
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
    main_strategy: RecurseStrategy,
) -> tuple[ReLUOutputPatchNode, list[Coordinate]]:
    relu_coord_by_ui_children = {}
    for relu_coord in relu_coords:
        its_children = list(raw_graph.successors(relu_coord))
        ui_children = must_consume(
            its_children, raw_graph, cache, tfm_graph, main_strategy
        )
        relu_coord_by_ui_children[relu_coord] = ui_children

    min_y, min_x, max_y, max_x = _get_patch_boundaries(relu_coords)
    
    # We assume all relu_coords belong to the same layer and channel 
    # since they are all successors of Conv2dInputCoordinates from the same Conv2dSliceCoordinate
    # which points to a specific in_channel.
    representative = relu_coords[0]

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
        input_coordinates=_tuplify_input_coords(relu_coord_by_ui_children),
    )

    all_ui_children = list(
        itertools.chain.from_iterable(relu_coord_by_ui_children.values())
    )
    return patch_node, all_ui_children


def _tuplify_input_coords(
    ip_coord_by_ui_children: dict[ReLUOutputCoordinate, list[Coordinate]],
) -> TuplifiedInputCoordinates:
    # We reuse TuplifiedInputCoordinates even though it says Conv2dInputCoordinate in types.py
    # Actually, let's check TuplifiedInputCoordinates definition.
    # It is: tuple[tuple[Conv2dInputCoordinate, tuple["Coordinate", ...]], ...]
    # I should probably update that type or use a more generic one if it's strictly typed.
    list_of_tuples = [(k, tuple(v)) for (k, v) in ip_coord_by_ui_children.items()]
    return tuple(list_of_tuples)


def _get_patch_boundaries(input_coords: list[ReLUOutputCoordinate]):
    min_y = min(coord.y for coord in input_coords)
    max_y = max(coord.y for coord in input_coords)
    min_x = min(coord.x for coord in input_coords)
    max_x = max(coord.x for coord in input_coords)
    return (min_y, min_x, max_y, max_x)
