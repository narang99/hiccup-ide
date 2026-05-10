import itertools
from typing import cast
import networkx as nx
from neural_data.types import (
    Coordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    Conv2dInputPatchNode,
    TuplifiedInputCoordinates,
)
from .core import Skip, Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .common import must_consume


class ErrorStrategy:
    """Guards against invalid or unexpected graph states.

    Raises an exception if it encounters nodes like Conv2dInputCoordinate
    in isolation, as these are strictly expected to be handled by
    Conv2dPatchStrategy's consolidation logic.
    """

    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        if isinstance(root, Conv2dInputCoordinate):
            raise Exception(
                "found a Conv2dInputCoordinate while traversing raw graph to create the UI graph\n"
                "Currently it is assumed that Conv2dInputCoordinate would always be a child of Conv2dSliceCoordinate\n"
                "and this transform gobbles them when handling Conv2dSliceCoordinate\n"
                f"node={root}"
            )
        return Skip()


class Conv2dPatchStrategy:
    """Consolidates convolution receptive fields into a single UI Patch node.

    This strategy looks for a Conv2dSliceCoordinate that is followed strictly
    by Conv2dInputCoordinates. It 'gobbles' these children and replaces the
    subgraph with a single Conv2dInputPatchNode.
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

        # Handle caching
        if root in cache:
            return Consumed(nodes=cache[root])

        patch_node, children = _get_patch_and_children(
            root,
            cast(list[Conv2dInputCoordinate], successors),
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
    coords: list[Conv2dInputCoordinate],
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
    main_strategy: RecurseStrategy,
) -> tuple[Conv2dInputPatchNode, list[Coordinate]]:
    ip_coord_by_ui_children = {}
    for ip_coord in coords:
        its_children = list(raw_graph.successors(ip_coord))
        ui_children = must_consume(
            its_children, raw_graph, cache, tfm_graph, main_strategy
        )
        ip_coord_by_ui_children[ip_coord] = ui_children

    min_y, min_x, max_y, max_x = _get_patch_boundaries(coords)
    patch_node = Conv2dInputPatchNode(
        type="Conv2dInputPatchNode",
        layer_name=root.layer_name,
        layer_type="conv2d",
        coordinate_type="input_patch",
        in_channel=root.in_channel,
        out_channel=root.out_channel,
        patch_min_y=min_y,
        patch_min_x=min_x,
        patch_max_y=max_y,
        patch_max_x=max_x,
        input_coordinates=_tuplify_input_coords(ip_coord_by_ui_children),
    )

    all_ui_children = list(
        itertools.chain.from_iterable(ip_coord_by_ui_children.values())
    )
    return patch_node, all_ui_children


def _tuplify_input_coords(
    ip_coord_by_ui_children: dict[Conv2dInputCoordinate, list[Coordinate]],
) -> TuplifiedInputCoordinates:
    list_of_tuples = [(k, tuple(v)) for (k, v) in ip_coord_by_ui_children.items()]
    return tuple(list_of_tuples)


def _get_patch_boundaries(input_coords: list[Conv2dInputCoordinate]):
    min_y = min(coord.y for coord in input_coords)
    max_y = max(coord.y for coord in input_coords)
    min_x = min(coord.x for coord in input_coords)
    max_x = max(coord.x for coord in input_coords)
    return (min_y, min_x, max_y, max_x)
