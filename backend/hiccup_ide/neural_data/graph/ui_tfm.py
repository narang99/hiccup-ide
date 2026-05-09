"""Graph transformer for converting raw computation graphs to UI-optimized graphs.

This module handles the transformation of raw neural network dependency graphs
into forms optimized for UI visualization, specifically consolidating receptive
field relationships into single patch nodes.
"""

import itertools
from typing import cast, assert_never
from enum import Enum
import networkx as nx
from neural_data.types import (
    Coordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    Conv2dOutputCoordinate,
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
)
from neural_data.ui_graph_types import (
    UIGraphNode,
    Conv2dInputPatchNode,
    TuplifiedInputCoordinates,
)


class TransformAction(Enum):
    """Actions to take for different coordinate types during transformation."""

    PASS_THROUGH = "pass_through"  # Use node as-is
    DEFER_TO_SLICE = "defer_to_slice"  # Will be handled when slice is processed
    REMOVE = "remove"  # Remove from UI graph entirely


CacheType = dict[Coordinate, list[UIGraphNode]]


def raw_to_ui_graph(g: nx.DiGraph) -> nx.DiGraph:
    roots = [n for n, d in g.in_degree() if d == 0]
    cache: CacheType = {}
    tfm_graph = nx.DiGraph()
    for root in roots:
        # already added to graph
        _tfm_root(root, g, cache, tfm_graph)
    return tfm_graph


def _tfm_root(
    root: Coordinate,
    raw_graph: nx.DiGraph,
    cache: dict[Coordinate, list[UIGraphNode]],
    tfm_graph: nx.DiGraph,
) -> list[UIGraphNode]:
    # there is no cycle detection for now btw
    if root in cache:
        return cache[root]

    match root:
        case Conv2dSliceCoordinate() as r:
            patch_node, children = _get_patch_and_children(
                r, raw_graph, cache, tfm_graph
            )
            tfm_graph.add_node(patch_node)
            cache[r] = [patch_node]
            for c in children:
                tfm_graph.add_edge(patch_node, c)
            return [patch_node]
        case ReLUInputCoordinate() as r:
            return _pass_through_dfs(r, raw_graph, cache, tfm_graph)
        case ModelInputCoordinate() as r:
            return _pass_through_dfs(r, raw_graph, cache, tfm_graph)
        case Conv2dOutputCoordinate() as r:
            return _pass_through_dfs(r, raw_graph, cache, tfm_graph)
        case ReLUOutputCoordinate() as r:
            children = raw_graph.successors(r)
            children = itertools.chain.from_iterable(
                (_tfm_root(c, raw_graph, cache, tfm_graph) for c in children)
            )
            children = list(children)
            cache[r] = children
            return children

        case Conv2dInputCoordinate() as r:
            raise Exception(
                "found a Conv2dInputCoordinate while traversing raw graph to create the UI graph\n"
                "Currently it is assumed that Conv2dInputCoordinate would always be a child of Conv2dSliceCoordinate\n"
                "and this transform gobbles them when handling Conv2dSliceCoordinate\n"
                f"node={r}"
            )
        case _:
            assert_never(root)


def _pass_through_dfs(
    r: ReLUInputCoordinate | ModelInputCoordinate | Conv2dOutputCoordinate,
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
) -> list[UIGraphNode]:
    cache[r] = [r]
    tfm_graph.add_node(r)
    for c in raw_graph.successors(r):
        ui_children = _tfm_root(c, raw_graph, cache, tfm_graph)
        for ui_child in ui_children:
            tfm_graph.add_edge(r, ui_child)
    return [r]


def _get_patch_and_children(
    root: Conv2dSliceCoordinate, raw_graph: nx.DiGraph, cache, tfm_graph
) -> tuple[Conv2dInputPatchNode, list[UIGraphNode]]:
    nodes = raw_graph.successors(root)
    _assert_nodes_of_type(nodes, Conv2dInputCoordinate)
    coords = cast(list[Conv2dInputCoordinate], nodes)
    ip_coord_by_ui_children = _get_ip_coords_by_ui_children(
        coords, raw_graph, cache, tfm_graph
    )
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
    children = itertools.chain.from_iterable(ip_coord_by_ui_children.values())
    return patch_node, list(children)


def _get_ip_coords_by_ui_children(
    input_coords: list[Conv2dInputCoordinate],
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
) -> dict[Conv2dInputCoordinate, list[UIGraphNode]]:
    ip_coord_by_children = {c: raw_graph.successors(c) for c in input_coords}
    ip_coord_by_ui_children = {}
    for ip_coord, its_children in ip_coord_by_children.items():
        ui_children = (_tfm_root(c, raw_graph, cache, tfm_graph) for c in its_children)
        ui_children = itertools.chain.from_iterable(ui_children)
        ip_coord_by_ui_children[ip_coord] = ui_children
    return ip_coord_by_ui_children


def _tuplify_input_coords(
    ip_coord_by_ui_children: dict[Conv2dInputCoordinate, list[UIGraphNode]],
) -> TuplifiedInputCoordinates:
    list_of_tuples = [(k, tuple(v)) for (k, v) in ip_coord_by_ui_children.items()]
    return tuple(list_of_tuples)


def _get_patch_boundaries(input_coords: list[Conv2dInputCoordinate]):
    min_y = min(coord.y for coord in input_coords)
    max_y = max(coord.y for coord in input_coords)
    min_x = min(coord.x for coord in input_coords)
    max_x = max(coord.x for coord in input_coords)
    return (min_y, min_x, max_y, max_x)


def _assert_nodes_of_type(nodes: list[Coordinate], typ, message=""):
    for n in nodes:
        if not isinstance(n, typ):
            raise Exception(
                f"{message} expected node to be of type: {typ} got type={type(n)} node={n}"
            )