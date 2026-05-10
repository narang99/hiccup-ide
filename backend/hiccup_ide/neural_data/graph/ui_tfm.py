"""Graph transformer for converting raw computation graphs to UI-optimized graphs.

This module handles the transformation of raw neural network dependency graphs
into forms optimized for UI visualization, specifically consolidating receptive
field relationships into single patch nodes.
"""

import itertools
from typing import cast, Protocol, runtime_checkable, Union, Literal, Annotated
import networkx as nx
from pydantic import Field, BaseModel
from neural_data.types import (
    Coordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
    Conv2dInputPatchNode,
    TuplifiedInputCoordinates,
)

# --- Pydantic Result Models ---

class Skip(BaseModel):
    status: Literal["skip"] = "skip"

class Consumed(BaseModel):
    status: Literal["consumed"] = "consumed"
    nodes: list[Coordinate]

RecurseStrategyResult = Annotated[
    Union[Skip, Consumed],
    Field(discriminator="status")
]

CacheType = dict[Coordinate, list[Coordinate]]

@runtime_checkable
class RecurseStrategy(Protocol):
    """Protocol defining the contract for graph transformation strategies.

    A strategy is responsible for evaluating a node (and potentially its subgraph)
    and deciding how to represent it in the transformed UI graph.
    """
    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: "RecurseStrategy",
    ) -> RecurseStrategyResult:
        ...

# --- Concrete Strategies ---

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
        if not successors or not all(isinstance(s, Conv2dInputCoordinate) for s in successors):
            return Skip()

        # Handle caching
        if root in cache:
            return Consumed(nodes=cache[root])

        patch_node, children = _get_patch_and_children(
            root, cast(list[Conv2dInputCoordinate], successors), raw_graph, cache, tfm_graph, main_strategy
        )
        tfm_graph.add_node(patch_node)

        for c in children:
            tfm_graph.add_edge(patch_node, c)

        cache[root] = [patch_node]
        return Consumed(nodes=[patch_node])

class OmitNodeStrategy:
    """Transparently removes specific coordinates from the UI graph.

    Handles coordinates like ReLUOutputCoordinate or ModelInputCoordinate 
    that are redundant for UI visualization. It returns the results of its 
    children directly, effectively 'short-circuiting' itself out of the graph.
    """
    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        if isinstance(root, ModelInputCoordinate):
            cache[root] = []
            return Consumed(nodes=[])

        if isinstance(root, ReLUOutputCoordinate):
            if root in cache:
                return Consumed(nodes=cache[root])

            children_coords = list(raw_graph.successors(root))
            ui_children = _must_consume(
                children_coords, raw_graph, cache, tfm_graph, main_strategy
            )

            cache[root] = ui_children
            return Consumed(nodes=ui_children)

        return Skip()

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

class PassThroughStrategy:
    """Default strategy that preserves raw nodes in the UI graph.

    Acts as the terminal catch-all in the strategy chain. It adds the 
    current node to the UI graph and recursively links it to the UI nodes 
    returned by its computational children.
    """
    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        # Catch-all for ReLUInputCoordinate, Conv2dOutputCoordinate, etc.
        if root in cache:
            return Consumed(nodes=cache[root])

        tfm_graph.add_node(root)
        cache[root] = [root]

        children_coords = list(raw_graph.successors(root))
        ui_children = _must_consume(
            children_coords, raw_graph, cache, tfm_graph, main_strategy
        )
        for ui_child in ui_children:
            tfm_graph.add_edge(root, ui_child)

        return Consumed(nodes=[root])

class FirstMatchingOrFallback:
    """Orchestrator for the Chain of Responsibility transformation.

    Iterates through a list of RecurseStrategy objects until one returns 
    a Consumed result. It enforces the contract that every node must 
    eventually be consumed or a fallback error is raised.
    """
    def __init__(self, strategies: list[RecurseStrategy]):        self.strategies = strategies

    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        if root in cache:
            return Consumed(nodes=cache[root])

        for strategy in self.strategies:
            result = strategy(root, raw_graph, cache, tfm_graph, main_strategy)
            if isinstance(result, Consumed):
                return result
        
        raise Exception(f"No strategy handled node: {root}")

# --- Helpers ---

def _get_patch_and_children(
    root: Conv2dSliceCoordinate, 
    coords: list[Conv2dInputCoordinate],
    raw_graph: nx.DiGraph, 
    cache: CacheType, 
    tfm_graph: nx.DiGraph,
    main_strategy: RecurseStrategy
) -> tuple[Conv2dInputPatchNode, list[Coordinate]]:
    ip_coord_by_ui_children = {}
    for ip_coord in coords:
        its_children = list(raw_graph.successors(ip_coord))
        ui_children = _must_consume(
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
    
    all_ui_children = list(itertools.chain.from_iterable(ip_coord_by_ui_children.values()))
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

def _must_consume(
    coords: list[Coordinate],
    raw_graph: nx.DiGraph,
    cache: CacheType,
    tfm_graph: nx.DiGraph,
    main_strategy: RecurseStrategy,
) -> list[Coordinate]:
    ui_nodes = []
    for c in coords:
        res = main_strategy(c, raw_graph, cache, tfm_graph, main_strategy)
        if isinstance(res, Skip):
            raise RuntimeError(f"Main strategy failed to consume node during recursion: {c}")
        ui_nodes.extend(res.nodes)
    return ui_nodes

# --- Entry Point ---

def raw_to_ui_graph(g: nx.DiGraph) -> nx.DiGraph:
    roots = [n for n, d in g.in_degree() if d == 0]
    cache: CacheType = {}
    tfm_graph = nx.DiGraph()
    
    main_strategy = FirstMatchingOrFallback([
        Conv2dPatchStrategy(),
        OmitNodeStrategy(),
        ErrorStrategy(),
        PassThroughStrategy(),
    ])
    
    for root in roots:
        main_strategy(root, g, cache, tfm_graph, main_strategy)
        
    return tfm_graph
