"""Graph transformer for converting raw computation graphs to UI-optimized graphs.

This module handles the transformation of raw neural network dependency graphs
into forms optimized for UI visualization, specifically consolidating receptive
field relationships into single patch nodes.
"""

import networkx as nx
from neural_data.types import (
    Coordinate,
    ReLUInputCoordinate,
)
from .core import Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .pass_through import PassThroughStrategy
from .omit_type import OmitNodeStrategy
from .slice2patch import Slice2PatchStrategy


class FirstMatchingOrFallback:
    """Orchestrator for the Chain of Responsibility transformation.

    Iterates through a list of RecurseStrategy objects until one returns
    a Consumed result. It enforces the contract that every node must
    eventually be consumed or a fallback error is raised.
    """

    def __init__(self, strategies: list[RecurseStrategy]):
        self.strategies = strategies

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


def raw_to_ui_graph(g: nx.DiGraph) -> nx.DiGraph:
    roots = [n for n, d in g.in_degree() if d == 0]
    cache: CacheType = {}
    tfm_graph = nx.DiGraph()

    main_strategy = FirstMatchingOrFallback(
        [
            Slice2PatchStrategy(),
            # Conv2dPatchStrategy(),
            OmitNodeStrategy(ReLUInputCoordinate),
            # OmitNodeStrategy(ModelInputCoordinate),
            # ErrorStrategy(),
            PassThroughStrategy(),
        ]
    )

    for root in roots:
        main_strategy(root, g, cache, tfm_graph, main_strategy)

    return tfm_graph
