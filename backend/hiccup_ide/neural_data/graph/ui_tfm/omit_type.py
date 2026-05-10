import networkx as nx
from neural_data.types import (
    Coordinate,
)
from .core import Skip, Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .common import must_consume


class OmitNodeStrategy:
    """Transparently removes specific coordinates from the UI graph.

    Handles coordinates like ReLUOutputCoordinate or ModelInputCoordinate
    that are redundant for UI visualization. It returns the results of its
    children directly, effectively 'short-circuiting' itself out of the graph.
    """

    def __init__(self, typ: type[Coordinate]):
        self.typ = typ

    def __call__(
        self,
        root: Coordinate,
        raw_graph: nx.DiGraph,
        cache: CacheType,
        tfm_graph: nx.DiGraph,
        main_strategy: RecurseStrategy,
    ) -> RecurseStrategyResult:
        if isinstance(root, self.typ):
            if root in cache:
                return Consumed(nodes=cache[root])

            children_coords = list(raw_graph.successors(root))
            ui_children = must_consume(
                children_coords, raw_graph, cache, tfm_graph, main_strategy
            )

            cache[root] = ui_children
            return Consumed(nodes=ui_children)

        return Skip()
