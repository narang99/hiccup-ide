import networkx as nx
from neural_data.types import Coordinate
from .core import Consumed, RecurseStrategy, RecurseStrategyResult, CacheType
from .common import must_consume


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
        ui_children = must_consume(
            children_coords, raw_graph, cache, tfm_graph, main_strategy
        )
        for ui_child in ui_children:
            tfm_graph.add_edge(root, ui_child)

        return Consumed(nodes=[root])
