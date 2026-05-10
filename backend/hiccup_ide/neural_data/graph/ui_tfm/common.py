import networkx as nx
from neural_data.types import (
    Coordinate,
)
from .core import Skip, RecurseStrategy, CacheType


def must_consume(
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
            raise RuntimeError(
                f"Main strategy failed to consume node during recursion: {c}"
            )
        ui_nodes.extend(res.nodes)
    return ui_nodes
