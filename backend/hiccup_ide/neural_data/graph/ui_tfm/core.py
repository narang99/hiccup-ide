from typing import Protocol, runtime_checkable, Union, Literal, Annotated
import networkx as nx
from pydantic import Field, BaseModel
from neural_data.types import (
    Coordinate,
)


class Skip(BaseModel):
    status: Literal["skip"] = "skip"


class Consumed(BaseModel):
    status: Literal["consumed"] = "consumed"
    nodes: list[Coordinate]


RecurseStrategyResult = Annotated[Union[Skip, Consumed], Field(discriminator="status")]

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
    ) -> RecurseStrategyResult: ...
