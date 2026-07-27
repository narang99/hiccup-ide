from olt.models.cifar_inception import (
    ANALYSABLE_1X1_LAYERS,
    BLOCK_BRANCHES,
    CifarInception,
    InceptionBlock,
    cifar_inception,
)
from olt.models.stl_inception import (
    StlInception,
    StlInceptionBlock,
    stl_inception,
)

# NOTE: stl_inception also defines BLOCK_BRANCHES / ANALYSABLE_1X1_LAYERS
# (byte-identical values, same layer names). They are NOT re-exported here to
# avoid a top-level name clash with the CIFAR ones — import them from
# `olt.models.stl_inception` directly if a caller needs the STL model's copy.

__all__ = [
    "ANALYSABLE_1X1_LAYERS",
    "BLOCK_BRANCHES",
    "CifarInception",
    "InceptionBlock",
    "StlInception",
    "StlInceptionBlock",
    "cifar_inception",
    "stl_inception",
]
