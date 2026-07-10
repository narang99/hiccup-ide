from dataclasses import dataclass


@dataclass
class DependencyMatch:
    """Everything resolved about a dependency-layer neuron at (dep_layer, dep_channel, dep_y, dep_x)."""

    dep_layer_name: str
    dep_channel: int
    dep_y: int
    dep_x: int
    best_cid: int
    best_sim: object
    dep_patch: object
    best_patches: object
    dep_w: object
    noise: object
    label_by_points: dict
    ratio: float
    point_dist: object
    op_act: object
