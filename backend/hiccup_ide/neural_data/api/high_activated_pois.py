from fontTools.varLib.instancer.names import _updateUniqueIdNameRecord
from typing import Optional
from django.shortcuts import get_object_or_404
from ..models import Model, Input, SaliencyMap, Activation, Weight
from ..schemas import (
    InputLayerMeta,
    POIPoint,
    HighActivatedPOIOut,
    HighActivatedPOIsResponse,
    ActivationOut,
    UniqueActivationId,
)


def _get_input_layers(
    model_definition: dict, current_layer_name: str
) -> list[InputLayerMeta]:
    "Find input layers for a given layer by traversing the model graph."

    # Find nodes and edges in model definition
    nodes = model_definition.get("nodes", [])
    edges = model_definition.get("edges", [])

    # Create a mapping of node ids to node data
    node_map = {node["id"]: node for node in nodes}

    # Find input layers by looking at edges that target our layer
    input_layers = []
    for edge in edges:
        if edge["target"] == current_layer_name:
            source_node_id = edge["source"]
            if source_node_id in node_map:
                source_node = node_map[source_node_id]
                input_layers.append(
                    InputLayerMeta(type=source_node["type"], layer_name=source_node_id)
                )

    return input_layers


def _get_input_slice_coordinate_of_conv_kernel_slice_v2(
    input_layer_meta: InputLayerMeta,
    weight: Weight,
) -> str:
    if (
        weight.coordinate_type != "input_output_channel"
        or weight.layer_type != "Conv2d"
    ):
        raise ValueError(
            f"This function only supports Conv2d input_output_channel coordinates, got {weight.layer_type} {weight.coordinate_type}"
        )
    if weight.output_channel is None or weight.input_channel is None:
        raise ValueError(
            f"Weight has null output or input channel. output_channel={weight.output_channel} input_channel={weight.input_channel}, coordinate={weight.coordinate}"
        )
    return f"{input_layer_meta.layer_name}.out_{weight.input_channel}"


def _get_input_slice_coordinate_of_conv_kernel_slice(
    input_layer_meta: InputLayerMeta,
    out_coordinate: str,
    output_layer_name: str,
    coordinate_type: str,
    layer_type: str,
) -> str:
    "Convert output coordinate to corresponding input coordinate."
    # Validate that this is for Conv2d input_output_channel coordinates
    if coordinate_type != "input_output_channel" or layer_type != "Conv2d":
        raise ValueError(
            f"This function only supports Conv2d input_output_channel coordinates, got {layer_type} {coordinate_type}"
        )

    # Cut off the output layer name from the coordinate and replace with input layer name
    if out_coordinate.startswith(output_layer_name + "."):
        remaining_coordinate = out_coordinate[len(output_layer_name) + 1 :]
        return f"{input_layer_meta.layer_name}.{remaining_coordinate}"
    else:
        raise Exception(
            f"output coordinate {out_coordinate} does not start with layer name={output_layer_name}"
        )


def _flattened_contribs_of_one_saliency_map(saliency_map: SaliencyMap):
    all_contributions = []
    if not isinstance(saliency_map.data, list):
        print(
            f"WARN: Saliency map input={saliency_map.input.pk} coordinate={saliency_map.coordinate}, `data` attribute is not a list, type={type(saliency_map.data)}"
        )
        return None

    for row_idx, row in enumerate(saliency_map.data):
        if not isinstance(row, list):
            print(
                f"WARN: Saliency map input={saliency_map.input.pk} coordinate={saliency_map.coordinate}, data is not a 2D nested list, type of element={type(row)}"
            )
            return None
        for col_idx, value in enumerate(row):
            if isinstance(value, (int, float)):
                all_contributions.append((row_idx, col_idx, float(value), saliency_map))
    return all_contributions


def _find_highest_contributions(
    saliency_maps: list[SaliencyMap], k: int = 10
) -> list[tuple[int, int, float, SaliencyMap]]:
    "Find the highest K contributions from a list of saliency maps."
    all_contributions = []

    for saliency_map in saliency_maps:
        single_contribs = _flattened_contribs_of_one_saliency_map(saliency_map)
        if single_contribs is None:
            print(
                f"WARN: saliency map={saliency_map} , id={saliency_map.pk} gave empty contribs, maybe it wasn't the correct shape?"
            )
        else:
            all_contributions.extend(single_contribs)

    # Sort by value in descending order and take top K
    all_contributions.sort(key=lambda x: x[2], reverse=True)
    return all_contributions[:k]


def _collect_saliency_maps_for_coordinate(
    model: Model, coordinate: str
) -> tuple[list[SaliencyMap], dict]:
    "Collect all saliency maps for a coordinate across all inputs."
    # Query all saliency maps for this coordinate across all inputs of the model
    saliency_maps = list(
        SaliencyMap.objects.filter(
            input__model=model, coordinate=coordinate
        ).select_related("input")
    )

    # Create mapping from saliency map id to input object
    input_mapping = {smap.pk: smap.input for smap in saliency_maps}

    return saliency_maps, input_mapping


def _create_activation_out(activation: Activation) -> ActivationOut:
    """
    Convert an Activation model instance to ActivationOut schema.
    """
    return ActivationOut(
        id=activation.pk,
        coordinate=activation.coordinate,
        data=activation.data,
        shape=activation.shape,
        layer_type=activation.layer_type,
        coordinate_type=activation.coordinate_type,
        output_channel=activation.output_channel,
        input_channel=activation.input_channel,
    )


def _get_input_activations_for_poi(
    input_obj: Input,
    input_coordinates: list[str],
) -> Optional[list[UniqueActivationId]]:
    """
    Get all input activations for a given POI coordinate.
    None if any of the input activations are not found for all input layers
    """
    input_activations = []

    for input_coordinate in input_coordinates:
        try:
            input_activations.append(
                UniqueActivationId(
                    input_alias=input_obj.alias,
                    model_alias=input_obj.model.alias,
                    coordinate=input_coordinate,
                )
            )
        except Activation.DoesNotExist:
            print(
                f"WARN: activation not found for input={input_obj.alias} coordinate={input_coordinate}"
            )
            return None

    return input_activations


def _create_poi_from_contribution(
    row: int,
    col: int,
    value: float,
    saliency_map: SaliencyMap,
    input_mapping: dict[int, Input],
    coordinate: str,
    input_coordinates: list[str],
) -> Optional[HighActivatedPOIOut]:
    "Create a HighActivatedPOIOut from a single contribution."
    input_obj = input_mapping[saliency_map.pk]

    try:
        input_activations = _get_input_activations_for_poi(
            input_obj, input_coordinates
        )
        if input_activations is None:
            return None
        return HighActivatedPOIOut(
            output_activation=UniqueActivationId(
                input_alias=input_obj.alias,
                model_alias=input_obj.model.alias,
                coordinate=coordinate,
            ),
            input_activations=input_activations,
            points=[POIPoint(row=row, col=col, value=value)],
        )
    except Activation.DoesNotExist:
        print(
            f"WARN: activation not found for input={input_obj.alias} coordinate={coordinate}"
        )
        return None


def _get_input_coordinates(model: Model, coordinate: str):
    weight = get_object_or_404(Weight, model=model, coordinate=coordinate)
    input_layers = _get_input_layers(model.definition, weight.layer_name)
    return [
        _get_input_slice_coordinate_of_conv_kernel_slice_v2(
            input_layer,
            weight,
        )
        for input_layer in input_layers
    ]


def _contributions_to_high_activated_pois(
    top_contributions, input_mapping, coordinate, input_coordinates
):
    results = []
    for row, col, value, saliency_map in top_contributions:
        poi = _create_poi_from_contribution(
            row,
            col,
            value,
            saliency_map,
            input_mapping,
            coordinate,
            input_coordinates,
        )
        if poi:
            results.append(poi)
    return results


def get_high_activated_pois_for_slice_coordinate(
    model_alias: str, coordinate: str, k: int = 10
) -> HighActivatedPOIsResponse:
    """
    Main function to get high-activated POIs for a slice coordinate.

    Args:
        model_alias: Model alias
        coordinate: Slice coordinate
        k: Number of top contributions to return

    Returns:
        HighActivatedPOIsResponse object
    """
    model = get_object_or_404(Model, alias=model_alias)
    saliency_maps, input_mapping = _collect_saliency_maps_for_coordinate(
        model, coordinate
    )
    print("unique inputs", set([i.pk for i in input_mapping.values()]))
    if not saliency_maps:
        return HighActivatedPOIsResponse(pois=[])
    input_coordinates = _get_input_coordinates(model, coordinate)
    top_contributions = _find_highest_contributions(saliency_maps, k)
    results = _contributions_to_high_activated_pois(
        top_contributions, input_mapping, coordinate, input_coordinates
    )

    return HighActivatedPOIsResponse(pois=results)
