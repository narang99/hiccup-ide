from neural_data.models import Weight, SaliencyMap, Activation
from tqdm import tqdm
from pt_to_api.utils import get_receptive
import numpy as np


def get_full_conv_kernel_at_coordinate(coordinate: str):
    weight_objs = list(Weight.objects.filter(coordinate__startswith=coordinate, data_type="weights"))
    return np.stack([w.data for w in weight_objs])

def get_all_saliency_map_ids_and_patches(
    kernel_coordinate: str,
    input_layer_name: str,
    pos_contrib_thres: float,
    neg_contrib_thres: float,
):
    sm = SaliencyMap.objects.filter(coordinate=kernel_coordinate).first()
    if sm is None:
        print(f"Could not find any saliency map for coordainte={kernel_coordinate}")
        return None
    R = len(sm.data)
    W = len(sm.data[0])
    all_patches = []
    all_sm_ids = []
    for r in range(R):
        print(f"step {r}/{R}")
        for c in range(W):
            sm_ids, patches = get_saliency_map_ids_and_patches(
                (r,c), kernel_coordinate, input_layer_name, pos_contrib_thres, neg_contrib_thres
            )
            all_sm_ids.extend(sm_ids)
            all_patches.extend(patches)
    return all_sm_ids, all_patches


def get_saliency_map_ids_and_patches(
    output_grid_coord: tuple[int, int],
    kernel_coordinate: str,
    input_layer_name: str,
    pos_contrib_thres: float,
    neg_contrib_thres: float,
):
    r, c = output_grid_coord
    (y0,x0), (y1,x1) = get_receptive(r, c, 3, 2, 1)
    sm_ids = []
    patches = []

    for sm in tqdm(SaliencyMap.objects.filter(coordinate=kernel_coordinate)):
        v = sm.data[r][c]
        if v >= 0 and np.abs(v) < np.abs(pos_contrib_thres):
            continue
        if v < 0 and np.abs(v) < np.abs(neg_contrib_thres):
            continue
        # acts = Activation.objects.filter(coordinate__startswith=input_layer_name, input=sm.input).order_by("coordinate")
        # acts = np.stack([a.data for a in acts])
        acts = get_full_activations_of_layer(input_layer_name, sm.input)
        patch = acts[:, y0:y1, x0:x1]
        sm_ids.append({"coordinate": sm.coordinate, "input": sm.input.alias, "grid_coord": (r,c)})
        patches.append(patch)
    return sm_ids, patches


def get_full_activations_of_layer(layer_name, input):
    acts = Activation.objects.filter(coordinate__startswith=layer_name, input=input).order_by("coordinate")
    acts = np.stack([a.data for a in acts])
    return acts