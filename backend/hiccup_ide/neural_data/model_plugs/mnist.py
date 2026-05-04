from pt_to_api.weights_processor import process_model_weights_to_coordinates
from pt_to_api.contrib_processor import process_contribs_to_coordinates
from pt_to_api.activation_processor import process_activations_to_coordinates
from pt_to_api.mnist import get_contribs_for_inp_vectorized
from pt_to_api.capture import get_model_internals


def get_processed_model(model_pt_file):
    "return the processed model schema and weights for storing in Model and Weight class in the database"
    from pt_to_api.model_to_json import model_to_json

    input_shape = (1, 1, 28, 28)
    model = _get_loaded_model(model_pt_file)
    weight_coordinates = process_model_weights_to_coordinates(model)
    model_schema = model_to_json(model, input_shape)
    return model_schema, weight_coordinates


def _get_loaded_model(model_pt_file):
    from pt_to_api.mnist import SimpleMNIST
    import torch

    model = SimpleMNIST()
    model.load_state_dict(torch.load(model_pt_file, map_location="cpu"))
    model.to("cpu")
    model.eval()
    return model


def load_input_tensor(input_pt_file):
    import torch

    return torch.load(input_pt_file, weights_only=False, map_location="cpu")


def get_processed_activations_and_contribs(model, input_tensor, label):
    "return processed activation and saliency maps, to be stored in respective models"
    # label can be 0, 1, 2 and all (int)
    from pt_to_api.utils import zeros_with_1_at

    TOTAL_CLASSES_IN_MNIST = 10
    model = _get_loaded_model(model)
    activations, parameters = get_model_internals(model, input_tensor)
    contribs, _, _ = get_contribs_for_inp_vectorized(
        input_tensor,
        model,
        zeros_with_1_at(TOTAL_CLASSES_IN_MNIST, label),
        "layers.5",
        "cpu",
    )
    act_coords = process_activations_to_coordinates(activations, parameters, model)
    contrib_coords = process_contribs_to_coordinates(contribs, 0)
    return act_coords, contrib_coords
