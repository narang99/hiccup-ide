from collections import OrderedDict

import numpy as np
import torch
from lucent.optvis import objectives, param, transform


def get_feature_viz_input(
    model,
    objective_f,
    param_f=None,
    transforms=None,
    preprocess=True,
    fixed_image_size=None,
):
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    # if optimizer is None:
    #     optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    # optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook, features = hook_model(model, image_f, return_hooks=True)
    objective_f = objectives.as_objective(objective_f)

    return (image_f, params, transform_f, objective_f, hook, features)


# def opt_loop(
#     thresholds,
#     progress,
#     model,
#     optimizer,
#     transform_f,
#     image_f,
#     objective_f,
#     hook,
#     verbose,
#     features,
# ):
#     images = []
#     if verbose:
#         model(transform_f(image_f()))
#         print("Initial loss: {:.3f}".format(objective_f(hook)))
#     try:
#         for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):

#             def closure():
#                 optimizer.zero_grad()
#                 try:
#                     model(transform_f(image_f()))
#                 except RuntimeError as ex:
#                     if i == 1:
#                         # Only display the warning message
#                         # on the first iteration, no need to do that
#                         # every iteration
#                         warnings.warn(
#                             "Some layers could not be computed because the size of the "
#                             "image is not big enough. It is fine, as long as the non"
#                             "computed layers are not used in the objective function"
#                             f"(exception details: '{ex}')"
#                         )
#                 loss = objective_f(hook)
#                 loss.backward()
#                 return loss

#             optimizer.step(closure)
#             if i in thresholds:
#                 image = tensor_to_img_array(image_f())
#                 if verbose:
#                     print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
#                 images.append(image)
#     except KeyboardInterrupt:
#         print("Interrupted optimization at step {:d}.".format(i))
#         if verbose:
#             print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
#         images.append(tensor_to_img_array(image_f()))

#     # Clear hooks
#     for module_hook in features.values():
#         del module_hook.module._forward_hooks[module_hook.hook.id]


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    # Check if the image is single channel and convert to 3-channel
    if len(image.shape) == 4 and image.shape[3] == 1:  # Single channel image
        image = np.repeat(image, 3, axis=3)
    return image


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        # This doesn't actually do anything
        self.hook.remove()


def hook_model(model, image_f, return_hooks=False):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, (
                f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            )
            out = features[layer].features
        assert out is not None, (
            "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        )
        return out

    if return_hooks:
        return hook, features
    return hook
