import torch
import torch.nn.functional as F


def compute_conv_input_channel_contributions(module, input_tensor):
    """
    Compute individual input channel contributions for a conv layer.
    Returns a dictionary with keys like 'out_{out_ch}.in_{in_ch}' 
    """
    if not hasattr(module, 'weight'):
        return {}
        
    contributions = {}
    x = input_tensor[0]  # [batch, in_channels, H, W]
    weight = module.weight  # [out_channels, in_channels, kH, kW]
    bias = module.bias if hasattr(module, 'bias') and module.bias is not None else None
    
    # Ensure input has batch dimension
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            # Extract single input channel and corresponding kernel slice
            input_slice = x[:, in_ch:in_ch+1]  # [batch, 1, H, W]
            kernel_slice = weight[out_ch:out_ch+1, in_ch:in_ch+1]  # [1, 1, kH, kW]
            
            # Compute convolution for this specific input->output channel pair
            contribution = F.conv2d(
                input_slice, 
                kernel_slice,
                bias=None,  # Don't add bias in conv2d call
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=1
            )
            
            # Add evenly distributed bias if present
            if bias is not None:
                # Each input channel gets bias[out_ch] / in_channels portion
                bias_portion = bias[out_ch] / in_channels
                contribution = contribution + bias_portion
            
            coord_key = f"out_{out_ch}.in_{in_ch}"
            contributions[coord_key] = contribution.detach().cpu()
    
    return contributions


class ModelSnapshot:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.parameters = {}  # Dictionary to store weights and biases
        self.hooks = []
        
        # Capture static parameters once during init
        self._collect_parameters()
        # Register hooks for dynamic activations
        self._register_hooks()

    def _collect_parameters(self):
        """Extracts weights and biases for all layers that possess them."""
        for name, module in self.model.named_modules():
            layer_params = {}
            
            if hasattr(module, 'weight') and module.weight is not None:
                layer_params['weight'] = module.weight.detach().cpu()
            
            if hasattr(module, 'bias') and module.bias is not None:
                layer_params['bias'] = module.bias.detach().cpu()
            
            # Store conv-specific parameters for slice-wise computation
            if hasattr(module, '__class__') and 'Conv' in module.__class__.__name__:
                layer_params['stride'] = getattr(module, 'stride', 1)
                layer_params['padding'] = getattr(module, 'padding', 0)
                layer_params['dilation'] = getattr(module, 'dilation', 1)
                layer_params['groups'] = getattr(module, 'groups', 1)
            
            # Only add to the dictionary if the layer actually had parameters or config
            if layer_params:
                self.parameters[name] = layer_params

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            # Store the main activation output
            self.activations[name] = output.detach().cpu()
            
            # For Conv layers, also compute input channel contributions
            if hasattr(module, '__class__') and 'Conv' in module.__class__.__name__:
                input_contributions = compute_conv_input_channel_contributions(module, input)
                # Store with layer name prefix
                for coord_key, contribution in input_contributions.items():
                    full_key = f"{name}.{coord_key}"
                    self.activations[full_key] = contribution
                    
        return hook_fn

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_model_internals(model, input_tensor):
    snapshot = ModelSnapshot(model)
    model.eval()

    with torch.no_grad():
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        _ = model(input_tensor)

    # Clean up hooks but keep the data
    activations = snapshot.activations
    parameters = snapshot.parameters
    snapshot.remove()
    
    return activations, parameters