import torch
import torch.fx as fx
import json
from typing import Dict, Any


def model_to_json(model: torch.nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    """
    Convert a PyTorch model to JSON representation using torch.fx
    
    Args:
        model: PyTorch model to convert
        input_shape: Input tensor shape for shape inference
    
    Returns:
        Dictionary containing nodes and edges for graph visualization
    """
    try:
        # Trace the model
        traced = fx.symbolic_trace(model)
        
        # Get model parameters for shape inference
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        nodes = []
        edges = []
        
        # Track shapes through the network
        with torch.no_grad():
            # Hook to capture intermediate shapes
            shapes = {}
            def capture_shape(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        shapes[name] = list(output.shape)
                    elif isinstance(output, (list, tuple)):
                        shapes[name] = [list(o.shape) if isinstance(o, torch.Tensor) else str(o) for o in output]
                return hook
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if name:  # Skip empty name (root module)
                    hook = module.register_forward_hook(capture_shape(name))
                    hooks.append(hook)
            
            # Run forward pass to capture shapes
            _ = model(dummy_input)
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        # Process FX graph nodes
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                # Input node
                nodes.append({
                    # "id": node.name,
                    "id": node.target,
                    "type": "Input",
                    "params": {},
                    "shape": list(input_shape)
                })
            
            elif node.op == 'call_module':
                # Module call (Conv2d, ReLU, etc.)
                module = dict(model.named_modules())[node.target]
                module_type = type(module).__name__
                
                # Extract module parameters
                params = {}
                if hasattr(module, 'in_channels'):
                    params['in_channels'] = module.in_channels
                if hasattr(module, 'out_channels'):
                    params['out_channels'] = module.out_channels
                if hasattr(module, 'kernel_size'):
                    params['kernel_size'] = module.kernel_size
                if hasattr(module, 'stride'):
                    params['stride'] = module.stride
                if hasattr(module, 'padding'):
                    params['padding'] = module.padding
                if hasattr(module, 'num_features'):
                    params['num_features'] = module.num_features
                
                nodes.append({
                    # "id": node.name,
                    "id": node.target,
                    "type": module_type,
                    "params": params,
                    "shape": shapes.get(node.target, [])
                })
                
                # Add edges from inputs
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        edges.append({
                            # "source": arg.name,
                            # "target": node.name
                            "source": arg.target,
                            "target": node.target,
                        })
            
            elif node.op == 'call_function':
                # Function call (torch.flatten, F.relu, etc.)
                func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                
                nodes.append({
                    # "id": node.name,
                    "id": node.target,
                    "type": func_name,
                    "params": {},
                    "shape": []
                })
                
                # Add edges from inputs
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        edges.append({
                            # "source": arg.name,
                            # "target": node.name
                            "source": arg.target,
                            "target": node.target
                        })
            
            elif node.op == 'output':
                # Output node
                nodes.append({
                    # "id": node.name,
                    "id": node.target,
                    "type": "Output",
                    "params": {},
                    "shape": []
                })
                
                # Add edges from inputs
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        edges.append({
                            # "source": arg.name,
                            # "target": node.name
                            "source": arg.target,
                            "target": node.target,
                        })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    except Exception as e:
        # Fallback to named_modules if tracing fails
        print(f"torch.fx tracing failed: {e}")
        return _fallback_to_named_modules(model, input_shape)


def _fallback_to_named_modules(model: torch.nn.Module, input_shape: tuple) -> Dict[str, Any]:
    """Fallback method using named_modules"""
    nodes = []
    edges = []
    
    # Add input node
    nodes.append({
        "id": "input",
        "type": "Input",
        "params": {},
        "shape": list(input_shape)
    })
    
    prev_node = "input"
    
    for name, module in model.named_modules():
        if name:  # Skip root module
            module_type = type(module).__name__
            
            params = {}
            if hasattr(module, 'in_channels'):
                params['in_channels'] = module.in_channels
            if hasattr(module, 'out_channels'):
                params['out_channels'] = module.out_channels
            if hasattr(module, 'kernel_size'):
                params['kernel_size'] = module.kernel_size
            
            nodes.append({
                "id": name,
                "type": module_type,
                "params": params,
                "shape": []
            })
            
            edges.append({
                "source": prev_node,
                "target": name
            })
            
            prev_node = name
    
    return {
        "nodes": nodes,
        "edges": edges
    }


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    
    # Test with InceptionV1 (GoogLeNet)
    model = models.googlenet(pretrained=False)
    graph_json = model_to_json(model)
    
    # Save to file
    with open('inception_graph.json', 'w') as f:
        json.dump(graph_json, f, indent=2)
    
    print(f"Generated graph with {len(graph_json['nodes'])} nodes and {len(graph_json['edges'])} edges")