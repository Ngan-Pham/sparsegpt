import torch
import torch.nn as nn

def collect_layer_scales(layer, inps, attention_mask=None, alibi=None):
    """
    Registers forward hooks on all linear layers in the decoder block,
    passes inputs through the block to collect activation scales, and removes hooks.
    """
    scales = {}
    hooks = []
    
    def get_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].data
            if len(x.shape) == 3:
                x = x.reshape(-1, x.shape[-1])
            elif len(x.shape) == 2:
                pass
            abs_max = torch.max(torch.abs(x), dim=0)[0].detach().cpu()
            if name not in scales:
                scales[name] = abs_max
            else:
                scales[name] = torch.maximum(scales[name], abs_max)
        return hook_fn

    # Register hook on every nn.Linear module
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    # Run a subset of inputs to collect scales
    num_samples = min(32, len(inps))
    
    # We must be in eval and no_grad mode
    layer.eval()
    for j in range(num_samples):
        kwargs = {}
        if attention_mask is not None:
            kwargs['attention_mask'] = attention_mask
        if alibi is not None:
            kwargs['alibi'] = alibi
            
        with torch.no_grad():
            layer(inps[j].unsqueeze(0), **kwargs)
            
    # Remove all hooks
    for hook in hooks:
        hook.remove()
        
    return scales

def smooth_group(linears, layernorm, scales, layer, alpha=0.5):
    """
    Performs weight scaling and LayerNorm scale division for a group of linear layers.
    """
    if not linears:
        return
        
    device = linears[0].weight.device
    dtype = linears[0].weight.dtype
    
    # Find relative names of linears to retrieve their activation scales
    names = []
    for name, module in layer.named_modules():
        if module in linears:
            names.append(name)
            
    # Compute max activation scale across the linears in the group
    act_scale = None
    for name in names:
        if name in scales:
            if act_scale is None:
                act_scale = scales[name].to(device=device, dtype=dtype)
            else:
                act_scale = torch.maximum(act_scale, scales[name].to(device=device, dtype=dtype))
                
    if act_scale is None:
        return
        
    # Get max absolute weight per input channel across all linears in group
    w_scale = []
    for lin in linears:
        w_max = torch.max(torch.abs(lin.weight.data), dim=0)[0]
        w_scale.append(w_max)
    w_scale = torch.stack(w_scale, dim=0)
    w_scale = torch.max(w_scale, dim=0)[0]
    
    # Compute smoothing factor s
    s = (act_scale ** alpha) / (w_scale ** (1 - alpha)).clamp(min=1e-5)
    s = s.clamp(min=1e-5)
    
    # Scale linear weights: W = W * diag(s) (multiply columns by s)
    for lin in linears:
        lin.weight.data.mul_(s.view(1, -1))
        
    # Divide layernorm scales: LN.weight = LN.weight / s, LN.bias = LN.bias / s
    if hasattr(layernorm, 'weight') and layernorm.weight is not None:
        layernorm.weight.data.div_(s)
    if hasattr(layernorm, 'bias') and layernorm.bias is not None:
        layernorm.bias.data.div_(s)

def apply_smoothquant_to_layer(layer, model_type, scales, alpha=0.5):
    """
    Applies SmoothQuant to a decoder layer based on the model architecture type.
    """
    model_type = model_type.lower()
    
    if 'llama' in model_type:
        # Group 1: Attn projections (q, k, v) and input_layernorm
        qkv_modules = []
        if hasattr(layer.self_attn, 'q_proj'):
            qkv_modules.append(layer.self_attn.q_proj)
        if hasattr(layer.self_attn, 'k_proj'):
            qkv_modules.append(layer.self_attn.k_proj)
        if hasattr(layer.self_attn, 'v_proj'):
            qkv_modules.append(layer.self_attn.v_proj)
            
        if qkv_modules and hasattr(layer, 'input_layernorm'):
            smooth_group(qkv_modules, layer.input_layernorm, scales, layer, alpha)
            
        # Group 2: MLP gate and up projections and post_attention_layernorm
        mlp_modules = []
        if hasattr(layer.mlp, 'gate_proj'):
            mlp_modules.append(layer.mlp.gate_proj)
        if hasattr(layer.mlp, 'up_proj'):
            mlp_modules.append(layer.mlp.up_proj)
            
        if mlp_modules and hasattr(layer, 'post_attention_layernorm'):
            smooth_group(mlp_modules, layer.post_attention_layernorm, scales, layer, alpha)
            
    elif 'opt' in model_type:
        # Group 1: Attn projections (q, k, v) and self_attn_layer_norm
        qkv_modules = []
        if hasattr(layer.self_attn, 'q_proj'):
            qkv_modules.append(layer.self_attn.q_proj)
        if hasattr(layer.self_attn, 'k_proj'):
            qkv_modules.append(layer.self_attn.k_proj)
        if hasattr(layer.self_attn, 'v_proj'):
            qkv_modules.append(layer.self_attn.v_proj)
            
        if qkv_modules and hasattr(layer, 'self_attn_layer_norm'):
            smooth_group(qkv_modules, layer.self_attn_layer_norm, scales, layer, alpha)
            
        # Group 2: MLP fc1 projection and final_layer_norm
        mlp_modules = []
        if hasattr(layer, 'fc1'):
            mlp_modules.append(layer.fc1)
            
        if mlp_modules and hasattr(layer, 'final_layer_norm'):
            smooth_group(mlp_modules, layer.final_layer_norm, scales, layer, alpha)
            
    elif 'bloom' in model_type:
        # Group 1: Attn query_key_value projection and input_layernorm
        qkv_modules = []
        if hasattr(layer.self_attn, 'query_key_value'):
            qkv_modules.append(layer.self_attn.query_key_value)
            
        if qkv_modules and hasattr(layer, 'input_layernorm'):
            smooth_group(qkv_modules, layer.input_layernorm, scales, layer, alpha)
            
        # Group 2: MLP dense_h_to_4h projection and post_attention_layernorm
        mlp_modules = []
        if hasattr(layer.mlp, 'dense_h_to_4h'):
            mlp_modules.append(layer.mlp.dense_h_to_4h)
            
        if mlp_modules and hasattr(layer, 'post_attention_layernorm'):
            smooth_group(mlp_modules, layer.post_attention_layernorm, scales, layer, alpha)
