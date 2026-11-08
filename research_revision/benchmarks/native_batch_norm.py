from types import MethodType

_registered_forward = None

def _native_operators(torch):
    global _registered_forward
    if _registered_forward is not None:
        return _registered_forward
    aten = torch.ops.aten

    @torch.library.custom_op(
        "lgts_cost_bn::training_forward", mutates_args=(), device_types="cuda",
        schema="(Tensor x, Tensor weight, Tensor bias, Tensor mean, Tensor var, "
               "float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    )
    def forward(x, weight, bias, mean, var, momentum, eps):

        updated_mean, updated_var = mean.clone(), var.clone()
        y, saved_mean, invstd, reserve, implementation = aten._batch_norm_impl_index.default(
            x, weight, bias, updated_mean, updated_var, True, momentum, eps, True
        )
        if implementation != 1:
            raise RuntimeError(f"Native cuDNN training BatchNorm required, got backend {implementation}")
        return y, saved_mean, invstd, reserve, updated_mean, updated_var

    @forward.register_fake
    def fake_forward(x, weight, bias, mean, var, momentum, eps):


        reserve_size = torch._C._get_cudnn_batch_norm_reserve_space_size(x, True)
        reserve = x.new_empty((reserve_size,), dtype=torch.uint8)
        return (torch.empty_like(x), torch.empty_like(mean), torch.empty_like(var),
                reserve, torch.empty_like(mean), torch.empty_like(var))

    @torch.library.custom_op(
        "lgts_cost_bn::training_backward", mutates_args=(), device_types="cuda",
        schema="(Tensor x, Tensor dy, Tensor weight, Tensor mean, Tensor var, "
               "Tensor saved_mean, Tensor invstd, Tensor reserve, float eps) -> (Tensor, Tensor, Tensor)",
    )
    def backward_kernel(x, dy, weight, mean, var, saved_mean, invstd, reserve, eps):
        return aten._batch_norm_impl_index_backward.default(
            1, x, dy, weight, mean, var, saved_mean, invstd, True, eps, [True] * 3, reserve
        )

    @backward_kernel.register_fake
    def fake_backward(x, dy, weight, mean, var, saved_mean, invstd, reserve, eps):
        return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(weight)

    def setup_context(ctx, inputs, output):
        x, weight, _bias, _mean, _var, _momentum, ctx.eps = inputs
        _y, saved_mean, invstd, reserve, mean, var = output
        ctx.mark_non_differentiable(saved_mean, invstd, reserve, mean, var)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weight, mean, var, saved_mean, invstd, reserve)

    def backward(ctx, dy, *_auxiliary):
        x, weight, mean, var, saved_mean, invstd, reserve = ctx.saved_tensors
        dx, dw, db = backward_kernel(x, dy, weight, mean, var, saved_mean, invstd, reserve, ctx.eps)
        return dx, dw, db, None, None, None, None

    forward.register_autograd(backward, setup_context=setup_context)
    _registered_forward = forward
    return forward

BACKBONES = {
    "EfficientNet": ("EfficientNetV2S", 110, "features.0.1"),
    "MobileNetV2": ("MobileNetV2", 52, "features.0.1"),
    "ResNet": ("ResNet18", 20, "bn1"),
}


def compile_builtin_training_graph(torch, model, evidence):

    if not model.training or not torch.backends.cudnn.enabled:
        raise ValueError("The panel supplement requires cuDNN and a training-mode model")
    backbone = BACKBONES.get(type(model).__name__)
    if backbone is None:
        raise ValueError(f"Unrecognized panel backbone: {type(model).__name__}")
    name, count, stem = backbone
    layers = [(key, layer) for key, layer in model.named_modules()
              if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    if len(layers) != count or layers[0][0] != stem:
        raise ValueError(f"Unexpected BatchNorm layout in {name}")
    if any(type(layer) is not torch.nn.BatchNorm2d or not layer.training or not layer.affine
           or not layer.track_running_stats or layer.momentum is None for _, layer in layers):
        raise ValueError(f"Unsupported native BatchNorm configuration in {name}")

    native_forward = _native_operators(torch)

    def training_forward(module, x):
        if not module.training:
            raise RuntimeError("Training BatchNorm invoked outside training mode")
        module.num_batches_tracked.add_(1)
        y, _, _, _, mean, var = native_forward(
            x, module.weight, module.bias, module.running_mean, module.running_var,
            module.momentum, module.eps,
        )
        module.running_mean.copy_(mean)
        module.running_var.copy_(var)
        return y

    for _, layer in layers:
        layer.forward = MethodType(training_forward, layer)
    evidence.update(backend="cudagraphs", fullgraph=True, dynamic=False, backbone=name,
                    native_training_batch_norm_count=count, first_batch_norm=stem,
                    momentum_values=sorted({layer.momentum for _, layer in layers}),
                    eps_values=sorted({layer.eps for _, layer in layers}),
                    forward_dispatch="aten._batch_norm_impl_index (cuDNN implementation 1)",
                    backward_dispatch="aten._batch_norm_impl_index_backward (cuDNN implementation 1, saved reserve)")
    return torch.compile(model, backend="cudagraphs", fullgraph=True, dynamic=False)
