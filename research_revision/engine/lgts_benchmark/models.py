from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from types import MethodType

from .activations import AlgebraicBaseLGTS, AlgebraicLogisticLGTS, LGTSBase, make_activation


def _key(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(" ", "")


def is_native_activation(value: str) -> bool:
    return _key(value) in {"native", "default", "original"}


def native_activation_variant(model_name: str) -> str:
    key = _key(model_name)
    if key.startswith("resnet"):
        return "Native-ReLU"
    if key == "mobilenetv2":
        return "Native-ReLU6"
    if key in {"efficientnetb0", "efficientnetv2s"}:
        return "Native-SiLU"
    if key == "convnexttiny":
        return "Native-GELU"
    return "Native"


def _activation_kwargs(
    log_t: float,
    lgts_learnable: bool,
    lgts_learn_alpha: bool | None,
    lgts_learn_beta: bool | None,
    lgts_alpha: float,
    lgts_beta: float,
    lgts_alpha_min: float,
    lgts_alpha_max: float,
    lgts_beta_min: float,
    lgts_beta_max: float,
) -> dict:
    return {
        "log_t": log_t,
        "lgts_learnable": lgts_learnable,
        "lgts_learn_alpha": lgts_learn_alpha,
        "lgts_learn_beta": lgts_learn_beta,
        "lgts_alpha": lgts_alpha,
        "lgts_beta": lgts_beta,
        "lgts_alpha_min": lgts_alpha_min,
        "lgts_alpha_max": lgts_alpha_max,
        "lgts_beta_min": lgts_beta_min,
        "lgts_beta_max": lgts_beta_max,
    }


class PaperCNN(nn.Module):
    """Controlled shallow CNN used only as a supplementary architecture."""

    def __init__(self, in_channels: int, num_classes: int, activation: str, activation_kwargs: dict):
        super().__init__()
        blocks = []
        channels = in_channels
        for _ in range(3):
            blocks.extend([
                nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                make_activation(activation, channels=32, **activation_kwargs),
                nn.MaxPool2d(2),
            ])
            channels = 32
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            make_activation(activation, channels=256, **activation_kwargs),
            nn.Linear(256, num_classes),
        )
        self.apply(self._he_init)

    @staticmethod
    def _he_init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def _activation_types() -> tuple[type[nn.Module], ...]:
    # Deliberately excludes Sigmoid/Softmax so attention, SE and output gates
    # are never replaced during full-backbone experiments.
    return (
        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.PReLU, nn.GELU,
        nn.SiLU, nn.Mish, nn.Hardswish, nn.Tanh,
    )


def _preceding_channels(module: nn.Module, child_name: str) -> int | None:
    # torchvision's SqueezeExcitation registers ``fc2`` before ``activation``
    # even though forward() applies activation immediately after ``fc1``.
    # Looking only at registration order therefore assigns input_channels to
    # a tensor whose real channel count is squeeze_channels. Scalar
    # activations hide this bug, while channel-wise Qian-Gated and ACON-C
    # correctly reject the mismatch.
    if (
        child_name == "activation"
        and isinstance(getattr(module, "fc1", None), nn.Conv2d)
        and isinstance(getattr(module, "fc2", None), nn.Conv2d)
    ):
        return int(module.fc1.out_channels)

    siblings = list(module.named_children())
    position = next((index for index, (name, _) in enumerate(siblings) if name == child_name), 0)
    for _, sibling in reversed(siblings[:position]):
        if isinstance(sibling, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return int(sibling.num_features)
        if isinstance(sibling, nn.Conv2d):
            return int(sibling.out_channels)
        if isinstance(sibling, nn.Linear):
            return int(sibling.out_features)
    return None


def _replace_activations(module: nn.Module, activation: str, activation_kwargs: dict) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, _activation_types()):
            channels = _preceding_channels(module, child_name)
            setattr(module, child_name, make_activation(activation, channels=channels, **activation_kwargs))
        else:
            _replace_activations(child, activation, activation_kwargs)


class SplitActivationBasicBlock(nn.Module):
    """Torchvision BasicBlock with independent modules at its two call sites."""

    expansion = 1

    def __init__(self, block: nn.Module, activation: str, activation_kwargs: dict) -> None:
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride
        channels = int(block.bn2.num_features)
        if is_native_activation(activation):
            self.act1 = nn.ReLU(inplace=False)
            self.act2 = nn.ReLU(inplace=False)
        else:
            self.act1 = make_activation(activation, channels=channels, **activation_kwargs)
            self.act2 = make_activation(activation, channels=channels, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        # Out-of-place addition is compatible with every candidate activation.
        out = out + identity
        return self.act2(out)


def _split_resnet18_activations(
    model: nn.Module,
    activation: str,
    activation_kwargs: dict,
    activation_scope: str,
) -> None:
    layers = (model.layer1, model.layer2, model.layer3, model.layer4)
    target_layers = layers if activation_scope == "full_backbone" else (model.layer4,)
    for layer in target_layers:
        for index, block in enumerate(layer):
            layer[index] = SplitActivationBasicBlock(block, activation, activation_kwargs)


def _adapt_resnet_small_input_stem(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """Use a 3x3/stride-1 stem and no max-pool for inputs up to 64 px."""
    old = model.conv1
    replacement = nn.Conv2d(
        in_channels, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False,
    )
    with torch.no_grad():
        if pretrained:
            weight = old.weight
            if in_channels != old.in_channels:
                weight = weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            replacement.weight.copy_(weight[:, :, 2:5, 2:5])
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = replacement
    model.maxpool = nn.Identity()
    model.stem_type = "small_3x3_stride1_no_maxpool"


def _adapt_conv_stem_stride_one(model: nn.Module, stem_label: str) -> None:
    """Change an ImageNet stem's first spatial convolution to stride 1.

    MobileNetV2 and EfficientNet otherwise downsample a 32x32 input before the
    first feature stage. Keeping the kernel/weights unchanged makes the
    adaptation explicit and comparable while preserving the model family.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.stride = (1, 1)
            model.stem_type = stem_label
            return
    raise RuntimeError("No convolutional stem was found for small-input adaptation")


def _make_efficientnet_residuals_out_of_place(model: nn.Module) -> None:
    """Avoid torchvision EfficientNet's ``result += input`` residual update.

    Some replacement activations need their output unchanged for backward.
    Out-of-place addition is mathematically identical and lets the ReLU control
    use native ``nn.ReLU`` rather than a slower torch.where implementation.
    """
    def forward(block, x):
        result = block.block(x)
        if block.use_res_connect:
            result = block.stochastic_depth(result)
            result = result + x
        return result

    for module in model.modules():
        if (
            module.__class__.__name__ in {"MBConv", "FusedMBConv"}
            and hasattr(module, "block")
            and hasattr(module, "use_res_connect")
            and hasattr(module, "stochastic_depth")
        ):
            module.forward = MethodType(forward, module)


def _adapt_first_conv(model: nn.Module, in_channels: int) -> None:
    if in_channels == 3:
        return
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                replacement = nn.Conv2d(
                    in_channels, child.out_channels, child.kernel_size, child.stride,
                    child.padding, child.dilation, child.groups, child.bias is not None,
                    child.padding_mode,
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(parent, name, replacement)
                return


def _alpha_stage_group(model_name: str, layer_name: str) -> str:
    """Architecture-defined stages, fixed independently of observed results."""
    key = _key(model_name)
    parts = layer_name.split(".")
    if key.startswith("resnet"):
        return parts[0] if parts[0].startswith("layer") else "stem"
    if key == "mobilenetv2" and parts[0] == "features":
        index = int(parts[1])
        stages = ((0, 0, "stem"), (1, 1, "stage_c16"), (2, 3, "stage_c24"),
                  (4, 6, "stage_c32"), (7, 10, "stage_c64"), (11, 13, "stage_c96"),
                  (14, 16, "stage_c160"), (17, 17, "stage_c320"), (18, 18, "head"))
        return next(label for low, high, label in stages if low <= index <= high)
    if key.startswith("efficientnet") and parts[0] == "features":
        return f"features_stage_{parts[1]}"
    if key in {"papercnn", "cnn", "cnn3"} and parts[0] == "features":
        return f"conv_stage_{int(parts[1]) // 4}"
    if key == "convnexttiny" and parts[0] == "features":
        return f"features_stage_{parts[1]}"
    return parts[0]


def apply_activation_sharing(model: nn.Module, model_name: str, mode: str = "site") -> nn.Module:
    """Tie alpha parameters only, before optimizer construction.

    A stage is an architecture stage rather than an individual residual
    block. Stem and head activations use their own stage groups. Sharing is
    restricted to the algebraic f4/Squareplus family; beta is never tied.
    All sites remain separate modules, but aliases refer to one Parameter.
    """
    if mode not in {"site", "stage", "network"}:
        raise ValueError("activation_sharing must be site, stage or network")
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, LGTSBase)]
    if mode != "site" and (not layers or any(
        not isinstance(layer, (AlgebraicBaseLGTS, AlgebraicLogisticLGTS))
        or not isinstance(layer.raw_alpha, nn.Parameter) for _, layer in layers
    )):
        raise ValueError("stage/network alpha sharing requires a learned-alpha f4/Squareplus candidate")
    representatives: dict[str, nn.Parameter] = {}
    groups: dict[str, list[str]] = {}
    for name, layer in layers:
        group = name if mode == "site" else ("network" if mode == "network" else _alpha_stage_group(model_name, name))
        if mode != "site":
            if group in representatives:
                layer.raw_alpha = representatives[group]
            else:
                representatives[group] = layer.raw_alpha
        layer.alpha_sharing_group = group
        groups.setdefault(group, []).append(name)
    model.activation_sharing = mode
    model.activation_parameter_groups = groups
    return model


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    activation: str,
    log_t: float = 0.6,
    lgts_learnable: bool = True,
    lgts_learn_alpha: bool | None = None,
    lgts_learn_beta: bool | None = None,
    lgts_alpha: float = 0.25,
    lgts_beta: float = 1.0,
    lgts_alpha_min: float = 0.05,
    lgts_alpha_max: float = 1.5,
    lgts_beta_min: float = 0.5,
    lgts_beta_max: float = 3.0,
    activation_scope: str = "full_backbone",
    pretrained: bool = True,
    small_input_stem: bool = False,
    activation_sharing: str = "site",
) -> nn.Module:
    key = _key(name)
    kwargs = _activation_kwargs(
        log_t, lgts_learnable, lgts_learn_alpha, lgts_learn_beta, lgts_alpha, lgts_beta,
        lgts_alpha_min, lgts_alpha_max, lgts_beta_min, lgts_beta_max,
    )
    if key in {"papercnn", "cnn", "cnn3"}:
        return apply_activation_sharing(PaperCNN(in_channels, num_classes, activation, kwargs), name, activation_sharing)
    if activation_scope not in {"last_stage", "full_backbone"}:
        raise ValueError("activation_scope must be last_stage or full_backbone; head-only is intentionally disabled")

    if key == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if small_input_stem:
            _adapt_resnet_small_input_stem(model, in_channels, pretrained)
        else:
            _adapt_first_conv(model, in_channels)
            model.stem_type = "imagenet_7x7_stride2_maxpool"
        _split_resnet18_activations(model, activation, kwargs, activation_scope)
        if not is_native_activation(activation):
            target = model if activation_scope == "full_backbone" else model.layer4
            _replace_activations(target, activation, kwargs)
        model.activation_site_policy = "independent_call_sites"
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    if key == "resnet50":
        if is_native_activation(activation):
            raise ValueError("Native ResNet50 is supported, but independent three-site Bottleneck replacement is not yet enabled; use ResNet18 for the confirmatory protocol.")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        _adapt_first_conv(model, in_channels)
        _replace_activations(model if activation_scope == "full_backbone" else model.layer4, activation, kwargs)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    if key == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        _adapt_first_conv(model, in_channels)
        if small_input_stem:
            _adapt_conv_stem_stride_one(model, "small_mobilenetv2_stride1")
        target = model.features if activation_scope == "full_backbone" else model.features[-2:]
        if not is_native_activation(activation):
            _replace_activations(target, activation, kwargs)
        if not small_input_stem:
            model.stem_type = "native_mobilenetv2_stride2"
        model.activation_site_policy = "independent_modules"
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    if key == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        _make_efficientnet_residuals_out_of_place(model)
        _adapt_first_conv(model, in_channels)
        if small_input_stem:
            _adapt_conv_stem_stride_one(model, "small_efficientnetb0_stride1")
        target = model.features if activation_scope == "full_backbone" else model.features[-2:]
        if not is_native_activation(activation):
            _replace_activations(target, activation, kwargs)
        if not small_input_stem:
            model.stem_type = "native_efficientnet_stride2"
        model.activation_site_policy = "independent_modules"
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    if key == "efficientnetv2s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        _make_efficientnet_residuals_out_of_place(model)
        _adapt_first_conv(model, in_channels)
        if small_input_stem:
            _adapt_conv_stem_stride_one(model, "small_efficientnetv2_stride1")
        target = model.features if activation_scope == "full_backbone" else model.features[-2:]
        if not is_native_activation(activation):
            _replace_activations(target, activation, kwargs)
        if not small_input_stem:
            model.stem_type = "native_efficientnetv2_stride2"
        model.activation_site_policy = "independent_modules"
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    if key == "convnexttiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        _adapt_first_conv(model, in_channels)
        target = model.features if activation_scope == "full_backbone" else model.features[-1]
        if not is_native_activation(activation):
            _replace_activations(target, activation, kwargs)
        model.stem_type = "native_convnext"
        model.activation_site_policy = "independent_modules"
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return apply_activation_sharing(model, name, activation_sharing)

    raise ValueError(
        f"Unknown model '{name}'. Available: PaperCNN, ResNet18, MobileNetV2, "
        "ResNet50, EfficientNetB0, EfficientNetV2S, ConvNeXtTiny"
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
