from __future__ import annotations

from types import MethodType

import torch
from torch import nn
from torchvision import models

from .activations import make_activation
from .models import (
    _activation_kwargs,
    _adapt_first_conv,
    _key,
    _replace_activations,
    build_model as build_base_model,
    is_native_activation,
    native_activation_variant as base_native_activation_variant,
)


def native_activation_variant(model_name: str) -> str:
    if _key(model_name) == "densenet121":
        return "Native-ReLU"
    return base_native_activation_variant(model_name)


def _adapt_densenet_small_input_stem(
    model: nn.Module,
    in_channels: int,
    pretrained: bool,
) -> None:
    old = model.features.conv0
    replacement = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            weight = old.weight
            if in_channels != old.in_channels:
                weight = weight.mean(dim=1, keepdim=True).repeat(
                    1, in_channels, 1, 1,
                )
            replacement.weight.copy_(weight[:, :, 2:5, 2:5])
        else:
            nn.init.kaiming_normal_(
                replacement.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
    model.features.conv0 = replacement
    model.features.pool0 = nn.Identity()
    model.stem_type = "small_densenet_3x3_stride1_no_maxpool"


def _make_final_activation_explicit(
    model: nn.Module,
    activation: str,
    activation_kwargs: dict,
) -> None:
    channels = int(model.classifier.in_features)
    model.final_activation = (
        nn.ReLU(inplace=False)
        if is_native_activation(activation)
        else make_activation(
            activation,
            channels=channels,
            **activation_kwargs,
        )
    )

    def forward(densenet, x):
        features = densenet.features(x)
        out = densenet.final_activation(features)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return densenet.classifier(out)

    model.forward = MethodType(forward, model)


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
) -> nn.Module:
    if _key(name) != "densenet121":
        return build_base_model(
            name,
            in_channels,
            num_classes,
            activation,
            log_t=log_t,
            lgts_learnable=lgts_learnable,
            lgts_learn_alpha=lgts_learn_alpha,
            lgts_learn_beta=lgts_learn_beta,
            lgts_alpha=lgts_alpha,
            lgts_beta=lgts_beta,
            lgts_alpha_min=lgts_alpha_min,
            lgts_alpha_max=lgts_alpha_max,
            lgts_beta_min=lgts_beta_min,
            lgts_beta_max=lgts_beta_max,
            activation_scope=activation_scope,
            pretrained=pretrained,
            small_input_stem=small_input_stem,
        )
    if activation_scope not in {"last_stage", "full_backbone"}:
        raise ValueError(
            "activation_scope must be last_stage or full_backbone; "
            "head-only is intentionally disabled"
        )

    kwargs = _activation_kwargs(
        log_t,
        lgts_learnable,
        lgts_learn_alpha,
        lgts_learn_beta,
        lgts_alpha,
        lgts_beta,
        lgts_alpha_min,
        lgts_alpha_max,
        lgts_beta_min,
        lgts_beta_max,
    )
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    if small_input_stem:
        _adapt_densenet_small_input_stem(model, in_channels, pretrained)
    else:
        _adapt_first_conv(model, in_channels)
        model.stem_type = "imagenet_7x7_stride2_maxpool"

    target = (
        model.features
        if activation_scope == "full_backbone"
        else model.features.denseblock4
    )
    if not is_native_activation(activation):
        _replace_activations(target, activation, kwargs)
    _make_final_activation_explicit(model, activation, kwargs)
    model.activation_site_policy = (
        "independent_modules_plus_explicit_final"
    )
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
