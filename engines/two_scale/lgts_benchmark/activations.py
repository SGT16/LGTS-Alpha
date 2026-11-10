from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


ACTIVATION_IMPLEMENTATION_VERSION = "major-revision-v1-fp32-stable-squareplus"


class _StableSquareplus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        root = torch.hypot(x, alpha)
        denominator = 0.5 * root - 0.5 * x.clamp_max(0.0)
        denominator = torch.where(denominator > 0, denominator, 1.0)
        negative = 0.25 * alpha * (alpha / denominator)
        positive = 0.5 * x + 0.5 * root
        ctx.save_for_backward(x, alpha, root)
        return torch.where(x < 0, negative, positive)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, alpha, root = ctx.saved_tensors
        safe_root = torch.where(root > 0, root, 1.0)
        ratio = x / safe_root
        alpha_ratio = alpha / safe_root
        negative_dx = 0.5 * alpha_ratio.square() / (1.0 - ratio.clamp_max(0.0))
        dx = torch.where(x < 0, negative_dx, 0.5 * (1.0 + ratio))
        dx = torch.where(root > 0, dx, 0.0)
        da = torch.where(root > 0, 0.5 * alpha_ratio, 0.5)
        return grad_output * dx, (grad_output * da).sum_to_size(alpha.shape)


def stable_squareplus(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        result = _StableSquareplus.apply(x.to(dtype=dtype), alpha.to(dtype=dtype))
    return result.to(dtype=x.dtype)


def stable_algebraic_logistic(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        work_x, work_alpha, work_beta = x.to(dtype), alpha.to(dtype), beta.to(dtype)
        result = _StableSquareplus.apply(work_x, work_alpha)
        result = result - work_alpha * torch.sigmoid(-work_beta * work_x)
    return result.to(dtype=x.dtype)


class PELU(nn.Module):

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            work_x, a, b = x.to(dtype), self.a.to(dtype), self.b.to(dtype)
            negative = a * torch.expm1(work_x.clamp_max(0.0) / b)
            result = torch.where(work_x >= 0, (a / b) * work_x, negative)
        return result.to(x.dtype)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.a.clamp_(min=0.1)
        self.b.clamp_(min=0.1)


class Swish(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualSafeReLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, x, 0.0)


def _channel_parameter(parameter: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if parameter.numel() == 1:
        return parameter
    if x.ndim < 2 or x.shape[1] != parameter.numel():
        raise ValueError(
            f"Channel-wise activation expected {parameter.numel()} channels, "
            f"but received shape {tuple(x.shape)}."
        )
    return parameter.view(1, -1, *([1] * (x.ndim - 2)))


class AconC(nn.Module):

    def __init__(self, channels: int | None = None) -> None:
        super().__init__()
        size = int(channels) if channels else 1
        self.p1 = nn.Parameter(torch.ones(size))
        self.p2 = nn.Parameter(torch.zeros(size))
        self.beta = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = _channel_parameter(self.p1, x)
        p2 = _channel_parameter(self.p2, x)
        beta = _channel_parameter(self.beta, x)
        difference = p1 - p2
        return difference * x * torch.sigmoid(beta * difference * x) + p2 * x


class QianGated(nn.Module):

    def __init__(self, channels: int | None = None, omega: float = 1.0) -> None:
        super().__init__()
        size = int(channels) if channels else 1
        self.omega = nn.Parameter(torch.full((size,), float(omega)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = _channel_parameter(self.omega, x)
        gate = torch.sigmoid(omega * x)
        lrelu = F.leaky_relu(x, negative_slope=0.01)
        elu = F.elu(x, alpha=1.0)
        return gate * lrelu + (1.0 - gate) * elu


def _inverse_sigmoid_range(value: float, low: float, high: float) -> torch.Tensor:
    if not low < high:
        raise ValueError(f"Invalid parameter range [{low}, {high}].")
    value = min(max(float(value), low + 1e-6), high - 1e-6)
    y = (value - low) / (high - low)
    return torch.tensor(math.log(y / (1.0 - y)), dtype=torch.float32)


class LGTSBase(nn.Module):

    variant = "LGTS"
    ablation_role = "legacy"
    formula = "unspecified"

    def __init__(
        self,
        alpha: float = 0.25,
        *,
        learnable: bool = True,
        learn_alpha: bool | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        raw_alpha = _inverse_sigmoid_range(alpha, self.alpha_min, self.alpha_max)
        alpha_is_learnable = learnable if learn_alpha is None else learn_alpha
        if alpha_is_learnable:
            self.raw_alpha = nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)

    @property
    def learning_mode(self) -> str:
        alpha_is_learned = isinstance(self.raw_alpha, nn.Parameter)
        beta_is_learned = any(
            isinstance(getattr(self, name, None), nn.Parameter)
            for name in ("raw_beta", "raw_beta_plus", "raw_beta_minus")
        )
        if alpha_is_learned and beta_is_learned:
            return "both"
        if alpha_is_learned:
            return "alpha"
        if beta_is_learned:
            return "beta"
        return "none"


class _SymmetricLGTSBase(LGTSBase):
    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 1.0,
        *,
        learnable: bool = True,
        learn_alpha: bool | None = None,
        learn_beta: bool | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 1.5,
        beta_min: float = 0.5,
        beta_max: float = 3.0,
    ) -> None:
        super().__init__(
            alpha, learnable=learnable, learn_alpha=learn_alpha,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        raw_beta = _inverse_sigmoid_range(beta, self.beta_min, self.beta_max)
        beta_is_learnable = learnable if learn_beta is None else learn_beta
        if beta_is_learnable:
            self.raw_beta = nn.Parameter(raw_beta)
        else:
            self.register_buffer("raw_beta", raw_beta)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_beta)


class LGTS(_SymmetricLGTSBase):

    variant = "LGTS-f1"
    formula = "x*sigmoid(beta*x)+alpha*tanh(x)*sigmoid(-beta*x)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.beta * x)
        negative = self.alpha * torch.tanh(x)
        return negative + gate * (x - negative)


class SlopeMatchedLGTS(_SymmetricLGTSBase):

    variant = "LGTS-f2-Slope"
    formula = "x*sigmoid(beta*x)+alpha*tanh(x/alpha)*sigmoid(-beta*x)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.beta * x)
        negative = self.alpha * torch.tanh(x / self.alpha)
        return negative + gate * (x - negative)


class AsymmetricDualGateLGTS(LGTSBase):

    variant = "LGTS-f3-Dual"
    formula = "x*sigmoid(beta_plus*x)+alpha*tanh(x/alpha)*sigmoid(-beta_minus*x)"

    def __init__(
        self,
        alpha: float = 0.25,
        beta_plus: float = 1.0,
        beta_minus: float = 1.0,
        *,
        learnable: bool = True,
        learn_alpha: bool | None = None,
        learn_beta: bool | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 1.5,
        beta_min: float = 0.5,
        beta_max: float = 3.0,
    ) -> None:
        super().__init__(
            alpha, learnable=learnable, learn_alpha=learn_alpha,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        raw_plus = _inverse_sigmoid_range(beta_plus, self.beta_min, self.beta_max)
        raw_minus = _inverse_sigmoid_range(beta_minus, self.beta_min, self.beta_max)
        beta_is_learnable = learnable if learn_beta is None else learn_beta
        if beta_is_learnable:
            self.raw_beta_plus = nn.Parameter(raw_plus)
            self.raw_beta_minus = nn.Parameter(raw_minus)
        else:
            self.register_buffer("raw_beta_plus", raw_plus)
            self.register_buffer("raw_beta_minus", raw_minus)

    @property
    def beta_plus(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_beta_plus)

    @property
    def beta_minus(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_beta_minus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive_gate = torch.sigmoid(self.beta_plus * x)
        negative_gate = torch.sigmoid(-self.beta_minus * x)
        return positive_gate * x + negative_gate * self.alpha * torch.tanh(x / self.alpha)


class AlgebraicBaseLGTS(LGTSBase):

    variant = "LGTS-f4-Base-Squareplus"
    ablation_role = "base_squareplus_fixed"
    formula = "0.5*(x+sqrt(x^2+alpha^2))"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stable_squareplus(x, self.alpha)


class LearnedAlgebraicBaseLGTS(AlgebraicBaseLGTS):

    variant = "LGTS-f4-Base-Squareplus-Learned"
    ablation_role = "base_squareplus_learned_alpha_b_equals_alpha_squared"


class AlgebraicLogisticLGTS(_SymmetricLGTSBase):

    variant = "LGTS-f4-Full"
    ablation_role = "algebraic_plus_logistic_both_learned"
    formula = "0.5*(x+sqrt(x^2+alpha^2))-alpha*sigmoid(-beta*x)"

    def __init__(self, *args, variant: str | None = None, ablation_role: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if variant is not None:
            self.variant = variant
        if ablation_role is not None:
            self.ablation_role = ablation_role

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stable_algebraic_logistic(x, self.alpha, self.beta)


class FixedSplitAlgebraicLGTS(AlgebraicLogisticLGTS):

    variant = "LGTS-f4-SplitFixed"
    ablation_role = "fixed_anchored_width_and_tail_depth"
    formula = "Squareplus(x,s)-t*sigmoid(log(s/(2*t-s))-beta*x)"

    def __init__(self, width, depth, beta, *, alpha_min, alpha_max, beta_min, beta_max):
        if depth is None or not (0 < width < 2 * depth and beta > 0):
            raise ValueError("SplitFixed requires 0 < width < 2*depth and beta > 0")
        if not alpha_min < width < alpha_max or not alpha_min < depth < alpha_max:
            raise ValueError("Width and depth must lie inside the original encoding range")
        super().__init__(alpha=width, beta=beta, learnable=False, learn_alpha=False, learn_beta=False,
                         alpha_min=alpha_min, alpha_max=alpha_max, beta_min=beta_min, beta_max=beta_max)
        self.same_levels = width == depth
        self.register_buffer("raw_depth", _inverse_sigmoid_range(depth, alpha_min, alpha_max))

    @property
    def depth(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_depth)

    def forward(self, x):
        if self.same_levels:
            return super().forward(x)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_work = x.to(dtype)
            width, depth, beta = self.alpha.to(dtype), self.depth.to(dtype), self.beta.to(dtype)
            shift = torch.log(width / (2 * depth - width))
            result = _StableSquareplus.apply(x_work, width) - depth * torch.sigmoid(shift - beta * x_work)
        return result.to(x.dtype)


class ZeroBoundAlgebraicLogisticLGTS(AlgebraicLogisticLGTS):

    def __init__(self, alpha: float = 0.4, **kwargs) -> None:
        if kwargs.get("alpha_min", 0.0) != 0.0:
            raise ValueError("LGTS-f4-AlphaZero requires alpha_min=0")
        kwargs["alpha_min"] = 0.0
        super().__init__(
            alpha=alpha, variant="LGTS-f4-AlphaZero",
            ablation_role="alpha_zero_bound_direct_projected", **kwargs,
        )
        if not 0.0 <= alpha <= self.alpha_max:
            raise ValueError("Projected alpha initialization must be in [0, alpha_max]")
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.raw_alpha.clamp_(min=0.0, max=self.alpha_max)


@torch.no_grad()
def project_activation_parameters_(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, (PELU, ZeroBoundAlgebraicLogisticLGTS)):
            layer.project_parameters_()


class LogMeanActivation(nn.Module):

    def __init__(self, left: Callable[[torch.Tensor], torch.Tensor], right: Callable[[torch.Tensor], torch.Tensor], t: float = 0.6):
        super().__init__()
        if not 0.0 < t < 1.0:
            raise ValueError("t must be strictly between 0 and 1")
        self.left = left
        self.right = right
        self.register_buffer("log_t", torch.tensor(math.log(t), dtype=torch.float32))
        self.register_buffer("log_one_minus_t", torch.tensor(math.log(1.0 - t), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.left(x) + self.log_t.to(dtype=x.dtype)
        b = self.right(x) + self.log_one_minus_t.to(dtype=x.dtype)
        return torch.logsumexp(torch.stack((a, b), dim=0), dim=0)


def _activation_key(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def canonical_activation_name(name: str) -> str:
    key = _activation_key(name)
    aliases = {
        "lgts": "LGTS-f1",
        "lgtsf1": "LGTS-f1",
        "originallgts": "LGTS-f1",
        "lgtssymmetric": "LGTS-f2-Slope",
        "lgtsslope": "LGTS-f2-Slope",
        "lgtsf2": "LGTS-f2-Slope",
        "lgtsf2slope": "LGTS-f2-Slope",
        "slopematchedlgts": "LGTS-f2-Slope",
        "lgtsdual": "LGTS-f3-Dual",
        "lgtsf3": "LGTS-f3-Dual",
        "lgtsf3dual": "LGTS-f3-Dual",
        "asymmetricdualgatelgts": "LGTS-f3-Dual",
        "lgtsf4": "LGTS-f4-Full",
        "lgtsf4full": "LGTS-f4-Full",
        "algebraiclogisticlgts": "LGTS-f4-Full",
        "f4full": "LGTS-f4-Full",
        "lgtsf4base": "LGTS-f4-Base-Squareplus",
        "lgtsf4basesquareplus": "LGTS-f4-Base-Squareplus",
        "f4base": "LGTS-f4-Base-Squareplus",
        "squarepluscontrol": "LGTS-f4-Base-Squareplus",
        "squarepluslearned": "LGTS-f4-Base-Squareplus-Learned",
        "lgtsf4basesquarepluslearned": "LGTS-f4-Base-Squareplus-Learned",
        "lgtsf4fixed": "LGTS-f4-Fixed",
        "lgtsf4splitfixed": "LGTS-f4-SplitFixed",
        "f4fixed": "LGTS-f4-Fixed",
        "lgtsf4alpha": "LGTS-f4-Alpha",
        "f4alpha": "LGTS-f4-Alpha",
        "lgtsf4alphazero": "LGTS-f4-AlphaZero",
        "f4alphazero": "LGTS-f4-AlphaZero",
        "lgtsf4beta": "LGTS-f4-Beta",
        "f4beta": "LGTS-f4-Beta",
    }
    return aliases.get(key, name)


def is_lgts_activation(name: str) -> bool:
    return canonical_activation_name(name).startswith("LGTS-f")


def make_activation(
    name: str,
    *,
    channels: int | None = None,
    log_t: float = 0.6,
    lgts_learnable: bool = True,
    lgts_learn_alpha: bool | None = None,
    lgts_learn_beta: bool | None = None,
    lgts_alpha: float = 0.25,
    split_depth: float | None = None,
    lgts_beta: float = 1.0,
    lgts_alpha_min: float = 0.05,
    lgts_alpha_max: float = 1.5,
    lgts_beta_min: float = 0.5,
    lgts_beta_max: float = 3.0,
) -> nn.Module:
    key = _activation_key(canonical_activation_name(name))
    common = dict(
        alpha=lgts_alpha,
        learnable=lgts_learnable,
        learn_alpha=lgts_learn_alpha,
        learn_beta=lgts_learn_beta,
        alpha_min=lgts_alpha_min,
        alpha_max=lgts_alpha_max,
        beta_min=lgts_beta_min,
        beta_max=lgts_beta_max,
    )
    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=False),
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
        "prelu": lambda: nn.PReLU(num_parameters=1),
        "preluscalar": lambda: nn.PReLU(num_parameters=1),
        "preluchannel": lambda: nn.PReLU(num_parameters=channels or 1),
        "elu": lambda: nn.ELU(inplace=False),
        "celu": lambda: nn.CELU(alpha=1.0, inplace=False),
        "pelu": PELU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": Swish,
        "silu": nn.SiLU,
        "mish": lambda: nn.Mish(inplace=False),
        "softplus": nn.Softplus,
        "acon": lambda: AconC(channels),
        "aconc": lambda: AconC(channels),
        "qian": lambda: QianGated(channels),
        "qiangated": lambda: QianGated(channels),
        "lgts": lambda: LGTS(beta=lgts_beta, **common),
        "lgtsf1": lambda: LGTS(beta=lgts_beta, **common),
        "originallgts": lambda: LGTS(beta=lgts_beta, **common),
        "lgtssymmetric": lambda: SlopeMatchedLGTS(beta=lgts_beta, **common),
        "lgtsslope": lambda: SlopeMatchedLGTS(beta=lgts_beta, **common),
        "lgtsf2": lambda: SlopeMatchedLGTS(beta=lgts_beta, **common),
        "lgtsf2slope": lambda: SlopeMatchedLGTS(beta=lgts_beta, **common),
        "slopematchedlgts": lambda: SlopeMatchedLGTS(beta=lgts_beta, **common),
        "lgtsdual": lambda: AsymmetricDualGateLGTS(beta_plus=lgts_beta, beta_minus=lgts_beta, **common),
        "lgtsf3": lambda: AsymmetricDualGateLGTS(beta_plus=lgts_beta, beta_minus=lgts_beta, **common),
        "lgtsf3dual": lambda: AsymmetricDualGateLGTS(beta_plus=lgts_beta, beta_minus=lgts_beta, **common),
        "asymmetricdualgatelgts": lambda: AsymmetricDualGateLGTS(beta_plus=lgts_beta, beta_minus=lgts_beta, **common),
        "lgtsf4basesquareplus": lambda: AlgebraicBaseLGTS(
            alpha=lgts_alpha, learnable=False, learn_alpha=False,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
        ),
        "lgtsf4basesquarepluslearned": lambda: LearnedAlgebraicBaseLGTS(
            alpha=lgts_alpha, learnable=True, learn_alpha=True,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
        ),
        "lgtsf4fixed": lambda: AlgebraicLogisticLGTS(
            alpha=lgts_alpha, beta=lgts_beta, learnable=False,
            learn_alpha=False, learn_beta=False,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
            variant="LGTS-f4-Fixed", ablation_role="algebraic_plus_logistic_fixed",
        ),
        "lgtsf4splitfixed": lambda: FixedSplitAlgebraicLGTS(
            lgts_alpha, split_depth, lgts_beta, alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
        ),
        "lgtsf4alpha": lambda: AlgebraicLogisticLGTS(
            alpha=lgts_alpha, beta=lgts_beta, learnable=True,
            learn_alpha=True, learn_beta=False,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
            variant="LGTS-f4-Alpha", ablation_role="algebraic_plus_logistic_alpha_learned",
        ),
        "lgtsf4alphazero": lambda: ZeroBoundAlgebraicLogisticLGTS(
            alpha=lgts_alpha, beta=lgts_beta, learnable=True,
            learn_alpha=True, learn_beta=False,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
        ),
        "lgtsf4beta": lambda: AlgebraicLogisticLGTS(
            alpha=lgts_alpha, beta=lgts_beta, learnable=True,
            learn_alpha=False, learn_beta=True,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
            variant="LGTS-f4-Beta", ablation_role="algebraic_plus_logistic_beta_learned",
        ),
        "lgtsf4full": lambda: AlgebraicLogisticLGTS(
            alpha=lgts_alpha, beta=lgts_beta, learnable=True,
            learn_alpha=True, learn_beta=True,
            alpha_min=lgts_alpha_min, alpha_max=lgts_alpha_max,
            beta_min=lgts_beta_min, beta_max=lgts_beta_max,
            variant="LGTS-f4-Full", ablation_role="algebraic_plus_logistic_both_learned",
        ),
        "ltr": lambda: LogMeanActivation(torch.tanh, F.relu, t=log_t),
        "ltl": lambda: LogMeanActivation(torch.tanh, lambda x: F.leaky_relu(x, 0.01), t=log_t),
        "lil": lambda: LogMeanActivation(lambda x: x, lambda x: F.leaky_relu(x, 0.01), t=log_t),
    }
    if key not in factories:
        raise ValueError(f"Unknown activation '{name}'. Available keys: {', '.join(sorted(factories))}")
    return factories[key]()


def _sample_sd(values: torch.Tensor) -> float:
    return float(values.std(unbiased=True)) if values.numel() > 1 else 0.0


def _stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_mean": None, f"{prefix}_sd": None, f"{prefix}_min": None, f"{prefix}_max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        f"{prefix}_mean": float(tensor.mean()),
        f"{prefix}_sd": _sample_sd(tensor),
        f"{prefix}_min": float(tensor.min()),
        f"{prefix}_max": float(tensor.max()),
    }


def lgts_parameter_rows(model: nn.Module) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for layer_name, layer in model.named_modules():
        if not isinstance(layer, LGTSBase):
            continue
        row: dict[str, float | str] = {
            "layer": layer_name,
            "variant": layer.variant,
            "ablation_role": layer.ablation_role,
            "formula": layer.formula,
            "learning_mode": layer.learning_mode,
            "alpha": float(layer.alpha.detach().cpu()),
            "alpha_parameterization": (
                "direct_projected" if isinstance(layer, ZeroBoundAlgebraicLogisticLGTS)
                else "bounded_sigmoid"
            ),
            "alpha_min": layer.alpha_min,
            "alpha_max": layer.alpha_max,
            "sharing_group": getattr(layer, "alpha_sharing_group", layer_name),
            "numerical_implementation": ACTIVATION_IMPLEMENTATION_VERSION,
        }
        if isinstance(layer, AlgebraicBaseLGTS):
            row["squareplus_b"] = float(layer.alpha.detach().square().cpu())
        if isinstance(layer, FixedSplitAlgebraicLGTS):
            row["split_width"] = row["alpha"]
            row["split_depth"] = float(layer.depth.detach().cpu())
            row["numerical_implementation"] = "fixed_split_anchored_stable_squareplus_v1"
        if isinstance(layer, _SymmetricLGTSBase):
            row["beta"] = float(layer.beta.detach().cpu())
        elif isinstance(layer, AsymmetricDualGateLGTS):
            beta_plus = float(layer.beta_plus.detach().cpu())
            beta_minus = float(layer.beta_minus.detach().cpu())
            row.update(
                beta_plus=beta_plus,
                beta_minus=beta_minus,
                beta_gap=beta_plus - beta_minus,
                beta_ratio=beta_plus / beta_minus,
            )
        rows.append(row)
    return rows


def lgts_parameter_stats(model: nn.Module) -> dict[str, float | int | str | None]:
    rows = lgts_parameter_rows(model)
    result: dict[str, float | int | str | None] = {
        "variant": rows[0]["variant"] if rows else None,
        "ablation_role": rows[0]["ablation_role"] if rows else None,
        "formula": rows[0]["formula"] if rows else None,
        "learning_mode": rows[0]["learning_mode"] if rows else None,
        "count": len(rows),
    }
    result.update(_stats([float(row["alpha"]) for row in rows], "alpha"))
    result.update(_stats([float(row["squareplus_b"]) for row in rows if "squareplus_b" in row], "squareplus_b"))
    result.update(_stats([float(row["beta"]) for row in rows if "beta" in row], "beta"))
    result.update(_stats([float(row["beta_plus"]) for row in rows if "beta_plus" in row], "beta_plus"))
    result.update(_stats([float(row["beta_minus"]) for row in rows if "beta_minus" in row], "beta_minus"))
    result.update(_stats([float(row["beta_gap"]) for row in rows if "beta_gap" in row], "beta_gap"))
    result.update(_stats([float(row["beta_ratio"]) for row in rows if "beta_ratio" in row], "beta_ratio"))
    lgts_layers = [layer for layer in model.modules() if isinstance(layer, LGTSBase)]
    near_lower = [
        float(layer.alpha.detach().cpu()) <= layer.alpha_min + 0.005
        for layer in lgts_layers
    ]
    result["alpha_near_lower_fraction"] = float(sum(near_lower) / len(near_lower)) if near_lower else None
    result["alpha_unique_parameter_count"] = len({
        id(layer.raw_alpha) for layer in lgts_layers if isinstance(layer.raw_alpha, nn.Parameter)
    })
    result["alpha_sharing"] = getattr(model, "activation_sharing", "site")
    result["numerical_implementation"] = ACTIVATION_IMPLEMENTATION_VERSION
    return result


def lgts_parameter_snapshot(model: nn.Module) -> dict[str, float | None]:
    stats = lgts_parameter_stats(model)
    return {
        "lgts_alpha": stats["alpha_mean"],
        "lgts_beta": stats["beta_mean"],
        "lgts_beta_plus": stats["beta_plus_mean"],
        "lgts_beta_minus": stats["beta_minus_mean"],
        "lgts_beta_gap": stats["beta_gap_mean"],
        "lgts_beta_ratio": stats["beta_ratio_mean"],
    }


def adaptive_activation_parameter_rows(model: nn.Module) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, AconC):
            rows.append({
                "layer": layer_name, "activation": "ACON-C", "channels": layer.p1.numel(),
                "p1_mean": float(layer.p1.detach().mean().cpu()),
                "p2_mean": float(layer.p2.detach().mean().cpu()),
                "beta_mean": float(layer.beta.detach().mean().cpu()),
            })
        elif isinstance(layer, QianGated):
            rows.append({
                "layer": layer_name, "activation": "Qian-Gated", "channels": layer.omega.numel(),
                "omega_mean": float(layer.omega.detach().mean().cpu()),
            })
        elif isinstance(layer, PELU):
            rows.append({
                "layer": layer_name, "activation": "PELU", "channels": 1,
                "a": float(layer.a.detach().cpu()), "b": float(layer.b.detach().cpu()),
                "constraint": "projected_min_0.1", "initial_a": 1.0, "initial_b": 1.0,
            })
        elif isinstance(layer, nn.PReLU):
            rows.append({
                "layer": layer_name, "activation": "PReLU", "channels": layer.weight.numel(),
                "negative_slope_mean": float(layer.weight.detach().mean().cpu()),
            })
    return rows


def adaptive_activation_parameter_stats(model: nn.Module) -> dict[str, float | int | None]:
    acon_layers = [layer for layer in model.modules() if isinstance(layer, AconC)]
    qian_layers = [layer for layer in model.modules() if isinstance(layer, QianGated)]
    pelu_layers = [layer for layer in model.modules() if isinstance(layer, PELU)]
    prelu_layers = [layer for layer in model.modules() if isinstance(layer, nn.PReLU)]
    result: dict[str, float | int | None] = {
        "adaptive_layer_count": len(acon_layers) + len(qian_layers) + len(pelu_layers) + len(prelu_layers),
    }
    for prefix, values in (
        ("acon_p1", [float(value) for layer in acon_layers for value in layer.p1.detach().cpu().flatten()]),
        ("acon_p2", [float(value) for layer in acon_layers for value in layer.p2.detach().cpu().flatten()]),
        ("acon_beta", [float(value) for layer in acon_layers for value in layer.beta.detach().cpu().flatten()]),
        ("qian_omega", [float(value) for layer in qian_layers for value in layer.omega.detach().cpu().flatten()]),
        ("pelu_a", [float(layer.a.detach().cpu()) for layer in pelu_layers]),
        ("pelu_b", [float(layer.b.detach().cpu()) for layer in pelu_layers]),
        ("prelu_slope", [float(value) for layer in prelu_layers for value in layer.weight.detach().cpu().flatten()]),
    ):
        result.update(_stats(values, prefix))
    return result
