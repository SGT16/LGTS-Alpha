import unittest

import torch
from torch import nn

from lgts_benchmark.activations import (
    AconC,
    AlgebraicBaseLGTS,
    AlgebraicLogisticLGTS,
    AsymmetricDualGateLGTS,
    LGTS,
    QianGated,
    ResidualSafeReLU,
    SlopeMatchedLGTS,
    Swish,
    lgts_parameter_stats,
    make_activation,
)
from lgts_benchmark.models import SplitActivationBasicBlock, build_model, native_activation_variant


class ActivationFormulaTests(unittest.TestCase):
    @staticmethod
    def derivative_at_zero(module: nn.Module) -> float:
        x = torch.tensor(0.0, requires_grad=True)
        y = module(x)
        return float(torch.autograd.grad(y, x)[0])

    def test_zero_value_and_slopes(self):
        f1 = LGTS(alpha=0.25, learnable=False)
        f2 = SlopeMatchedLGTS(alpha=0.25, learnable=False)
        f3 = AsymmetricDualGateLGTS(alpha=0.25, learnable=False)
        for module in (f1, f2, f3):
            self.assertAlmostEqual(float(module(torch.tensor(0.0))), 0.0, places=7)
        self.assertAlmostEqual(self.derivative_at_zero(f1), 0.625, places=6)
        self.assertAlmostEqual(self.derivative_at_zero(f2), 1.0, places=6)
        self.assertAlmostEqual(self.derivative_at_zero(f3), 1.0, places=6)

        f4 = AlgebraicLogisticLGTS(alpha=0.4, beta=1.0, learnable=False)
        self.assertAlmostEqual(float(f4(torch.tensor(0.0))), 0.0, places=7)
        self.assertAlmostEqual(self.derivative_at_zero(f4), 0.6, places=6)

    def test_negative_tail_is_asymptote(self):
        for cls in (LGTS, SlopeMatchedLGTS, AsymmetricDualGateLGTS):
            module = cls(alpha=0.25, learnable=False)
            self.assertAlmostEqual(float(module(torch.tensor(-100.0))), -0.25, places=5)
        f4 = AlgebraicLogisticLGTS(alpha=0.25, learnable=False)
        self.assertAlmostEqual(float(f4(torch.tensor(-100_000.0))), -0.25, places=5)

    def test_f4_base_matches_squareplus_reference(self):
        x = torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64)
        module = AlgebraicBaseLGTS(alpha=0.4, learnable=False).double()
        reference = 0.5 * (x + torch.sqrt(x.square() + module.alpha.square()))
        self.assertTrue(torch.allclose(module(x), reference, atol=2e-14, rtol=2e-12))

    def test_f4_fast_form_matches_stable_branch_on_training_range(self):
        x_fast = torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64, requires_grad=True)
        x_reference = x_fast.detach().clone().requires_grad_(True)
        module = AlgebraicLogisticLGTS(alpha=0.4, beta=1.0, learnable=False).double()
        alpha, beta = module.alpha, module.beta
        root = torch.sqrt(x_reference.square() + alpha.square())
        stable_base = 0.5 * torch.where(
            x_reference >= 0,
            x_reference + root,
            alpha.square() / (root - x_reference),
        )
        reference = stable_base - alpha * torch.sigmoid(-beta * x_reference)
        optimized = module(x_fast)
        self.assertTrue(torch.allclose(optimized, reference, atol=2e-14, rtol=2e-12))
        optimized.sum().backward()
        reference.sum().backward()
        self.assertTrue(torch.allclose(x_fast.grad, x_reference.grad, atol=2e-13, rtol=2e-11))

    def test_f4_is_strictly_monotone_across_parameter_bounds(self):
        for alpha, beta in ((0.05, 0.5), (0.4, 1.0), (1.49, 2.99)):
            x = torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64, requires_grad=True)
            module = AlgebraicLogisticLGTS(
                alpha=alpha, beta=beta, learnable=False,
            ).double()
            derivative = torch.autograd.grad(module(x).sum(), x)[0]
            self.assertGreater(float(derivative.min()), 0.0)

    def test_f4_named_ablation_learning_modes(self):
        expected = {
            "LGTS-f4-Base-Squareplus": (AlgebraicBaseLGTS, "none"),
            "LGTS-f4-Fixed": (AlgebraicLogisticLGTS, "none"),
            "LGTS-f4-Alpha": (AlgebraicLogisticLGTS, "alpha"),
            "LGTS-f4-Beta": (AlgebraicLogisticLGTS, "beta"),
            "LGTS-f4-Full": (AlgebraicLogisticLGTS, "both"),
        }
        for name, (kind, learning_mode) in expected.items():
            with self.subTest(name=name):
                module = make_activation(name)
                self.assertIsInstance(module, kind)
                self.assertEqual(module.learning_mode, learning_mode)

    def test_f4_tends_to_relu_as_alpha_tends_to_zero(self):
        x = torch.linspace(-5.0, 5.0, 2001, dtype=torch.float64)
        module = AlgebraicLogisticLGTS(
            alpha=1e-4, beta=1.0, learnable=False,
            alpha_min=1e-8, alpha_max=1.5,
        ).double()
        self.assertLess(float((module(x) - torch.relu(x)).abs().max()), 1.1e-4)

    def test_f3_starts_as_f2_when_gates_are_tied(self):
        x = torch.linspace(-8.0, 8.0, 2001)
        f2 = SlopeMatchedLGTS(alpha=0.4, beta=1.0, learnable=False)
        f3 = AsymmetricDualGateLGTS(alpha=0.4, beta_plus=1.0, beta_minus=1.0, learnable=False)
        self.assertTrue(torch.allclose(f2(x), f3(x), atol=1e-7, rtol=1e-6))

    def test_optimized_f1_and_f2_match_two_sigmoid_reference(self):
        for module, slope_matched in (
            (LGTS(alpha=0.4, beta=1.3, learnable=True), False),
            (SlopeMatchedLGTS(alpha=0.4, beta=1.3, learnable=True), True),
        ):
            x_fast = torch.linspace(-8.0, 8.0, 2001, requires_grad=True)
            x_reference = x_fast.detach().clone().requires_grad_(True)
            alpha, beta = module.alpha, module.beta
            negative = alpha * torch.tanh(
                x_reference / alpha if slope_matched else x_reference
            )
            reference = (
                x_reference * torch.sigmoid(beta * x_reference)
                + negative * torch.sigmoid(-beta * x_reference)
            )
            optimized = module(x_fast)
            self.assertTrue(torch.allclose(optimized, reference, atol=2e-6, rtol=2e-6))
            optimized.sum().backward()
            reference.sum().backward()
            self.assertTrue(torch.allclose(x_fast.grad, x_reference.grad, atol=2e-6, rtol=2e-6))

    def test_learned_parameters_remain_in_configured_ranges(self):
        module = AsymmetricDualGateLGTS(alpha=0.25, learnable=True)
        with torch.no_grad():
            module.raw_alpha.fill_(100.0)
            module.raw_beta_plus.fill_(-100.0)
            module.raw_beta_minus.fill_(100.0)
        self.assertLessEqual(float(module.alpha.detach()), 1.5)
        self.assertGreaterEqual(float(module.beta_plus.detach()), 0.5)
        self.assertLessEqual(float(module.beta_minus.detach()), 3.0)

    def test_alpha_only_and_beta_only_learning_modes(self):
        alpha_only = SlopeMatchedLGTS(learn_alpha=True, learn_beta=False)
        beta_only = SlopeMatchedLGTS(learn_alpha=False, learn_beta=True)
        self.assertEqual(set(dict(alpha_only.named_parameters())), {"raw_alpha"})
        self.assertEqual(set(dict(beta_only.named_parameters())), {"raw_beta"})

    def test_acon_c_initialization_equals_swish(self):
        x = torch.linspace(-5.0, 5.0, 101).repeat(2, 3, 1, 1)
        self.assertTrue(torch.allclose(AconC(3)(x), Swish()(x), atol=1e-7))

    def test_qian_gated_preserves_shape_and_learns_channel_gates(self):
        x = torch.randn(2, 4, 8, 8)
        module = QianGated(4)
        self.assertEqual(module(x).shape, x.shape)
        self.assertEqual(module.omega.numel(), 4)

    def test_residual_safe_relu_matches_relu_values_and_gradients(self):
        x_safe = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        x_reference = x_safe.detach().clone().requires_grad_(True)
        y_safe = ResidualSafeReLU()(x_safe)
        y_reference = torch.relu(x_reference)
        self.assertTrue(torch.equal(y_safe, y_reference))
        y_safe.sum().backward()
        y_reference.sum().backward()
        self.assertTrue(torch.equal(x_safe.grad, x_reference.grad))

    def test_benchmark_relu_is_native_and_prelu_modes_are_explicit(self):
        self.assertIsInstance(make_activation("ReLU"), nn.ReLU)
        self.assertEqual(make_activation("PReLU-Scalar", channels=8).num_parameters, 1)
        self.assertEqual(make_activation("PReLU-Channel", channels=8).num_parameters, 8)

    def test_locked_cli_activation_names_are_directly_supported(self):
        self.assertIsInstance(make_activation("LGTS-f2-Slope"), SlopeMatchedLGTS)
        self.assertIsInstance(make_activation("LGTS-f3-Dual"), AsymmetricDualGateLGTS)
        self.assertIsInstance(make_activation("LGTS-f4"), AlgebraicLogisticLGTS)
        self.assertIsInstance(make_activation("LGTS-f4-Base-Squareplus"), AlgebraicBaseLGTS)


class FullBackboneTests(unittest.TestCase):
    def test_resnet_head_is_linear_and_backbone_is_replaced(self):
        model = build_model("ResNet18", 3, 9, "LGTS-f3-Dual", pretrained=False, small_input_stem=True)
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertFalse(any(isinstance(module, nn.ReLU) for module in model.modules()))
        stats = lgts_parameter_stats(model)
        self.assertEqual(int(stats["count"]), 17)
        self.assertAlmostEqual(float(stats["beta_gap_mean"]), 0.0, places=6)
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, nn.Identity)
        blocks = [module for module in model.modules() if isinstance(module, SplitActivationBasicBlock)]
        self.assertEqual(len(blocks), 8)
        self.assertTrue(all(block.act1 is not block.act2 for block in blocks))

    def test_native_controls_are_model_specific(self):
        self.assertEqual(native_activation_variant("ResNet18"), "Native-ReLU")
        self.assertEqual(native_activation_variant("MobileNetV2"), "Native-ReLU6")
        self.assertEqual(native_activation_variant("EfficientNetV2S"), "Native-SiLU")
        mobile = build_model("MobileNetV2", 3, 10, "Native", pretrained=False, small_input_stem=True)
        self.assertTrue(any(isinstance(module, nn.ReLU6) for module in mobile.modules()))
        first_mobile_conv = next(module for module in mobile.modules() if isinstance(module, nn.Conv2d))
        self.assertEqual(first_mobile_conv.stride, (1, 1))
        self.assertEqual(mobile.stem_type, "small_mobilenetv2_stride1")

    def test_efficientnet_v2_s_head_is_linear_and_silu_is_replaced(self):
        model = build_model("EfficientNetV2S", 3, 9, "LGTS-f2-Slope", pretrained=False, small_input_stem=True)
        self.assertIsInstance(model.classifier[-1], nn.Linear)
        self.assertFalse(any(isinstance(module, nn.SiLU) for module in model.features.modules()))
        self.assertGreater(int(lgts_parameter_stats(model)["count"]), 0)
        first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
        self.assertEqual(first_conv.stride, (1, 1))
        self.assertEqual(model.stem_type, "small_efficientnetv2_stride1")

    def test_acon_c_is_channel_wise_in_resnet(self):
        model = build_model("ResNet18", 3, 10, "ACON-C", pretrained=False)
        acon_layers = [module for module in model.modules() if isinstance(module, AconC)]
        self.assertGreater(len(acon_layers), 0)
        self.assertTrue(all(module.p1.numel() > 1 for module in acon_layers))

    def test_efficientnet_v2_s_channel_wise_activations_handle_se_blocks(self):
        x = torch.randn(2, 3, 32, 32)
        for activation in ("Qian-Gated", "ACON-C"):
            with self.subTest(activation=activation):
                model = build_model(
                    "EfficientNetV2S", 3, 10, activation, pretrained=False,
                ).eval()
                with torch.no_grad():
                    output = model(x)
                self.assertEqual(output.shape, (2, 10))

    def test_efficientnet_v2_s_relu_supports_residual_backward(self):
        model = build_model(
            "EfficientNetV2S", 3, 10, "ReLU", pretrained=False,
        ).train()
        output = model(torch.randn(2, 3, 32, 32))
        output.sum().backward()

    def test_locked_lgts_name_builds_all_confirmatory_backbones(self):
        for model_name in ("ResNet18", "MobileNetV2", "EfficientNetV2S"):
            with self.subTest(model=model_name):
                model = build_model(
                    model_name, 3, 100, "LGTS-f2-Slope",
                    pretrained=False, small_input_stem=True,
                )
                self.assertGreater(int(lgts_parameter_stats(model)["count"]), 0)

    def test_f4_full_replaces_all_resnet_activation_sites(self):
        model = build_model(
            "ResNet18", 3, 100, "LGTS-f4-Full",
            pretrained=False, small_input_stem=True,
        )
        stats = lgts_parameter_stats(model)
        self.assertEqual(int(stats["count"]), 17)
        self.assertEqual(stats["learning_mode"], "both")
        self.assertEqual(stats["ablation_role"], "algebraic_plus_logistic_both_learned")


if __name__ == "__main__":
    unittest.main()
