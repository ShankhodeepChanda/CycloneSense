"""Tests for the multi-modal Bay of Bengal track and intensity forecaster."""

from __future__ import annotations

import torch

from ml.sequence_dataset import FEATURES, INPUT_STEPS, TARGET_LEADS
from ml.trajectory_model import (
    BayOfBengalCycloneForecaster,
    load_forecaster,
    save_forecaster_checkpoint,
)

BATCH = 3
HORIZON = len(TARGET_LEADS)


def tabular_batch(batch: int = BATCH) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, INPUT_STEPS, len(FEATURES))


def image_batch(batch: int = BATCH) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, INPUT_STEPS, 3, 224, 224)


class TestTabularOnlyPath:
    def test_forward_returns_the_documented_output_shapes(self):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        out = model(tabular_batch())
        assert out["delta"].shape == (BATCH, HORIZON, 2)
        assert out["intensity"].shape == (BATCH, HORIZON, 2)

    def test_outputs_are_finite(self):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        out = model(tabular_batch())
        assert torch.isfinite(out["delta"]).all()
        assert torch.isfinite(out["intensity"]).all()

    def test_eval_mode_forward_is_deterministic(self):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        tabular = tabular_batch()
        with torch.inference_mode():
            first = model(tabular)["delta"]
            second = model(tabular)["delta"]
        torch.testing.assert_close(first, second)

    def test_tabular_only_model_stays_small_enough_to_ship(self):
        model = BayOfBengalCycloneForecaster(use_vision=False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < 1_000_000

    def test_images_passed_to_a_tabular_only_model_are_ignored(self):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        tabular = tabular_batch()
        with torch.inference_mode():
            without = model(tabular)["delta"]
            with_images = model(tabular, image_batch(), torch.ones(BATCH, INPUT_STEPS))["delta"]
        torch.testing.assert_close(without, with_images)

    def test_gradients_reach_the_tabular_encoder(self):
        model = BayOfBengalCycloneForecaster(use_vision=False)
        model(tabular_batch())["delta"].sum().backward()
        grads = [p.grad for p in model.tabular.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


class TestVisionPath:
    def test_forward_accepts_image_sequences_with_a_presence_mask(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        out = model(tabular_batch(), image_batch(), torch.ones(BATCH, INPUT_STEPS))
        assert out["delta"].shape == (BATCH, HORIZON, 2)
        assert out["intensity"].shape == (BATCH, HORIZON, 2)
        assert torch.isfinite(out["delta"]).all()

    def test_a_fully_masked_batch_still_forecasts(self):
        """Every North Indian Ocean step may lack a chip; that must not break inference."""
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        out = model(tabular_batch(), torch.zeros(BATCH, INPUT_STEPS, 3, 224, 224), torch.zeros(BATCH, INPUT_STEPS))
        assert out["delta"].shape == (BATCH, HORIZON, 2)
        assert torch.isfinite(out["delta"]).all()
        assert torch.isfinite(out["intensity"]).all()

    def test_a_masked_step_uses_the_learned_missing_embedding_not_the_pixels(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        tabular, images = tabular_batch(), image_batch()
        masked = torch.zeros(BATCH, INPUT_STEPS)
        with torch.inference_mode():
            real_pixels = model(tabular, images, masked)["delta"]
            other_pixels = model(tabular, torch.randn_like(images), masked)["delta"]
        torch.testing.assert_close(real_pixels, other_pixels)

    def test_an_unmasked_step_does_depend_on_the_pixels(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        tabular, images = tabular_batch(), image_batch()
        present = torch.ones(BATCH, INPUT_STEPS)
        with torch.inference_mode():
            first = model(tabular, images, present)["delta"]
            second = model(tabular, torch.randn_like(images), present)["delta"]
        assert not torch.allclose(first, second)

    def test_missing_mask_defaults_to_treating_images_as_present(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        out = model(tabular_batch(), image_batch())
        assert out["delta"].shape == (BATCH, HORIZON, 2)

    def test_vision_model_without_images_falls_back_to_the_missing_embedding(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        out = model(tabular_batch())
        assert out["delta"].shape == (BATCH, HORIZON, 2)
        assert torch.isfinite(out["delta"]).all()

    def test_early_vision_layers_are_frozen(self):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False)
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        assert any("vision.trunk.conv1" in n for n in frozen)
        assert any("vision.trunk.layer4" in n for n, p in model.named_parameters() if p.requires_grad)


class TestCheckpointRoundTrip:
    def test_saved_checkpoint_reloads_to_identical_outputs(self, tmp_path):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        tabular = tabular_batch()
        with torch.inference_mode():
            before = model(tabular)["delta"]
        path = tmp_path / "forecaster.pt"
        save_forecaster_checkpoint(model, path, lead_track_rmse_km=[10.0] * HORIZON, metrics={"val_track_mae_km": 42.0})
        restored, meta = load_forecaster(path)
        with torch.inference_mode():
            after = restored.eval()(tabular)["delta"]
        torch.testing.assert_close(before, after)
        assert meta["use_vision"] is False
        assert meta["feature_names"] == FEATURES
        assert meta["target_leads"] == TARGET_LEADS
        assert meta["input_steps"] == INPUT_STEPS
        assert meta["lead_track_rmse_km"] == [10.0] * HORIZON
        assert meta["metrics"]["val_track_mae_km"] == 42.0
        assert "norm" in meta

    def test_a_vision_checkpoint_reloads_with_its_vision_branch(self, tmp_path):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        path = tmp_path / "vision.pt"
        save_forecaster_checkpoint(model, path, lead_track_rmse_km=[1.0] * HORIZON, metrics={})
        restored, meta = load_forecaster(path)
        assert meta["use_vision"] is True
        out = restored.eval()(tabular_batch(), image_batch(), torch.ones(BATCH, INPUT_STEPS))
        assert out["intensity"].shape == (BATCH, HORIZON, 2)
