"""Component-level tests (require torch; skipped automatically if torch is absent)."""

import pytest

torch = pytest.importorskip("torch")

from mids_plus.artifact import (LocalArtifactHead, query_orthogonality_loss,
                                weak_artifact_discovery_loss)
from mids_plus.config import MidsPlusConfig
from mids_plus.encoders import StubImageEncoder
from mids_plus.model import build_model
from mids_plus.selector import make_decision
from mids_plus.svd import EffortSVDLinear, replace_linears_with_svd, svd_regularization_losses


def smoke_cfg(**kw):
    base = dict(
        stub_encoders=True, stub_image_layers=2, dim=768, samples_per_image=3,
        clip_adapt="svd", svd_target_last_layers=2, svd_residual_rank=4,
        artifact_enabled=True, image_size=64, amp_dtype="none",
    )
    base.update(kw)
    return MidsPlusConfig.from_dict(base)


def test_svd_reconstructs_original_at_init():
    torch.manual_seed(0)
    lin = torch.nn.Linear(48, 64)
    svd = EffortSVDLinear(lin, residual_rank=8)
    # principal + residual must reconstruct the original weight at initialisation
    assert torch.allclose(svd.current_weight(), lin.weight.detach().float(), atol=1e-4)
    # Frobenius preservation ~ 0 at init; orthogonality finite and non-negative
    assert float(svd.keep_frobenius_loss().detach()) < 1e-2
    assert float(svd.orthogonal_loss().detach()) >= 0.0


def test_svd_residual_is_trainable_principal_is_frozen():
    lin = torch.nn.Linear(32, 32)
    svd = EffortSVDLinear(lin, residual_rank=4)
    assert svd.weight_main.requires_grad is False
    assert svd.U_residual.requires_grad and svd.S_residual.requires_grad and svd.V_residual.requires_grad


def test_replace_linears_targets_last_layers():
    enc = StubImageEncoder(hidden=64, num_layers=4)
    report = replace_linears_with_svd(enc, target_last_layers=2, targets=("self_attn.out_proj",), residual_rank=4)
    assert len(report.replaced) == 2
    losses = svd_regularization_losses(enc)
    assert "orth" in losses and "keep" in losses
    out = enc(torch.randn(2, 3, 64, 64))
    assert out.hidden_states[-1].shape[0] == 2


def test_artifact_head_shapes_and_losses():
    head = LocalArtifactHead(dim=768, num_queries=4)
    patch = torch.randn(3, 16, 768)
    token, amap, qlogits = head(patch)
    assert token.shape == (3, 768)
    assert amap.shape == (3, 16)
    assert qlogits.shape == (3, 4, 16)
    labels = torch.tensor([0, 1, 1])
    assert weak_artifact_discovery_loss(amap, labels).ndim == 0
    assert float(query_orthogonality_loss(qlogits).detach()) >= 0.0


def test_model_forward_and_backward():
    cfg = smoke_cfg()
    model = build_model(cfg)
    b, s = 2, cfg.samples_per_image
    input_ids = torch.randint(0, cfg.stub_text_vocab, (b * s, 12))
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    image = torch.randn(b, 3, cfg.image_size, cfg.image_size)
    labels = torch.randint(0, 4, (b * s,))
    cls_labels = torch.randint(0, 2, (b,))
    out = model(inputs, image, labels, b, s - 1, 0, cls_labels)
    assert out["logits"].shape == (b, s, 4)
    assert out["image_embed"].shape == (b, cfg.dim)
    assert out["artifact_map"].shape[0] == b
    out["cls_loss"].backward()
    grad_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_params) > 0


def test_model_has_seven_fusion_tokens_with_artifact():
    assert build_model(smoke_cfg(artifact_enabled=True)).n_fusion_tokens == 7
    assert build_model(smoke_cfg(artifact_enabled=False)).n_fusion_tokens == 6


def test_selector_picks_best_answer():
    # three answers: claims real/real/fake; only the fake claim is strongly supported
    results = ["real", "real", "fake"]
    scores = torch.tensor([
        [0.4, 0.0, 0.6, 0.0],   # real claim, weakly supported (class2 > class0)
        [0.5, 0.0, 0.5, 0.0],   # real claim, ambiguous
        [0.0, 0.05, 0.0, 0.95], # fake claim, strongly supported
    ])
    best_idx, pred, match, forgery = make_decision(results, scores)
    assert best_idx == 2 and pred == 1 and forgery > 0.9
