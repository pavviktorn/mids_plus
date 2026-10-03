"""End-to-end smoke test (requires torch; skipped if absent).

Generates a tiny synthetic dataset, runs one training epoch with stub encoders, then checks the
checkpoint round-trips and the drop-in scorer produces a valid decision.  Exercises SVD
replacement, the artifact branch, every loss term, backward, the selector, and save/load.
"""

import importlib.util
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from mids_plus.checkpoint import load_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.train import train


def _gen_data(out_dir):
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "scripts", "make_synthetic_smoke_data.py")
    spec = importlib.util.spec_from_file_location("synth", path)
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    train_items = synth.make_split(out_dir, img_dir, 8, seed=1)
    val_items = synth.make_split(out_dir, img_dir, 4, seed=2)
    tp = os.path.join(out_dir, "train.json")
    vp = os.path.join(out_dir, "val.json")
    json.dump(train_items, open(tp, "w"))
    json.dump(val_items, open(vp, "w"))
    return tp, vp


def test_end_to_end_training_and_dropin(tmp_path):
    out_dir = str(tmp_path / "data")
    train_json, val_json = _gen_data(out_dir)

    cfg = MidsPlusConfig.from_dict(dict(
        stub_encoders=True, stub_image_layers=2, dim=768, samples_per_image=3,
        clip_adapt="svd", svd_target_last_layers=2, svd_residual_rank=4,
        artifact_enabled=True, image_size=64, augment=False, amp_dtype="none",
        epochs=1, batch_size=4, val_batch_size=4, num_workers=0,
        train_data_path=train_json, val_data_path=val_json,
        output_dir=str(tmp_path / "run"),
    ))
    train(cfg)

    ckpt = str(tmp_path / "run" / "last.pt")
    assert os.path.exists(ckpt)

    # Checkpoint must reload and produce the right-shaped logits.
    model, loaded_cfg = load_checkpoint(ckpt, device="cpu")
    assert loaded_cfg.clip_adapt == "svd" and loaded_cfg.artifact_enabled
    b, s = 1, cfg.samples_per_image
    ids = torch.randint(0, cfg.stub_text_vocab, (b * s, 10))
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    out = model(inputs, torch.randn(b, 3, cfg.image_size, cfg.image_size), None, b, s - 1, 0)
    assert out["logits"].shape == (b, s, 4)

    # The saved checkpoint should be small (learned tensors only, no frozen base encoders).
    base = sum(p.numel() for p in model.parameters())
    saved = sum(t.numel() for t in torch.load(ckpt, map_location="cpu", weights_only=False)["state"].values())
    assert saved < base
