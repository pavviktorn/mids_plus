"""Pure-Python checks of the 4-class label / selector convention (no torch needed).

This guards the single most error-prone contract in the drop-in: the meaning of the 4 classes and
how the match score maps them back to a verdict.  It must stay identical to FFAA's selector.
"""


def label_of(cls_label: int, claim: str) -> int:
    """The convention encoded by data + selector: ``2*cls_label + (claim == 'fake')``."""
    return 2 * cls_label + (1 if claim == "fake" else 0)


def test_label_convention_matrix():
    assert label_of(0, "real") == 0   # real image, real claim  (correct)
    assert label_of(0, "fake") == 1   # real image, fake claim  (wrong)
    assert label_of(1, "real") == 2   # fake image, real claim  (wrong)
    assert label_of(1, "fake") == 3   # fake image, fake claim  (correct)


def _match_and_pred(claim: str, probs):
    """Pure-Python mirror of selector.make_decision's per-answer match score."""
    if claim == "real":
        return probs[0] / (probs[0] + probs[2]), 0
    return probs[3] / (probs[3] + probs[1]), 1


def test_selector_logic_matches_convention():
    # A confident "real-image / real-claim" answer -> class 0 dominant -> trusted, pred real.
    match, pred = _match_and_pred("real", [0.9, 0.0, 0.1, 0.0])
    assert pred == 0 and match > 0.5
    # A confident "fake-image / fake-claim" answer -> class 3 dominant -> trusted, pred fake.
    match, pred = _match_and_pred("fake", [0.0, 0.1, 0.0, 0.9])
    assert pred == 1 and match > 0.5
    # A "real" claim contradicted by the image (class 2 dominant) -> low match -> distrusted.
    match, _ = _match_and_pred("real", [0.1, 0.0, 0.9, 0.0])
    assert match < 0.5


def test_synthetic_generator_labels_are_consistent():
    import importlib.util
    import os

    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "scripts", "make_synthetic_smoke_data.py")
    spec = importlib.util.spec_from_file_location("synth", path)
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    for cls_label in (0, 1):
        for ans in synth.build_answers(cls_label):
            assert ans["label"] == label_of(cls_label, ans["result"])
