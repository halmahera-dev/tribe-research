from __future__ import annotations

import numpy as np

from app.services.scoring import build_prediction_from_preds


def test_build_prediction_from_preds_shape_and_bounds() -> None:
    rng = np.random.default_rng(42)
    preds = rng.normal(0.0, 1.0, size=(12, 90))
    result = build_prediction_from_preds(preds, segments=None, engine="tribev2")

    assert result.virality_score >= 0
    assert result.virality_score <= 100
    assert result.hook_score >= 0
    assert result.hook_score <= 100
    assert result.hold_rate >= 0
    assert result.hold_rate <= 100
    assert len(result.timeline) == 12
    assert len(result.brain_heatmap) > 0
