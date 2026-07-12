"""Model card rendering (architecture item 7): every saved run carries an honest MODEL_CARD.md."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from churn import model_card, registry
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def test_save_run_writes_a_model_card(tmp_path, sample):
    X, y, ids = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {"model_name": "logistic_regression", "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.6}}
    run_dir = registry.save_run(model, meta, base_dir=tmp_path)

    card = (run_dir / model_card.MODEL_CARD_FILE).read_text()
    assert card.startswith("# Model card -- logistic_regression")
    assert "## Intended use" in card and "## Limitations" in card
    flat = " ".join(card.split())
    assert "no number below is a real-world performance claim" in flat


def test_render_tolerates_minimal_metadata():
    card = model_card.render({})
    # Missing keys degrade to n/a, never a KeyError -- and the honesty framing always renders.
    assert "n/a" in card
    assert "blind to concept drift" in " ".join(card.split())


def test_render_carries_the_selection_honesty_numbers():
    meta = {
        "model_name": "logistic_regression",
        "run_id": "r1",
        "cv_results": {
            "logistic_regression": {"cv_auc_mean": 0.812, "cv_auc_std": 0.011},
            "random_forest": {"cv_auc_mean": 0.815, "cv_auc_std": 0.013},
        },
        "selection": {
            "tolerance": 0.005,
            "runner_up": "random_forest",
            "winner_auc_ci": {"auc_lo": 0.79, "auc_hi": 0.83},
            "winner_vs_runner_up_auc_diff_ci": {"diff_lo": -0.01, "diff_hi": 0.02},
        },
        "ev_sensitivity": [
            {"scenario": "base", "t_star": 0.42, "ev_target_per_1k": 310.0},
        ],
    }
    card = model_card.render(meta)
    assert "random_forest" in card
    assert "[0.790, 0.830]" in card
    assert "[-0.010, 0.020]" in card
    assert "| base | 0.420 | 310.000 |" in card


def test_write_card_rerenders_an_existing_run(tmp_path, sample):
    X, y, ids = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    run_dir = registry.save_run(model, {"model_name": "m"}, base_dir=tmp_path)

    out = model_card.write_card(run_dir)
    assert out == run_dir / model_card.MODEL_CARD_FILE
    assert out.read_text() == (run_dir / model_card.MODEL_CARD_FILE).read_text()
