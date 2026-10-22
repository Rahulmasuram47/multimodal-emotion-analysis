"""
Unit tests for Phase 17c: Latency benchmark and calibration pipeline.
Mocked/toy data only; no models, datasets, or tokens needed.
"""
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from evaluation.calibration import (
    compute_ece,
    compute_nll,
    fit_temperature,
    main as calib_main,
    run_calibration_pipeline,
)


def test_01_importing_benchmark_latency_does_not_raise():
    """Verify that importing scripts/benchmark_latency does not raise ImportError or other errors."""
    import scripts.benchmark_latency as bl
    assert hasattr(bl, "run_latency_benchmark")
    assert hasattr(bl, "get_benchmark_audio")


def test_02_calibration_without_run_writes_no_file(monkeypatch, tmp_path):
    """Verify calibration script with --help or without --run prints help and writes NO file."""
    out_json = tmp_path / "should_not_exist.json"
    out_md = tmp_path / "should_not_exist.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "calibration.py",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        calib_main()
    assert exc_info.value.code == 0
    assert not out_json.exists()
    assert not out_md.exists()


def test_03_fit_temperature_recovery_on_toy_logits():
    """Verify scalar temperature recovery on toy uncalibrated logits."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    n_classes = 4
    true_T = 2.0

    raw_logits = rng.randn(n_samples, n_classes).astype(np.float32) * 3.0
    scaled_probs = np.exp(raw_logits / true_T)
    scaled_probs /= np.sum(scaled_probs, axis=1, keepdims=True)

    labels = np.array([rng.choice(n_classes, p=scaled_probs[i]) for i in range(n_samples)])

    recovered_T = fit_temperature(raw_logits, labels)
    assert abs(recovered_T - true_T) < 0.35


def test_04_ece_near_zero_on_calibrated_toy_data():
    """Verify that Expected Calibration Error is near zero on perfectly calibrated probabilities."""
    rng = np.random.RandomState(42)
    n_samples = 3000
    confidences = rng.uniform(0.5, 1.0, size=n_samples)
    labels = (rng.rand(n_samples) < confidences).astype(int)
    probs = np.zeros((n_samples, 2), dtype=np.float32)
    probs[:, 1] = confidences
    probs[:, 0] = 1.0 - confidences

    ece, bins = compute_ece(probs, labels, n_bins=10)
    assert ece < 0.05
    assert len(bins) == 10


def test_05_not_applicable_names_cache_contents_and_decision_line(tmp_path):
    """Verify that 'not applicable' report describes inspected cache contents and specifies deciding file/line."""
    out_json = tmp_path / "calib.json"
    out_md = tmp_path / "calib.md"
    fake_cache_info = {
        "sample_keys": ["embedding"],
        "sample_shapes": {"embedding": [768]},
    }
    res = run_calibration_pipeline(
        dev_logits=None,
        dev_labels=None,
        cache_info=fake_cache_info,
        output_json=out_json,
        output_md=out_md,
    )
    assert res["status"] == "not applicable: no usable scores"
    assert "evaluation/calibration.py:215" in res["reason"]
    assert "embedding" in res["reason"]
    assert out_json.is_file()
    assert out_md.is_file()

    md_text = out_md.read_text(encoding="utf-8")
    assert "Acted speech; argmax and accuracy are unchanged by temperature scaling" in md_text
    assert "evaluation/calibration.py:215" in md_text
