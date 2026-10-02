"""
Tests for Prompt G Novelty Additions:
  1. Abstention gating on low combined reliability.
  2. Counterfactual sensitivity (perturbation of text confidence, direction, dip tolerance).
  3. Fusion Rule Audit table tracking in evaluation harness.
"""

import pytest
from typing import Any, Dict, List

import config
from app.conversation_analysis import ConversationAnalyzer
from app.fusion import MultimodalFusion
from evaluation.dataset import EvaluationDataset, EvaluationSample
from evaluation.harness import EvaluationHarness, SamplePrediction


class TestFusionAbstention:
    """Verify abstention behavior when combined reliability falls below threshold."""

    def test_abstention_triggers_on_low_combined_reliability(self):
        fusion = MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=0.6,
            abstention_threshold=0.35,
        )
        audio_res = {
            "emotion": "angry",
            "confidence": 0.85,
            "voice_reliability": 0.20,
        }
        text_res = {
            "sentiment": "positive",
            "confidence": 0.80,
            "text_reliability": 0.20,
        }
        res = fusion.fuse(audio_res, text_res)

        assert res["abstention"] is True
        assert res["pattern_label"] == "Insufficient reliability. Review audio"
        assert "abstention_low_reliability" in res["rule_ids_fired"]
        assert "Insufficient reliability" in res["interpretation"]

    def test_abstention_does_not_trigger_when_reliable(self):
        fusion = MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=0.6,
            abstention_threshold=0.35,
        )
        audio_res = {
            "emotion": "happy",
            "confidence": 0.85,
            "voice_reliability": 0.90,
        }
        text_res = {
            "sentiment": "positive",
            "confidence": 0.80,
            "text_reliability": 0.85,
        }
        res = fusion.fuse(audio_res, text_res)

        assert res["abstention"] is False
        assert "positive pattern" in res["pattern_label"]
        assert "abstention_low_reliability" not in res["rule_ids_fired"]


class TestCounterfactualSensitivity:
    """Verify deterministic counterfactual sensitivity explanations."""

    def test_segment_counterfactual_detects_text_confidence_cross(self):
        fusion = MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=0.6,
        )
        # Voice negative, text positive, but text confidence is below threshold (0.50 < 0.60)
        audio_res = {
            "emotion": "angry",
            "confidence": 0.85,
            "voice_reliability": 1.0,
        }
        text_res = {
            "sentiment": "positive",
            "confidence": 0.50,
            "text_reliability": 1.0,
        }
        res = fusion.fuse(audio_res, text_res)

        cfs = res.get("counterfactuals", [])
        assert len(cfs) > 0
        # Should contain text confidence perturbation note
        text_cf = [c for c in cfs if "text confidence rose" in c.lower()]
        assert len(text_cf) == 1
        assert "flip from" in text_cf[0].lower()

    def test_segment_counterfactual_detects_abstention_buffer(self):
        fusion = MultimodalFusion(
            abstention_threshold=0.35,
        )
        audio_res = {"emotion": "neutral", "confidence": 0.70, "voice_reliability": 0.90}
        text_res = {"sentiment": "neutral", "confidence": 0.70, "text_reliability": 0.90}
        res = fusion.fuse(audio_res, text_res)

        cfs = res.get("counterfactuals", [])
        buffer_cf = [c for c in cfs if "abstention buffer" in c.lower()]
        assert len(buffer_cf) == 1

    def test_verdict_counterfactual_detects_dip_tolerance_flip(self):
        # 3 segments with rising intensity and a tolerated drop of 0.05
        segments = [
            {"intensity": 0.20},
            {"intensity": 0.40},
            {"intensity": 0.35},  # drop of 0.05 (tolerated under 0.08)
            {"intensity": 0.55},
        ]
        verdict = ConversationAnalyzer.compute_unified_verdict(segments)
        assert verdict["verdict"] == "escalating"

        cfs = verdict.get("counterfactuals", [])
        assert len(cfs) > 0
        tolerance_cf = [c for c in cfs if "dip tolerance falls below" in c.lower()]
        assert len(tolerance_cf) == 1
        assert "0.05" in tolerance_cf[0]


class TestFusionRuleAuditTable:
    """Verify execution tracking in EvaluationHarness."""

    def test_audit_table_records_rule_firings(self):
        harness = EvaluationHarness()
        samples = [
            EvaluationSample(filename="clip1.wav", true_incongruent="yes"),
            EvaluationSample(filename="clip2.wav", true_text_sentiment="positive"),
        ]
        dataset = EvaluationDataset(samples=samples)

        preds = [
            SamplePrediction(
                filename="clip1.wav",
                extra={
                    "fusion_result": {
                        "rule_ids_fired": ["mismatch_positive_text_negative_voice"],
                        "mismatch": True,
                    }
                },
            ),
            SamplePrediction(
                filename="clip2.wav",
                extra={
                    "fusion_result": {
                        "rule_ids_fired": ["agreement_positive"],
                        "mismatch": False,
                    },
                    "pred_text_sentiment": "positive",
                },
            ),
        ]

        result = harness.evaluate_predictions(dataset, preds)
        audit = result.fusion_rule_audit

        assert len(audit) >= 2
        rule_map = {r["rule_id"]: r for r in audit}
        assert "mismatch_positive_text_negative_voice" in rule_map
        assert rule_map["mismatch_positive_text_negative_voice"]["times_fired"] == 1
        assert rule_map["mismatch_positive_text_negative_voice"]["matched_gt"] == 1
        assert rule_map["mismatch_positive_text_negative_voice"]["match_rate"] == 1.0

        assert "agreement_positive" in rule_map
        assert rule_map["agreement_positive"]["times_fired"] == 1
        assert rule_map["agreement_positive"]["matched_gt"] == 1

        summary = result.format_summary()
        assert "FUSION RULE AUDIT TABLE" in summary
