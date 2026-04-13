from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from domain_matcher import FontCosineColumn, SearchHit, SearchReport
from web_app import _apply_exact_benign_hit_override


def _report(*, matches: list[SearchHit], top_candidates: list[SearchHit], total_threshold_hits: int = 2) -> SearchReport:
    return SearchReport(
        query="google.com",
        normalized_query="google",
        scanned_rows=100,
        total_rows_target=100,
        total_threshold_hits=total_threshold_hits,
        duration_seconds=0.12,
        feature_mode="precomputed_projected",
        warnings=[],
        font_columns=[FontCosineColumn(key="cosine_deja", label="Deja")],
        matches=matches,
        top_candidates=top_candidates,
    )


class ExactOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "domains.csv"
        self.dataset_path.write_text("domain\ngoogle.com\ngoogle.co.in\ngoogle.de\n", encoding="utf-8")
        self.matcher = SimpleNamespace(
            dataset_path=self.dataset_path,
            precomputed_store=None,
            font_feature_names=["cosine_deja"],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_exact_override_promotes_the_exact_host_in_both_lists(self) -> None:
        report = _report(
            matches=[
                SearchHit(domain="google.co.in", mean_font_cosine=1.0, font_cosines={"cosine_deja": 1.0}),
                SearchHit(domain="google.com", mean_font_cosine=0.87, font_cosines={"cosine_deja": 0.87}),
            ],
            top_candidates=[
                SearchHit(domain="google.co.in", mean_font_cosine=1.0, font_cosines={"cosine_deja": 1.0}),
                SearchHit(domain="google.de", mean_font_cosine=0.96, font_cosines={"cosine_deja": 0.96}),
            ],
        )

        updated = _apply_exact_benign_hit_override(
            report,
            self.matcher,
            "https://www.google.com/search",
            top_k=2,
        )

        self.assertEqual(updated.matches[0].domain, "google.com")
        self.assertEqual(updated.top_candidates[0].domain, "google.com")
        self.assertTrue(updated.matches[0].exact_match)
        self.assertTrue(updated.top_candidates[0].exact_match)
        self.assertEqual(updated.matches[0].mean_font_cosine, 1.0)
        self.assertEqual(updated.top_candidates[0].mean_font_cosine, 1.0)
        self.assertEqual(len(updated.matches), 2)
        self.assertEqual(len(updated.top_candidates), 2)

    def test_exact_override_does_not_change_threshold_hit_count(self) -> None:
        report = _report(
            matches=[
                SearchHit(domain="google.co.in", mean_font_cosine=1.0, font_cosines={"cosine_deja": 1.0}),
            ],
            top_candidates=[
                SearchHit(domain="google.co.in", mean_font_cosine=1.0, font_cosines={"cosine_deja": 1.0}),
                SearchHit(domain="google.de", mean_font_cosine=0.96, font_cosines={"cosine_deja": 0.96}),
            ],
            total_threshold_hits=42,
        )

        updated = _apply_exact_benign_hit_override(
            report,
            self.matcher,
            "google.com",
            top_k=2,
        )

        self.assertEqual(updated.total_threshold_hits, 42)
        self.assertEqual(updated.matches[0].domain, "google.com")
        self.assertEqual(updated.top_candidates[0].domain, "google.com")
        self.assertEqual(len(updated.matches), 2)
        self.assertEqual(len(updated.top_candidates), 2)


if __name__ == "__main__":
    unittest.main()
