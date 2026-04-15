"""Tests for caveman-style compression hints."""

from greenrouting.core.compression import get_compression_hint
from greenrouting.core.taxonomy import Capability, QueryProfile


class TestCompressionHint:
    def test_trivial_query_gets_aggressive_compression(self):
        profile = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        hint = get_compression_hint(profile)
        assert hint.level == "aggressive"
        assert hint.should_compress
        assert hint.estimated_token_savings_pct == 0.65
        assert len(hint.system_prompt_addon) > 0

    def test_medium_query_gets_moderate_compression(self):
        profile = QueryProfile.single(Capability.CODE, difficulty=3)
        hint = get_compression_hint(profile)
        assert hint.level == "moderate"
        assert hint.should_compress
        assert hint.estimated_token_savings_pct == 0.30

    def test_hard_query_gets_no_compression(self):
        profile = QueryProfile.single(Capability.REASONING, difficulty=5)
        hint = get_compression_hint(profile)
        assert hint.level == "none"
        assert not hint.should_compress
        assert hint.estimated_token_savings_pct == 0.0
        assert hint.system_prompt_addon == ""

    def test_difficulty_2_gets_aggressive(self):
        profile = QueryProfile.single(Capability.MATH, difficulty=2)
        hint = get_compression_hint(profile)
        assert hint.level == "aggressive"

    def test_difficulty_4_gets_no_compression(self):
        profile = QueryProfile.single(Capability.CODE, difficulty=4)
        hint = get_compression_hint(profile)
        assert hint.level == "none"
