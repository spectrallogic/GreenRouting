"""Tests for the capability taxonomy and query profiles."""

from greenrouting.core.taxonomy import Capability, QueryProfile


class TestQueryProfile:
    def test_difficulty_clamped(self):
        p = QueryProfile.single(Capability.SIMPLE, difficulty=0)
        assert p.difficulty == 1

        p2 = QueryProfile.single(Capability.SIMPLE, difficulty=10)
        assert p2.difficulty == 5

    def test_is_trivial(self):
        trivial = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        assert trivial.is_trivial

        not_trivial = QueryProfile.single(Capability.REASONING, difficulty=1)
        assert not not_trivial.is_trivial

    def test_needs_strong_model(self):
        easy = QueryProfile.single(Capability.MATH, difficulty=2)
        assert not easy.needs_strong_model

        hard = QueryProfile.single(Capability.MATH, difficulty=4)
        assert hard.needs_strong_model

    def test_min_benchmark_threshold(self):
        p1 = QueryProfile.single(Capability.CODE, difficulty=1)
        p5 = QueryProfile.single(Capability.CODE, difficulty=5)

        assert p1.min_benchmark_threshold() < p5.min_benchmark_threshold()
        assert p1.min_benchmark_threshold() == 0.2
        assert abs(p5.min_benchmark_threshold() - 0.9) < 1e-9

    def test_all_capabilities_exist(self):
        assert len(Capability) == 8
        assert Capability.SIMPLE in Capability
        assert Capability.REASONING in Capability

    def test_mixed_profile(self):
        p = QueryProfile(
            capability_weights={"code": 0.5, "math": 0.35, "reasoning": 0.15},
            difficulty=4,
        )
        assert p.is_mixed
        assert p.primary_capability == Capability.CODE
        assert p.needs_strong_model

    def test_single_not_mixed(self):
        p = QueryProfile.single(Capability.MATH, difficulty=3)
        assert not p.is_mixed
        assert p.primary_capability == Capability.MATH
