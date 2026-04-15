"""Tests for ModelRegistry and ModelProfile."""

import pytest

from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry


class TestModelProfile:
    def test_creation(self):
        p = ModelProfile(name="test-model", provider="openai")
        assert p.name == "test-model"
        assert p.provider == "openai"
        assert p.model_id == "test-model"  # defaults to name

    def test_custom_model_id(self):
        p = ModelProfile(name="my-alias", provider="openai", model_id="gpt-4o-2024-05-13")
        assert p.model_id == "gpt-4o-2024-05-13"

    def test_has_energy_data(self):
        p = ModelProfile(name="a", provider="x")
        assert not p.has_energy_data

        p2 = ModelProfile(name="b", provider="x", estimated_params_b=70)
        assert p2.has_energy_data

        p3 = ModelProfile(name="c", provider="x", energy_per_query_wh=0.1)
        assert p3.has_energy_data

    def test_to_dict_roundtrip(self):
        p = ModelProfile(
            name="test",
            provider="openai",
            cost_per_1k_input=0.005,
            tags={"code", "math"},
            benchmark_scores={"mmlu": 0.9},
        )
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["tags"] == ["code", "math"]  # sorted

        p2 = ModelProfile.from_dict(d)
        assert p2.name == p.name
        assert p2.tags == p.tags
        assert p2.benchmark_scores == p.benchmark_scores


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        p = ModelProfile(name="gpt-4o", provider="openai")
        reg.register(p)
        assert reg.get("gpt-4o") is p
        assert len(reg) == 1
        assert "gpt-4o" in reg

    def test_unregister(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(name="m1", provider="x"))
        removed = reg.unregister("m1")
        assert removed.name == "m1"
        assert len(reg) == 0

    def test_unregister_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_get_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_models(self, registry):
        models = registry.list_models()
        assert len(models) == 3

    def test_list_models_with_tags(self, registry):
        code_models = registry.list_models(tags={"code"})
        names = {m.name for m in code_models}
        assert names == {"gpt-4o", "claude-sonnet"}

    def test_model_names(self, registry):
        assert set(registry.model_names) == {"gpt-4o", "gpt-4o-mini", "claude-sonnet"}

    def test_from_config(self):
        config = {
            "models": [
                {"name": "m1", "provider": "openai", "cost_per_1k_input": 0.01},
                {"name": "m2", "provider": "anthropic", "tags": ["code"]},
            ]
        }
        reg = ModelRegistry.from_config(config)
        assert len(reg) == 2
        assert reg.get("m1").cost_per_1k_input == 0.01
        assert "code" in reg.get("m2").tags

    def test_to_config_roundtrip(self, registry):
        config = registry.to_config()
        reg2 = ModelRegistry.from_config(config)
        assert set(reg2.model_names) == set(registry.model_names)
