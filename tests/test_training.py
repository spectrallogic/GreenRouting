"""Tests for synthetic data generation and training pipeline."""

from greenrouting.core.taxonomy import Capability
from greenrouting.training.synthetic_data import (
    generate_dataset,
    load_dataset,
    save_dataset,
)


class TestSyntheticData:
    def test_generate_dataset(self):
        examples = generate_dataset(n_per_category=5, seed=42)
        assert len(examples) > 0

        # Check all examples are valid
        valid_capabilities = {c.value for c in Capability}
        for ex in examples:
            # capability_weights keys should be valid capabilities
            for cap in ex.capability_weights:
                assert cap in valid_capabilities
            # weights should sum to approximately 1.0
            assert abs(sum(ex.capability_weights.values()) - 1.0) < 0.01
            assert 1 <= ex.difficulty <= 5
            assert ex.expected_output_length in ("short", "medium", "long")
            assert len(ex.query) > 0

    def test_dataset_covers_all_capabilities(self):
        examples = generate_dataset(n_per_category=5, seed=42)
        # Collect all capabilities that appear as primary (highest weight)
        capabilities = set()
        for ex in examples:
            top = max(ex.capability_weights, key=lambda k: ex.capability_weights[k])
            capabilities.add(top)
        assert len(capabilities) >= 6

    def test_dataset_covers_difficulty_range(self):
        examples = generate_dataset(n_per_category=10, seed=42)
        difficulties = {ex.difficulty for ex in examples}
        assert len(difficulties) >= 4

    def test_deterministic_with_seed(self):
        d1 = generate_dataset(n_per_category=5, seed=123)
        d2 = generate_dataset(n_per_category=5, seed=123)
        assert len(d1) == len(d2)
        assert d1[0].query == d2[0].query

    def test_save_and_load(self, tmp_path):
        examples = generate_dataset(n_per_category=3, seed=42)
        path = tmp_path / "test_data.jsonl"

        save_dataset(examples, path)
        loaded = load_dataset(path)

        assert len(loaded) == len(examples)
        assert loaded[0].query == examples[0].query
        assert loaded[0].capability_weights == examples[0].capability_weights

    def test_mixed_capability_examples_exist(self):
        """Dataset should include mixed-capability queries."""
        examples = generate_dataset(n_per_category=10, seed=42)
        mixed = [ex for ex in examples if len(ex.capability_weights) > 1]
        assert len(mixed) > 0
        # Mixed examples should have multiple weights
        for ex in mixed:
            assert sum(1 for w in ex.capability_weights.values() if w >= 0.1) >= 2

    def test_dataset_size_scales(self):
        """More examples per category should produce a larger dataset."""
        small = generate_dataset(n_per_category=5, seed=42)
        large = generate_dataset(n_per_category=50, seed=42)
        assert len(large) > len(small) * 5  # Should scale significantly
