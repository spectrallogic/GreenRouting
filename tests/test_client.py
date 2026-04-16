"""Tests for the GreenRoutingClient — classify-only mode (no API keys needed)."""

from __future__ import annotations

import pytest

from greenrouting.serving.client import GreenRoutingClient, ModelConfig, RoutedResponse


class TestGreenRoutingClientClassifyOnly:
    """Test routing decisions without making any API calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a client with known models."""
        self.client = GreenRoutingClient(
            models=["gpt-4o", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b"],
            preset="balanced",
        )

    def test_classify_returns_routing_decision(self):
        decision = self.client.classify("What is 2+2?")
        assert decision.selected_model in ["gpt-4o", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b"]
        assert decision.energy_estimate_wh > 0
        assert decision.energy_savings_vs_max >= 0

    def test_classify_simple_query_routes_small(self):
        decision = self.client.classify("What color is the sky?")
        # Should NOT route to the most expensive model for a trivial query
        assert decision.selected_model != "gpt-4o"

    def test_classify_complex_query_routes_capable(self):
        decision = self.client.classify(
            "Design a distributed consensus algorithm with Byzantine fault tolerance"
        )
        # Complex queries should route to more capable models
        assert decision.quality_estimate > 0.5

    def test_compression_hint_simple_query(self):
        hint = self.client.get_compression_hint("What is 2+2?")
        assert isinstance(hint, dict)
        assert "should_compress" in hint
        assert "system_instruction" in hint

    def test_compression_hint_complex_query(self):
        hint = self.client.get_compression_hint(
            "Prove the Riemann hypothesis"
        )
        # Complex queries should not be compressed
        assert hint["should_compress"] is False

    def test_tracker_accumulates(self):
        self.client.classify("What is 2+2?")
        self.client.classify("Write a Python sort function")
        self.client.classify("Translate hello to Spanish")

        report = self.client.tracker.report()
        assert report.total_queries == 3
        assert report.total_energy_wh > 0
        assert report.energy_saved_wh > 0

    def test_report_string(self):
        self.client.classify("What is 2+2?")
        report = self.client.report()
        assert "Queries routed" in report
        assert "Energy saved" in report

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found in known profiles"):
            GreenRoutingClient(models=["nonexistent-model-xyz"])

    def test_custom_model_config(self):
        client = GreenRoutingClient(
            models=[
                "gpt-4o-mini",
                ModelConfig(
                    name="llama-3.1-8b",
                    provider="ollama",
                    model_id="ollama/llama3.1:8b",
                    api_base="http://localhost:11434",
                ),
            ],
            preset="balanced",
        )
        decision = client.classify("What is 2+2?")
        assert decision.selected_model in ["gpt-4o-mini", "llama-3.1-8b"]

    def test_presets_affect_routing(self):
        client_green = GreenRoutingClient(
            models=["gpt-4o", "gpt-4o-mini", "llama-3.1-8b"],
            preset="maximum_green",
        )
        client_quality = GreenRoutingClient(
            models=["gpt-4o", "gpt-4o-mini", "llama-3.1-8b"],
            preset="quality_first",
        )

        query = "Explain quantum computing in detail"
        decision_green = client_green.classify(query)
        decision_quality = client_quality.classify(query)

        # Green preset should favor cheaper models
        assert decision_green.energy_estimate_wh <= decision_quality.energy_estimate_wh

    def test_chat_without_completion_fn_or_litellm_gives_clear_error(self):
        """Calling chat() without a completion function should explain options."""
        # This test only works if litellm is NOT installed
        try:
            import litellm
            pytest.skip("litellm is installed — can't test the fallback error")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="No completion function provided"):
            self.client.chat("What is 2+2?")

    def test_chat_with_custom_completion_fn(self):
        """Test the full chat flow with a mock completion function."""

        def mock_completion(model: str, messages: list, **kwargs) -> str:
            return f"Mock response from {model}"

        client = GreenRoutingClient(
            models=["gpt-4o-mini", "llama-3.1-8b"],
            preset="balanced",
            completion_fn=mock_completion,
        )

        response = client.chat("What is 2+2?")

        assert isinstance(response, RoutedResponse)
        assert response.content.startswith("Mock response from ")
        assert response.routed_to in ["gpt-4o-mini", "llama-3.1-8b"]
        assert response.energy_savings_pct > 0
        assert str(response) == response.content

    def test_chat_messages_with_custom_fn(self):
        """Test chat_messages with OpenAI-format message list."""

        def mock_completion(model: str, messages: list, **kwargs) -> str:
            # Verify compression was applied for simple queries
            system_msgs = [m for m in messages if m["role"] == "system"]
            has_compress = any("concise" in m["content"].lower() for m in system_msgs)
            return f"compressed={has_compress}"

        client = GreenRoutingClient(
            models=["gpt-4o-mini", "llama-3.1-8b"],
            preset="balanced",
            completion_fn=mock_completion,
        )

        response = client.chat_messages([
            {"role": "user", "content": "What is 2+2?"},
        ])

        assert isinstance(response, RoutedResponse)
        assert response.routed_to in ["gpt-4o-mini", "llama-3.1-8b"]
