"""GreenRouting Client — classifies queries and routes to the optimal model.

This is the main integration point for any application. The client works with
any setup — API providers, self-hosted models, agent frameworks, or custom
infrastructure. Users bring their own models and API access; the classifier
runs locally and picks the greenest option.

Three ways to use it:

1. **With litellm (easiest — supports 100+ providers):**
   ```python
   client = GreenRoutingClient(models=["gpt-4o-mini", "claude-haiku", "llama-3.1-8b"])
   response = client.chat("What is 2+2?")
   ```

2. **With a custom completion function (any backend):**
   ```python
   client = GreenRoutingClient(
       models=["gpt-4o-mini", "claude-haiku"],
       completion_fn=my_custom_llm_call,
   )
   ```

3. **Classify-only (you handle the API call):**
   ```python
   client = GreenRoutingClient(models=["gpt-4o-mini", "claude-haiku"])
   decision = client.classify("What is 2+2?")
   # decision.selected_model -> "gpt-4o-mini"
   # Now call the model yourself however you want
   ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from greenrouting.core.compression import get_compression_hint
from greenrouting.core.decision import RoutingDecision
from greenrouting.core.registry import ModelRegistry
from greenrouting.energy.profiles import get_known_profiles
from greenrouting.energy.tracker import EnergyTracker


# Type for user-provided completion functions
# Signature: (model: str, messages: list[dict], **kwargs) -> str
CompletionFn = Callable[..., str]


@dataclass
class RoutedResponse:
    """Response from a routed call — includes the model output plus routing metadata."""

    content: str
    routed_to: str
    decision: RoutingDecision
    energy_wh: float
    energy_savings_pct: float
    compressed: bool
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    def __str__(self) -> str:
        return self.content


@dataclass
class ModelConfig:
    """Configuration for a single model in the routing pool.

    For known models (gpt-4o, claude-haiku, etc.) you only need the name.
    For custom models, provide the provider and model_id that your backend expects.
    """

    name: str
    provider: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    def resolve_model_id(self) -> str:
        """Return the model identifier for API calls."""
        return self.model_id or self.name


class GreenRoutingClient:
    """Intelligent routing client for any AI application.

    Classifies each query, picks the most energy-efficient model from your pool,
    and optionally forwards the call. Works with any provider, framework, or
    infrastructure setup.

    Args:
        models: Model names (from known profiles) or ModelConfig objects.
            If None, all 11 known profiles are loaded.
        preset: Green Score preset — "balanced", "quality_first", or "maximum_green".
        completion_fn: Optional custom function to call models. Signature:
            ``(model: str, messages: list[dict], **kwargs) -> str``.
            If not provided, uses litellm (install with ``pip install litellm``).
        classifier_path: Path to custom trained classifier weights.
        tracker: Optional EnergyTracker to accumulate savings across calls.
    """

    def __init__(
        self,
        models: list[str] | list[ModelConfig] | None = None,
        preset: str = "balanced",
        completion_fn: CompletionFn | None = None,
        classifier_path: str | None = None,
        tracker: EnergyTracker | None = None,
    ) -> None:
        from greenrouting import load_pretrained

        # Build registry from user's model list
        known = get_known_profiles()
        self.registry = ModelRegistry()
        self.model_configs: dict[str, ModelConfig] = {}

        if models is None:
            for profile in known.values():
                self.registry.register(profile)
                self.model_configs[profile.name] = ModelConfig(
                    name=profile.name,
                    provider=profile.provider,
                    model_id=profile.model_id,
                )
        else:
            for model in models:
                if isinstance(model, str):
                    if model in known:
                        profile = known[model]
                        self.registry.register(profile)
                        self.model_configs[model] = ModelConfig(
                            name=model,
                            provider=profile.provider,
                            model_id=profile.model_id,
                        )
                    else:
                        raise ValueError(
                            f"Model '{model}' not found in known profiles. "
                            f"Available: {list(known.keys())}. "
                            f"For custom models, pass a ModelConfig object."
                        )
                elif isinstance(model, ModelConfig):
                    if model.name in known:
                        profile = known[model.name]
                        self.registry.register(profile)
                    else:
                        from greenrouting.core.model_profile import ModelProfile

                        self.registry.register(ModelProfile(
                            name=model.name,
                            provider=model.provider or "custom",
                            model_id=model.model_id or model.name,
                        ))
                    self.model_configs[model.name] = model

        # Load the classifier
        self.router = load_pretrained(
            model_dir=classifier_path,
            registry=self.registry,
            scorer_config={"green_score": {"preset": preset}},
        )

        self.tracker = tracker or EnergyTracker()
        self.preset = preset
        self._completion_fn = completion_fn

    # ── Classify only (user handles the API call) ────────────────────

    def classify(self, query: str) -> RoutingDecision:
        """Classify a query and return the routing decision without calling any API.

        Use this when you want to handle the model call yourself — in your own
        agent framework, custom pipeline, or infrastructure.

        Args:
            query: The user's query/prompt.

        Returns:
            RoutingDecision with selected_model, energy estimates, and reasoning.

        Example::

            decision = client.classify("Write a Python sort function")
            print(decision.selected_model)        # "gpt-4o-mini"
            print(decision.energy_savings_vs_max)  # 0.98

            # Now call the model yourself
            response = my_llm_call(model=decision.selected_model, prompt=query)
        """
        decision = self.router.route(query)

        max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
        max_cost = max(s.cost_estimate for s in decision.all_scores.values())
        self.tracker.record(
            decision.energy_estimate_wh, max_energy,
            decision.cost_estimate, max_cost,
        )

        return decision

    def get_compression_hint(self, query: str) -> dict[str, Any]:
        """Get compression guidance for a query (useful for agent frameworks).

        Returns a dict with should_compress, level, and a system prompt snippet
        that can be prepended to reduce output tokens on simple queries.

        Example::

            hint = client.get_compression_hint("What is 2+2?")
            if hint["should_compress"]:
                system_prompt += hint["system_instruction"]
        """
        profile = self.router.classify_query(query)
        hint = get_compression_hint(profile)
        return {
            "should_compress": hint.should_compress,
            "level": hint.level if hint.should_compress else None,
            "estimated_token_savings_pct": hint.estimated_token_savings_pct,
            "system_instruction": (
                "Be concise and direct. Give the shortest accurate answer possible."
                if hint.should_compress else ""
            ),
        }

    # ── Full routing + API call ──────────────────────────────────────

    def chat(
        self,
        message: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> RoutedResponse:
        """Classify, route, and call the optimal model in one step.

        This is the simplest way to use GreenRouting — send a message, get a response.
        The client handles everything: classification, model selection, compression
        hints, and the actual API call.

        Uses your custom completion_fn if provided, otherwise falls back to litellm.

        Args:
            message: The user's query/prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments passed to the completion function.

        Returns:
            RoutedResponse with the model's output and routing metadata.
        """
        # Classify and route
        decision = self.router.route(message)
        profile = self.router.classify_query(message)
        hint = get_compression_hint(profile)

        # Build messages
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        if hint.should_compress:
            compress_instruction = (
                "Be concise and direct. Give the shortest accurate answer possible."
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{compress_instruction}"
            else:
                messages.insert(0, {"role": "system", "content": compress_instruction})

        messages.append({"role": "user", "content": message})

        # Resolve model
        config = self.model_configs[decision.selected_model]
        model_id = config.resolve_model_id()

        # Call the model
        content, usage, raw = self._call_model(
            model_id=model_id,
            config=config,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Track energy
        max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
        max_cost = max(s.cost_estimate for s in decision.all_scores.values())
        self.tracker.record(
            decision.energy_estimate_wh, max_energy,
            decision.cost_estimate, max_cost,
        )

        return RoutedResponse(
            content=content,
            routed_to=decision.selected_model,
            decision=decision,
            energy_wh=decision.energy_estimate_wh,
            energy_savings_pct=decision.energy_savings_vs_max * 100,
            compressed=hint.should_compress,
            usage=usage,
            raw_response=raw,
        )

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> RoutedResponse:
        """Route and forward a full message list (OpenAI-compatible format).

        Uses the last user message for classification. Works as a drop-in
        replacement anywhere you currently call an LLM with a messages list.

        Args:
            messages: OpenAI-format message list.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments passed to the completion function.

        Returns:
            RoutedResponse with the model's output and routing metadata.
        """
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user message found in messages list.")

        query = user_messages[-1]["content"]

        # Classify and route
        decision = self.router.route(query)
        profile = self.router.classify_query(query)
        hint = get_compression_hint(profile)

        # Apply compression
        final_messages = list(messages)
        if hint.should_compress:
            compress_instruction = (
                "Be concise and direct. Give the shortest accurate answer possible."
            )
            if final_messages and final_messages[0]["role"] == "system":
                final_messages[0] = {
                    **final_messages[0],
                    "content": final_messages[0]["content"] + f"\n\n{compress_instruction}",
                }
            else:
                final_messages.insert(0, {"role": "system", "content": compress_instruction})

        # Call the model
        config = self.model_configs[decision.selected_model]
        model_id = config.resolve_model_id()

        content, usage, raw = self._call_model(
            model_id=model_id,
            config=config,
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
        max_cost = max(s.cost_estimate for s in decision.all_scores.values())
        self.tracker.record(
            decision.energy_estimate_wh, max_energy,
            decision.cost_estimate, max_cost,
        )

        return RoutedResponse(
            content=content,
            routed_to=decision.selected_model,
            decision=decision,
            energy_wh=decision.energy_estimate_wh,
            energy_savings_pct=decision.energy_savings_vs_max * 100,
            compressed=hint.should_compress,
            usage=usage,
            raw_response=raw,
        )

    # ── Reports ──────────────────────────────────────────────────────

    def report(self) -> str:
        """Return a cumulative energy savings report for all calls through this client."""
        return str(self.tracker.report())

    # ── Internal ─────────────────────────────────────────────────────

    def _call_model(
        self,
        model_id: str,
        config: ModelConfig,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int], Any]:
        """Call the model using the user's completion function or litellm."""

        if self._completion_fn:
            # User-provided completion function
            content = self._completion_fn(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return content, {}, None

        # Fall back to litellm
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "No completion function provided and litellm is not installed. "
                "Either:\n"
                "  1. Install litellm: pip install litellm\n"
                "  2. Pass a custom completion_fn to GreenRoutingClient\n"
                "  3. Use client.classify() for routing-only (no API call)"
            ) from None

        call_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        if config.api_key:
            call_kwargs["api_key"] = config.api_key
        if config.api_base:
            call_kwargs["api_base"] = config.api_base
        call_kwargs.update(kwargs)

        response = litellm.completion(**call_kwargs)

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return content, usage, response
