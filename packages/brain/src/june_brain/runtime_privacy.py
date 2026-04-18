"""Runtime and privacy status helpers for JuneAI."""

from __future__ import annotations

from typing import Any

from .config import (
    RuntimeConfig,
    build_runtime_preset_switch_plan,
    resolve_runtime_config,
    resolve_runtime_mode,
)

JsonDict = dict[str, Any]


def build_runtime_privacy_status(runtime: RuntimeConfig | None = None) -> JsonDict:
    """Build a structured privacy/runtime snapshot for the active model runtime."""
    runtime = runtime or resolve_runtime_config()
    mode = resolve_runtime_mode(runtime.provider, runtime.base_url)
    remote_label = runtime.base_url.strip() or "remote API"
    endpoint_label = "localhost" if mode == "local" else remote_label

    warnings: list[str] = []
    if runtime.provider == "openai_compatible" and mode == "api" and not runtime.base_url.strip():
        warnings.append("OpenAI-compatible runtime is not pointed at a local base URL.")

    summary = (
        "Local inference stays on this machine."
        if mode == "local"
        else "Model prompts are sent to a remote API, but memory remains stored locally."
    )

    return {
        "schema_version": 1,
        "mode": mode,
        "mode_label": "local LLM" if mode == "local" else "API mode",
        "privacy_label": runtime.privacy_label,
        "local_first": mode == "local",
        "provider": runtime.provider,
        "provider_label": runtime.label,
        "preset_key": runtime.preset_key,
        "model": runtime.model,
        "base_url": runtime.base_url,
        "tool_strategy": runtime.tool_strategy,
        "endpoint_label": endpoint_label,
        "memory_storage": "local filesystem",
        "privacy_boundary": runtime.privacy_boundary,
        "summary": summary,
        "warnings": warnings,
        "is_inspectable": True,
    }


def format_runtime_privacy_status(status: dict[str, Any]) -> str:
    """Render a compact runtime/privacy snapshot as a readable string."""
    lines = [
        f"Runtime privacy status for {status.get('provider_label', 'runtime')}:",
        f"- Mode: {status.get('mode_label', 'unknown')} ({status.get('privacy_label', 'unknown')})",
        f"- Provider: {status.get('provider', '')}",
        f"- Model: {status.get('model', '')}",
        f"- Endpoint: {status.get('endpoint_label', '')}",
        f"- Memory: {status.get('memory_storage', 'local')}",
        f"- Boundary: {status.get('privacy_boundary', '')}",
    ]
    warnings = status.get("warnings", [])
    if warnings:
        lines.append("- Warnings: " + "; ".join(str(item) for item in warnings))
    lines.append(f"- Summary: {status.get('summary', '')}")
    return "\n".join(lines)


def build_runtime_preset_switch_preview(
    preset_key: str,
    runtime: RuntimeConfig | None = None,
) -> JsonDict:
    """Build a safe preview for a runtime preset switch."""
    return build_runtime_preset_switch_plan(preset_key, runtime)


def format_runtime_preset_switch_plan(plan: dict[str, Any]) -> str:
    """Render a runtime preset switch plan for UI display."""
    lines = [
        f"Runtime switch preview for {plan.get('resolved_preset_key', plan.get('requested_preset_key', 'preset'))}:",
        f"- Current mode: {plan.get('current', {}).get('mode', 'unknown')}",
        f"- Target mode: {plan.get('target', {}).get('mode', 'unknown')}",
        f"- Confirmed: {'yes' if plan.get('applied') else 'no'}",
    ]
    warnings = plan.get("warnings", [])
    if warnings:
        lines.append("- Warnings: " + "; ".join(str(item) for item in warnings))
    current = plan.get("current", {})
    target = plan.get("target", {})
    lines.append(
        "- Current runtime: "
        f"{current.get('provider', '')} / {current.get('label', '')} / {current.get('model', '')}"
    )
    lines.append(
        "- Target runtime: "
        f"{target.get('provider', '')} / {target.get('label', '')} / {target.get('model', '')}"
    )
    return "\n".join(lines)
