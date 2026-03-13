"""Attention Extraction Hook: Captures attention weights from Megatron forward pass.

This module provides a non-invasive mechanism to extract attention weights from
the Megatron-LM model during its standard forward pass. It registers PyTorch
forward hooks on attention layers, storing the weights in a global buffer that
can be read by the shadow mask generation logic.

Why hooks instead of modifying Megatron?
    Megatron's pipeline engine is complex and not designed to return attention
    weights. Modifying GPTModel.forward() would require changes across multiple
    internal files. Hooks are zero-intrusion: they attach externally and can be
    enabled/disabled at runtime.

Usage:
    # Before training step:
    hooks = register_attention_hooks(model)

    # During forward pass (happens automatically):
    output = model(input_ids=..., ...)

    # After forward, in loss function:
    attn_weights = get_captured_attention()  # dict of layer_idx -> (B, H, S, S)

    # Cleanup:
    remove_attention_hooks(hooks)
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Global attention buffer ─────────────────────────────────────────────
_ATTENTION_BUFFER: OrderedDict[int, torch.Tensor] = OrderedDict()
_HOOKS_REGISTERED = False
MAX_BUFFER_LAYERS = 128  # Prevent memory leak in long training runs


def clear_attention_buffer():
    """Clear the global attention buffer. Call before each forward pass."""
    global _ATTENTION_BUFFER
    _ATTENTION_BUFFER.clear()


def get_captured_attention() -> OrderedDict[int, torch.Tensor]:
    """Retrieve captured attention weights from the last forward pass.

    Returns:
        OrderedDict mapping layer_index -> attention weights tensor.
        Each tensor has shape (batch, num_heads, seq_len, seq_len).
        Returns empty dict if no attention was captured.
    """
    return _ATTENTION_BUFFER


def get_aggregated_attention() -> torch.Tensor | None:
    """Get attention weights aggregated across all layers.

    Averages attention across all captured layers, then averages across heads.
    This gives a (seq_len, seq_len) matrix representing overall token importance.

    Returns:
        Aggregated attention tensor (seq_len, seq_len) or None if no data.
    """
    if not _ATTENTION_BUFFER:
        return None

    # Stack all layers and average
    all_attn = list(_ATTENTION_BUFFER.values())
    # Average across layers, then across batch and heads
    stacked = torch.stack(all_attn, dim=0)  # (L, B, H, S, S)
    aggregated = stacked.mean(dim=(0, 1, 2))  # (S, S)
    return aggregated


def get_per_key_importance(
    num_recent_queries: int = 64,
) -> torch.Tensor | None:
    """Compute per-key importance scores from captured attention (SnapKV-style).

    For each key position, sum the attention it receives from the last
    `num_recent_queries` query positions, averaged across all heads and layers.
    This is the standard SnapKV importance metric.

    Args:
        num_recent_queries: Number of most recent query positions to use.
            SnapKV paper recommends using the observation window size.

    Returns:
        Importance scores tensor (seq_len,) or None if no attention captured.
    """
    if not _ATTENTION_BUFFER:
        return None

    # Use only the last few captured layers (they're most informative)
    layers = list(_ATTENTION_BUFFER.values())
    if not layers:
        return None

    # Use the last layer's attention (most relevant for token selection)
    attn = layers[-1]  # Could be (H, S, S), (B, H, S, S), or (S, S)

    # Guard against empty tensor (Issue #6)
    if attn is None or attn.numel() == 0:
        return None

    # Normalize to 3D: (H, S, S)
    if attn.dim() == 4:
        attn = attn[0]  # Take first batch element → (H, S, S)
    elif attn.dim() == 2:
        attn = attn.unsqueeze(0)  # (S, S) → (1, S, S)
    # Now attn is (H, S, S)

    # Take attention from the last num_recent_queries query positions
    seq_len = attn.shape[-1]
    start_q = max(0, seq_len - num_recent_queries)
    recent_attn = attn[:, start_q:, :]  # (H, Q, S)

    # Sum attention received by each key, averaged across heads
    importance = recent_attn.mean(dim=0).sum(dim=0)  # (S,)

    return importance.detach()


# ── Hook registration ───────────────────────────────────────────────────

def _make_attention_hook(layer_idx: int):
    """Create a forward hook that captures attention weights for a specific layer."""

    def hook_fn(module, args, output):
        # Limit buffer size to prevent memory leak (Issue #8)
        if len(_ATTENTION_BUFFER) >= MAX_BUFFER_LAYERS:
            _ATTENTION_BUFFER.popitem(last=False)  # Remove oldest

        if hasattr(module, "attention_probs") and module.attention_probs is not None:
            _ATTENTION_BUFFER[layer_idx] = module.attention_probs.detach()
        elif hasattr(module, "attn_weights") and module.attn_weights is not None:
            _ATTENTION_BUFFER[layer_idx] = module.attn_weights.detach()

    return hook_fn


def _find_attention_layers(model: nn.Module) -> list[tuple[int, nn.Module]]:
    """Find all attention layers in the Megatron model.

    Searches for modules that match Megatron's attention layer patterns:
    - CoreAttention (standard attention)
    - SelfAttention
    - TransformerLayer (contains self_attention)

    Returns:
        List of (layer_index, module) tuples.
    """
    attention_layers = []
    layer_idx = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__

        # Match Megatron's attention modules
        if module_type in ("CoreAttention", "DotProductAttention"):
            attention_layers.append((layer_idx, module))
            layer_idx += 1
        elif "SelfAttention" in module_type and not any(
            "SelfAttention" in type(child).__name__
            for child in module.children()
        ):
            attention_layers.append((layer_idx, module))
            layer_idx += 1

    return attention_layers


def register_attention_hooks(model: nn.Module) -> list:
    """Register forward hooks on all attention layers to capture weights.

    Args:
        model: Megatron GPTModel (or model chunk in pipeline parallelism)

    Returns:
        List of hook handles (for later removal)
    """
    global _HOOKS_REGISTERED
    clear_attention_buffer()

    hooks = []
    attention_layers = _find_attention_layers(model)

    if not attention_layers:
        logger.warning(
            "No attention layers found for hook registration. "
            "Falling back to position-based shadow mask."
        )
        return hooks

    for layer_idx, module in attention_layers:
        handle = module.register_forward_hook(_make_attention_hook(layer_idx))
        hooks.append(handle)

    _HOOKS_REGISTERED = True
    logger.info(f"Registered attention hooks on {len(hooks)} layers")
    return hooks


def remove_attention_hooks(hooks: list):
    """Remove previously registered attention hooks.

    Args:
        hooks: List of hook handles returned by register_attention_hooks
    """
    global _HOOKS_REGISTERED
    for handle in hooks:
        handle.remove()
    clear_attention_buffer()
    _HOOKS_REGISTERED = False
    logger.info(f"Removed {len(hooks)} attention hooks")
