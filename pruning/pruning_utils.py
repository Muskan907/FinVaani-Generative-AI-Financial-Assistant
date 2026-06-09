"""
pruning_utils.py — Helper functions for Lottery Ticket Hypothesis pruning.
"""

import copy
import json
import sqlite3
import torch
import torch.nn.utils.prune as prune


def get_lora_modules(model) -> list:
    """
    Return list of (name, module) tuples for all LoRA adapter layers.
    Looks for modules that have lora_A and lora_B attributes.
    """
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_modules.append((name, module))
    return lora_modules


def apply_magnitude_pruning(model, amount: float):
    """
    Apply L1 unstructured magnitude pruning to all LoRA adapter layers.

    Args:
        model:  PEFT model with LoRA adapters.
        amount: Fraction of REMAINING weights to prune (not cumulative).

    Returns:
        Modified model with pruning masks applied.
    """
    lora_modules = get_lora_modules(model)
    pruned_count = 0

    for name, module in lora_modules:
        # Prune lora_A weights
        for key in ["default"]:
            if key in module.lora_A:
                linear = module.lora_A[key]
                prune.l1_unstructured(linear, name="weight", amount=amount)
                pruned_count += 1
            if key in module.lora_B:
                linear = module.lora_B[key]
                prune.l1_unstructured(linear, name="weight", amount=amount)
                pruned_count += 1

    print(f"[Prune] Applied L1 pruning (amount={amount:.2f}) to {pruned_count} LoRA layers")
    return model


def get_sparsity_percent(model) -> float:
    """
    Calculate percentage of zero weights across all LoRA layers.

    Returns:
        Float between 0 and 100.
    """
    total_weights = 0
    zero_weights  = 0

    for name, module in get_lora_modules(model):
        for key in ["default"]:
            for lora_dict in [module.lora_A, module.lora_B]:
                if key in lora_dict:
                    w = lora_dict[key].weight
                    total_weights += w.numel()
                    zero_weights  += (w == 0).sum().item()

    if total_weights == 0:
        return 0.0
    return 100.0 * zero_weights / total_weights


def get_param_count(model) -> dict:
    """
    Return parameter count statistics for the model.

    Returns:
        Dict with total_params, trainable_params, lora_params, active_lora_params.
    """
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_params = 0
    active_lora_params = 0
    for name, module in get_lora_modules(model):
        for key in ["default"]:
            for lora_dict in [module.lora_A, module.lora_B]:
                if key in lora_dict:
                    w = lora_dict[key].weight
                    lora_params += w.numel()
                    active_lora_params += (w != 0).sum().item()

    return {
        "total_params":       total_params,
        "trainable_params":   trainable_params,
        "lora_params":        lora_params,
        "active_lora_params": active_lora_params,
    }


def save_pruning_mask(model, round_num: int, conn: sqlite3.Connection) -> None:
    """
    Save binary pruning masks for all LoRA layers to the SQLite database.

    Args:
        model:     PEFT model with pruning masks applied.
        round_num: Current pruning round number.
        conn:      Open SQLite connection.
    """
    rows = []
    for name, module in get_lora_modules(model):
        for key in ["default"]:
            for lora_key, lora_dict in [("lora_A", module.lora_A),
                                         ("lora_B", module.lora_B)]:
                if key in lora_dict:
                    linear = lora_dict[key]
                    w = linear.weight

                    # Get mask (1 = kept, 0 = pruned)
                    if hasattr(linear, "weight_mask"):
                        mask = linear.weight_mask.cpu().bool().tolist()
                    else:
                        mask = (w != 0).cpu().tolist()

                    layer_name = f"{name}.{lora_key}.{key}"
                    sparsity   = 100.0 * sum(
                        1 for row in mask for v in row if not v
                    ) / (w.numel() or 1)
                    active     = (w != 0).sum().item()

                    rows.append((
                        round_num,
                        layer_name,
                        round(sparsity, 2),
                        json.dumps(mask),
                        active,
                    ))

    conn.executemany(
        "INSERT INTO pruning_masks "
        "(pruning_round, layer_name, sparsity_percent, mask_json, param_count_remaining) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"[Prune] Saved {len(rows)} mask entries for round {round_num}")


def reset_to_original_weights(model, original_state_dict: dict):
    """
    KEY LTH STEP: Reset weight values to original (pre-training) values
    while keeping the current pruning mask applied.

    This is what makes it the Lottery Ticket Hypothesis:
    - The mask identifies WHICH weights to keep (the 'winning ticket')
    - The values are reset to the ORIGINAL initialisation
    - Only the unmasked (kept) weights are retrained

    Args:
        model:               Current pruned model.
        original_state_dict: State dict saved before any pruning.

    Returns:
        Model with original weight values but current sparsity pattern.
    """
    with torch.no_grad():
        for name, module in get_lora_modules(model):
            for key in ["default"]:
                for lora_key, lora_dict in [("lora_A", module.lora_A),
                                             ("lora_B", module.lora_B)]:
                    if key in lora_dict:
                        linear = lora_dict[key]
                        param_key = f"base_model.model.{name}.{lora_key}.{key}.weight"

                        # Try alternate key formats
                        orig_weight = None
                        for k in [param_key,
                                  f"{name}.{lora_key}.{key}.weight",
                                  f"model.{name}.{lora_key}.{key}.weight"]:
                            if k in original_state_dict:
                                orig_weight = original_state_dict[k]
                                break

                        if orig_weight is None:
                            continue

                        # Get current mask
                        if hasattr(linear, "weight_mask"):
                            mask = linear.weight_mask
                        else:
                            mask = (linear.weight != 0).float()

                        # Reset to original values, then re-apply mask
                        linear.weight.data.copy_(orig_weight.to(linear.weight.device))
                        linear.weight.data.mul_(mask)

    print("[Prune] Weights reset to original values (mask preserved) — LTH step complete")
    return model


def make_pruning_permanent(model):
    """
    Remove pruning re-parametrisation and make masks permanent.
    Call this before saving a pruned checkpoint.
    """
    for name, module in get_lora_modules(model):
        for key in ["default"]:
            for lora_dict in [module.lora_A, module.lora_B]:
                if key in lora_dict:
                    linear = lora_dict[key]
                    try:
                        prune.remove(linear, "weight")
                    except ValueError:
                        pass  # Already permanent
    return model
