"""
Evaluation metrics for constraint-preserving text editing.

Key metrics:
1. Constraint Fidelity: % of locked tokens preserved exactly
2. Edit Locality: How much change occurred in editable vs locked regions
3. Token-level statistics: Insertions, deletions, substitutions
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
import statistics

import torch


@dataclass
class EditMetrics:
    """Metrics for a single edit operation."""
    
    # Constraint fidelity
    constraint_fidelity: float  # % locked tokens preserved (should be 100%)
    num_locked_tokens: int
    num_locked_preserved: int
    
    # Edit statistics
    num_editable_tokens: int
    num_editable_changed: int
    edit_rate: float  # % editable tokens that changed
    
    # Sequence statistics
    original_length: int
    edited_length: int
    length_change: int
    
    # Token-level changes
    num_substitutions: int = 0
    num_insertions: int = 0
    num_deletions: int = 0
    
    # Drift (changes outside editable region - should be 0)
    drift: float = 0.0  # % of locked tokens that changed
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "constraint_fidelity": self.constraint_fidelity,
            "num_locked_tokens": self.num_locked_tokens,
            "num_locked_preserved": self.num_locked_preserved,
            "num_editable_tokens": self.num_editable_tokens,
            "num_editable_changed": self.num_editable_changed,
            "edit_rate": self.edit_rate,
            "original_length": self.original_length,
            "edited_length": self.edited_length,
            "length_change": self.length_change,
            "drift": self.drift,
        }


@dataclass
class BatchMetrics:
    """Aggregated metrics over a batch of edits."""
    
    num_samples: int
    
    # Constraint fidelity
    mean_constraint_fidelity: float
    min_constraint_fidelity: float
    max_constraint_fidelity: float
    perfect_constraint_rate: float  # % samples with 100% fidelity
    
    # Edit statistics
    mean_edit_rate: float
    std_edit_rate: float
    
    # Drift
    mean_drift: float
    zero_drift_rate: float  # % samples with 0% drift
    
    # Length changes
    mean_length_change: float
    
    # Individual metrics
    individual_metrics: List[EditMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without individual metrics)."""
        return {
            "num_samples": self.num_samples,
            "mean_constraint_fidelity": self.mean_constraint_fidelity,
            "min_constraint_fidelity": self.min_constraint_fidelity,
            "max_constraint_fidelity": self.max_constraint_fidelity,
            "perfect_constraint_rate": self.perfect_constraint_rate,
            "mean_edit_rate": self.mean_edit_rate,
            "std_edit_rate": self.std_edit_rate,
            "mean_drift": self.mean_drift,
            "zero_drift_rate": self.zero_drift_rate,
            "mean_length_change": self.mean_length_change,
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Evaluation Results ({self.num_samples} samples)",
            "=" * 50,
            "",
            "Constraint Fidelity:",
            f"  Mean:    {self.mean_constraint_fidelity * 100:.2f}%",
            f"  Min:     {self.min_constraint_fidelity * 100:.2f}%",
            f"  Max:     {self.max_constraint_fidelity * 100:.2f}%",
            f"  Perfect: {self.perfect_constraint_rate * 100:.1f}% of samples",
            "",
            "Edit Statistics:",
            f"  Mean edit rate:    {self.mean_edit_rate * 100:.1f}% ± {self.std_edit_rate * 100:.1f}%",
            f"  Mean length change: {self.mean_length_change:+.1f} tokens",
            "",
            "Drift (changes in locked region):",
            f"  Mean drift:     {self.mean_drift * 100:.2f}%",
            f"  Zero drift:     {self.zero_drift_rate * 100:.1f}% of samples",
        ]
        return "\n".join(lines)


def compute_constraint_fidelity(
    original_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    lock_mask: torch.Tensor,
) -> Tuple[float, int, int]:
    """
    Compute constraint fidelity: % of locked tokens preserved.
    
    Args:
        original_ids: Original token IDs [L] or [B, L]
        edited_ids: Edited token IDs [L] or [B, L]
        lock_mask: Lock mask [L] or [B, L] where True = locked
        
    Returns:
        Tuple of (fidelity, num_preserved, num_locked)
    """
    # Flatten if batched
    if original_ids.dim() > 1:
        original_ids = original_ids.flatten()
        edited_ids = edited_ids.flatten()
        lock_mask = lock_mask.flatten()
    
    # Get locked tokens
    locked_original = original_ids[lock_mask]
    locked_edited = edited_ids[lock_mask]
    
    num_locked = lock_mask.sum().item()
    if num_locked == 0:
        return 1.0, 0, 0
    
    num_preserved = (locked_original == locked_edited).sum().item()
    fidelity = num_preserved / num_locked
    
    return fidelity, num_preserved, num_locked


def compute_edit_distance(
    original_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    edit_mask: Optional[torch.Tensor] = None,
) -> Tuple[int, int, int]:
    """
    Compute simple edit statistics.
    
    Args:
        original_ids: Original tokens [L]
        edited_ids: Edited tokens [L]
        edit_mask: Optional mask for editable positions
        
    Returns:
        Tuple of (num_changed, num_same, total)
    """
    if edit_mask is not None:
        original_ids = original_ids[edit_mask]
        edited_ids = edited_ids[edit_mask]
    
    # Handle length differences
    min_len = min(len(original_ids), len(edited_ids))
    
    same = (original_ids[:min_len] == edited_ids[:min_len]).sum().item()
    
    # Tokens beyond the shared length are all different
    len_diff = abs(len(original_ids) - len(edited_ids))
    
    total = max(len(original_ids), len(edited_ids))
    changed = total - same
    
    return changed, same, total


def compute_edit_metrics(
    original_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    lock_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> EditMetrics:
    """
    Compute comprehensive edit metrics.
    
    Args:
        original_ids: Original token IDs [L]
        edited_ids: Edited token IDs [L]
        lock_mask: Lock mask [L] where True = locked
        attention_mask: Optional padding mask
        
    Returns:
        EditMetrics object
    """
    # Apply attention mask if provided
    if attention_mask is not None:
        valid = attention_mask.bool()
        original_ids = original_ids[valid]
        edited_ids = edited_ids[valid]
        lock_mask = lock_mask[valid]
    
    edit_mask = ~lock_mask
    
    # Constraint fidelity
    fidelity, num_preserved, num_locked = compute_constraint_fidelity(
        original_ids, edited_ids, lock_mask
    )
    
    # Drift (opposite of fidelity)
    drift = 1.0 - fidelity
    
    # Edit statistics for editable region
    num_editable = edit_mask.sum().item()
    if num_editable > 0:
        editable_original = original_ids[edit_mask]
        editable_edited = edited_ids[edit_mask]
        min_len = min(len(editable_original), len(editable_edited))
        num_changed = (editable_original[:min_len] != editable_edited[:min_len]).sum().item()
        num_changed += abs(len(editable_original) - len(editable_edited))
        edit_rate = num_changed / num_editable
    else:
        num_changed = 0
        edit_rate = 0.0
    
    return EditMetrics(
        constraint_fidelity=fidelity,
        num_locked_tokens=num_locked,
        num_locked_preserved=num_preserved,
        num_editable_tokens=num_editable,
        num_editable_changed=num_changed,
        edit_rate=edit_rate,
        original_length=len(original_ids),
        edited_length=len(edited_ids),
        length_change=len(edited_ids) - len(original_ids),
        drift=drift,
    )


def compute_batch_metrics(
    metrics_list: List[EditMetrics],
) -> BatchMetrics:
    """
    Aggregate metrics over a batch.
    
    Args:
        metrics_list: List of EditMetrics objects
        
    Returns:
        BatchMetrics summary
    """
    if not metrics_list:
        raise ValueError("Empty metrics list")
    
    n = len(metrics_list)
    
    fidelities = [m.constraint_fidelity for m in metrics_list]
    edit_rates = [m.edit_rate for m in metrics_list]
    drifts = [m.drift for m in metrics_list]
    length_changes = [m.length_change for m in metrics_list]
    
    return BatchMetrics(
        num_samples=n,
        mean_constraint_fidelity=statistics.mean(fidelities),
        min_constraint_fidelity=min(fidelities),
        max_constraint_fidelity=max(fidelities),
        perfect_constraint_rate=sum(1 for f in fidelities if f == 1.0) / n,
        mean_edit_rate=statistics.mean(edit_rates),
        std_edit_rate=statistics.stdev(edit_rates) if n > 1 else 0.0,
        mean_drift=statistics.mean(drifts),
        zero_drift_rate=sum(1 for d in drifts if d == 0.0) / n,
        mean_length_change=statistics.mean(length_changes),
        individual_metrics=metrics_list,
    )


def summarize_metrics(batch_metrics: BatchMetrics) -> str:
    """Generate a formatted summary string."""
    return batch_metrics.summary()


def print_metrics_table(metrics_list: List[EditMetrics], max_rows: int = 10):
    """Print a table of individual metrics."""
    print(f"\n{'#':>3} | {'Fidelity':>8} | {'Edit Rate':>9} | {'Drift':>6} | {'Len Δ':>5}")
    print("-" * 45)
    
    for i, m in enumerate(metrics_list[:max_rows]):
        fid_str = f"{m.constraint_fidelity * 100:.1f}%"
        edit_str = f"{m.edit_rate * 100:.1f}%"
        drift_str = f"{m.drift * 100:.1f}%"
        len_str = f"{m.length_change:+d}"
        
        print(f"{i+1:>3} | {fid_str:>8} | {edit_str:>9} | {drift_str:>6} | {len_str:>5}")
    
    if len(metrics_list) > max_rows:
        print(f"... and {len(metrics_list) - max_rows} more rows")

