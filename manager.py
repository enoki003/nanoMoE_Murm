from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self._routing_token_counts: Dict[int, torch.Tensor] = {}
        self._routing_target_assignments: Dict[int, float] = {}
        self._routing_calls: Dict[int, int] = defaultdict(int)
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []

    def reset_routing_stats(self):
        self._routing_token_counts = {}
        self._routing_target_assignments = {}
        self._routing_calls = defaultdict(int)
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_routing_stats(self, layer_idx: int, used_capacity: torch.Tensor, target_assignments: int):
        """Track how many (possibly weighted) tokens each expert processed.

        Args:
            layer_idx: Index of the MoE layer within the transformer stack.
            used_capacity: Tensor of shape [n_exp] with the number of tokens routed
                to each expert after capacity filtering.
            target_assignments: Expected total assignments (num_tokens * top_k) prior
                to capacity filtering; used to estimate dropped traffic.
        """

        counts = used_capacity.detach().to('cpu').to(torch.float64)
        if layer_idx not in self._routing_token_counts:
            self._routing_token_counts[layer_idx] = counts.clone()
            self._routing_target_assignments[layer_idx] = float(target_assignments)
        else:
            self._routing_token_counts[layer_idx] += counts
            self._routing_target_assignments[layer_idx] += float(target_assignments)

        self._routing_calls[layer_idx] += 1
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)

    def routing_summary(self) -> Dict[str, float]:
        """Return scalar metrics describing the router load balance per layer."""

        summary: Dict[str, float] = {}
        for layer_idx, counts in self._routing_token_counts.items():
            total_assignments = counts.sum().item()
            expected_assignments = self._routing_target_assignments.get(layer_idx, 0.0)
            if total_assignments <= 0:
                continue

            fractions = (counts / max(total_assignments, 1e-6)).to(torch.float32)
            layer_tag = f"layer{layer_idx:02d}"
            for expert_idx, frac in enumerate(fractions.tolist()):
                summary[f"routing/{layer_tag}/expert{expert_idx:02d}_fraction"] = frac

            summary[f"routing/{layer_tag}/load_std"] = float(fractions.std(unbiased=False))
            summary[f"routing/{layer_tag}/load_gini"] = float(self._gini(fractions))

            if expected_assignments > 0:
                dropped = max(0.0, 1.0 - (total_assignments / expected_assignments))
                summary[f"routing/{layer_tag}/drop_fraction"] = dropped

            summary[f"routing/{layer_tag}/forward_calls"] = float(self._routing_calls[layer_idx])

        return summary

    @staticmethod
    def _gini(fractions: torch.Tensor) -> float:
        """Compute the Gini coefficient of a probability vector."""

        if fractions.numel() == 0:
            return 0.0
        sorted_vals, _ = torch.sort(fractions)
        n = fractions.numel()
        index = torch.arange(1, n + 1, dtype=sorted_vals.dtype)
        numerator = torch.sum((2 * index - n - 1) * sorted_vals)
        denominator = n * sorted_vals.sum()
        if denominator.abs() < 1e-12:
            return 0.0
        return float(numerator / denominator)

MANAGER = MOEManager()