"""Potential exposure-route matching rule.

Task E species-route features are intentionally postponed. This rule therefore
emits disabled/missing masks and never depends on the species feature module.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput


class RouteAccessRule:
    """Disabled route-access placeholder until species route features exist."""

    name = "route_access"
    required_inputs = ["species_route_features", "chemical_route_descriptors"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {"rule_route_access": None}
        flags: dict[str, bool | str | None] = {
            "rule_route_access_applicable": False,
            "rule_route_access_missing_inputs": "species_route_features",
            "rule_route_mismatch_flag": None,
            "rule_route_access_disabled": True,
        }
        return RuleOutput(
            features,
            {},
            flags,
            "Route-access rule is disabled because species potential-route features are not yet available.",
        )
