"""
Output formatting module for the Medicaid Provider Fraud Signal Detection Engine.

Builds the structured JSON output conforming to the competition schema.
Groups flags by provider NPI with signals arrays.
"""

from datetime import datetime


TOOL_VERSION = "1.0.0"

# Mapping from internal signal names to competition-required names
SIGNAL_NAME_MAP = {
    "excluded_provider": "excluded_provider",
    "billing_outlier": "billing_outlier",
    "rapid_escalation": "rapid_escalation",
    "workforce_impossibility": "workforce_impossibility",
    "shared_official": "shared_official",
    "geographic_implausibility": "geographic_implausibility",
}


def build_output(
    flags: list[dict],
    total_scanned: int,
    signal_counts: dict[str, int],
) -> dict:
    """Build the final JSON output document matching competition schema.

    Groups flags by provider NPI into provider objects with signals arrays.
    """
    # Group flags by NPI
    providers_map: dict[str, dict] = {}

    for f in flags:
        npi = f.get("npi", "")
        signal_type = f.get("signal_type", "unknown")

        if npi not in providers_map:
            providers_map[npi] = {
                "npi": npi,
                "provider_name": f.get("provider_name", ""),
                "entity_type": _normalize_entity_type(f.get("entity_type", "")),
                "taxonomy_code": f.get("taxonomy_code", ""),
                "state": f.get("state", ""),
                "enumeration_date": f.get("enumeration_date"),
                "total_paid_all_time": f.get("total_paid_all_time", 0.0),
                "total_claims_all_time": f.get("total_claims_all_time", 0),
                "total_unique_beneficiaries_all_time": f.get(
                    "total_unique_beneficiaries_all_time", 0
                ),
                "signals": [],
                "estimated_overpayment_usd": 0.0,
                "fca_relevance": f.get("fca_relevance", {}),
            }
        else:
            # Update totals if this signal has better data
            p = providers_map[npi]
            if f.get("total_paid_all_time", 0) > p["total_paid_all_time"]:
                p["total_paid_all_time"] = f["total_paid_all_time"]
            if f.get("total_claims_all_time", 0) > p["total_claims_all_time"]:
                p["total_claims_all_time"] = f["total_claims_all_time"]
            if (
                f.get("total_unique_beneficiaries_all_time", 0)
                > p["total_unique_beneficiaries_all_time"]
            ):
                p["total_unique_beneficiaries_all_time"] = f[
                    "total_unique_beneficiaries_all_time"
                ]

        # Add signal to provider's signals array
        providers_map[npi]["signals"].append(
            {
                "signal_type": signal_type,
                "severity": f.get("severity", "medium"),
                "evidence": f.get("evidence", {}),
            }
        )

        # Accumulate overpayment
        providers_map[npi]["estimated_overpayment_usd"] += f.get(
            "estimated_overpayment_usd", 0.0
        )

        # Use highest-severity signal's fca_relevance
        severity_rank = {"critical": 3, "high": 2, "medium": 1}
        current_sev = providers_map[npi]["fca_relevance"].get("_severity_rank", 0)
        new_sev = severity_rank.get(f.get("severity", "medium"), 0)
        if new_sev > current_sev:
            fca = f.get("fca_relevance", {}).copy()
            fca["_severity_rank"] = new_sev
            providers_map[npi]["fca_relevance"] = fca

    # Clean up and round
    flagged_providers = []
    for p in providers_map.values():
        # Remove internal tracking field
        p["fca_relevance"].pop("_severity_rank", None)
        p["estimated_overpayment_usd"] = round(p["estimated_overpayment_usd"], 2)
        p["total_paid_all_time"] = round(p["total_paid_all_time"], 2)
        flagged_providers.append(p)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "tool_version": TOOL_VERSION,
        "total_providers_scanned": total_scanned,
        "total_providers_flagged": len(flagged_providers),
        "signal_counts": signal_counts,
        "flagged_providers": flagged_providers,
    }


def _normalize_entity_type(et: str) -> str:
    """Convert entity type code to human-readable string."""
    if et == "1":
        return "individual"
    elif et == "2":
        return "organization"
    return et if et else "unknown"
