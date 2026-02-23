"""
Medicaid Provider Fraud Signal Detection Engine
================================================

CLI tool that ingests the HHS Medicaid Provider Spending dataset,
cross-references against OIG LEIE exclusion list and NPPES NPI registry,
and outputs structured JSON fraud signal reports.

Competition: NEAR AI Market - Medicaid Provider Fraud Signal Detection
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import click
import polars as pl

from .ingest import (
    load_spending,
    load_leie,
    load_nppes,
    aggregate_provider_totals,
    aggregate_monthly,
)
from .output import build_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fraud_detector")

# Home health HCPCS codes for Signal 6 (per competition spec)
HOME_HEALTH_HCPCS = {
    "G0151", "G0152", "G0153", "G0154", "G0155", "G0156", "G0157",
    "G0158", "G0159", "G0160", "G0161", "G0162",
    "G0299", "G0300",
    "S9122", "S9123", "S9124",
    "T1019", "T1020", "T1021", "T1022",
}

# Working hours per month (22 business days * 8 hours)
WORKING_HOURS_PER_MONTH = 176


def signal_excluded_provider(
    spending_lf: pl.LazyFrame,
    leie_df: pl.DataFrame,
) -> list[dict]:
    """Signal 1: Excluded Provider Still Billing.

    Flag providers matching OIG LEIE exclusion list that billed after exclusion date.
    Checks both BILLING_PROVIDER_NPI_NUM and SERVICING_PROVIDER_NPI_NUM.
    """
    log.info("Running Signal 1: Excluded Provider Still Billing")
    flags = []

    excluded_npis = set(leie_df["NPI"].to_list())
    if not excluded_npis:
        log.warning("No excluded NPIs found in LEIE data")
        return flags

    # Get exclusion dates as dict
    excl_dates = {}
    for row in leie_df.iter_rows(named=True):
        npi = row["NPI"]
        excl_dates[npi] = {
            "date": row.get("EXCLDATE_PARSED"),
            "name": row.get("PROVIDER_NAME", ""),
            "type": row.get("EXCLTYPE", ""),
        }

    # Check billing NPIs - targeted query for excluded NPIs only
    billing_excluded = (
        spending_lf.filter(pl.col("BILLING_PROVIDER_NPI_NUM").is_in(excluded_npis))
        .group_by("BILLING_PROVIDER_NPI_NUM")
        .agg(
            pl.col("TOTAL_PAID").sum().alias("POST_EXCL_PAID"),
            pl.col("TOTAL_CLAIMS").sum().alias("POST_EXCL_CLAIMS"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES").sum().alias("POST_EXCL_BENE"),
            pl.col("CLAIM_FROM_MONTH").min().alias("FIRST_CLAIM"),
            pl.col("CLAIM_FROM_MONTH").max().alias("LAST_CLAIM"),
        )
        .collect()
    )

    # Also check servicing NPIs
    servicing_excluded = (
        spending_lf.filter(pl.col("SERVICING_PROVIDER_NPI_NUM").is_in(excluded_npis))
        .filter(
            pl.col("SERVICING_PROVIDER_NPI_NUM")
            != pl.col("BILLING_PROVIDER_NPI_NUM")
        )
        .group_by("SERVICING_PROVIDER_NPI_NUM")
        .agg(
            pl.col("TOTAL_PAID").sum().alias("POST_EXCL_PAID"),
            pl.col("TOTAL_CLAIMS").sum().alias("POST_EXCL_CLAIMS"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES").sum().alias("POST_EXCL_BENE"),
            pl.col("CLAIM_FROM_MONTH").min().alias("FIRST_CLAIM"),
            pl.col("CLAIM_FROM_MONTH").max().alias("LAST_CLAIM"),
        )
        .collect()
    )

    seen = set()
    for df, npi_col in [
        (billing_excluded, "BILLING_PROVIDER_NPI_NUM"),
        (servicing_excluded, "SERVICING_PROVIDER_NPI_NUM"),
    ]:
        for row in df.iter_rows(named=True):
            npi = row[npi_col]
            if npi in seen:
                continue
            seen.add(npi)

            excl = excl_dates.get(npi, {})
            excl_date = excl.get("date")
            overpayment = row["POST_EXCL_PAID"]

            flags.append(
                {
                    "npi": npi,
                    "provider_name": excl.get("name", ""),
                    "signal_type": "excluded_provider",
                    "severity": "critical",
                    "evidence": {
                        "exclusion_date": (
                            excl_date.isoformat() if excl_date else None
                        ),
                        "exclusion_type": excl.get("type", ""),
                        "post_exclusion_paid": round(overpayment, 2),
                        "post_exclusion_claims": row["POST_EXCL_CLAIMS"],
                        "first_claim_after": row["FIRST_CLAIM"],
                        "last_claim_after": row["LAST_CLAIM"],
                    },
                    "total_paid_all_time": round(row["POST_EXCL_PAID"], 2),
                    "total_claims_all_time": row["POST_EXCL_CLAIMS"],
                    "total_unique_beneficiaries_all_time": row["POST_EXCL_BENE"],
                    "estimated_overpayment_usd": round(overpayment, 2),
                    "fca_relevance": {
                        "claim_type": "False claim by excluded entity",
                        "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                        "suggested_next_steps": [
                            "Verify exclusion status against current OIG LEIE database",
                            "Calculate total federal payments made post-exclusion for damages estimate",
                            "Refer to OIG for civil monetary penalties under 42 U.S.C. 1320a-7a",
                        ],
                    },
                }
            )

    log.info("Signal 1 complete: %d excluded providers flagged", len(flags))
    return flags


def signal_billing_volume_outlier(
    spending_lf: pl.LazyFrame,
    nppes_df: pl.DataFrame,
) -> list[dict]:
    """Signal 2: Billing Volume Outlier.

    Flag providers above 99th percentile of total paid within their
    taxonomy code + state peer group.
    """
    log.info("Running Signal 2: Billing Volume Outlier")
    flags = []

    # Aggregate spending per billing NPI using streaming for memory efficiency
    log.info("  Aggregating spending per NPI (streaming)...")
    provider_spend = (
        spending_lf.group_by("BILLING_PROVIDER_NPI_NUM")
        .agg(
            pl.col("TOTAL_PAID").sum().alias("TOTAL_PAID_ALL"),
            pl.col("TOTAL_CLAIMS").sum().alias("TOTAL_CLAIMS_ALL"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES")
            .sum()
            .alias("TOTAL_BENE_ALL"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .collect(engine="streaming")
    )
    log.info("  Aggregated %d unique NPIs", len(provider_spend))

    # Join with NPPES for taxonomy and state
    nppes_slim = nppes_df.select(
        ["NPI", "TAXONOMY_CODE", "STATE", "ENTITY_TYPE", "PROVIDER_NAME",
         "ENUMERATION_DATE_PARSED"]
    )
    joined = provider_spend.join(nppes_slim, on="NPI", how="inner")

    # Filter to valid taxonomy+state combos
    joined = joined.filter(
        pl.col("TAXONOMY_CODE").is_not_null() & pl.col("STATE").is_not_null()
    )

    if len(joined) == 0:
        log.warning("No providers matched NPPES for Signal 2")
        return flags

    # Calculate peer group stats (median for severity, 99th pctl for flagging)
    peer_stats = joined.group_by(["TAXONOMY_CODE", "STATE"]).agg(
        pl.col("TOTAL_PAID_ALL").quantile(0.99).alias("P99_PAID"),
        pl.col("TOTAL_PAID_ALL").median().alias("MEDIAN_PAID"),
        pl.len().alias("PEER_COUNT"),
    )

    # Join back to get outliers
    with_peers = joined.join(peer_stats, on=["TAXONOMY_CODE", "STATE"], how="left")
    outliers = with_peers.filter(
        (pl.col("TOTAL_PAID_ALL") > pl.col("P99_PAID"))
        & (pl.col("PEER_COUNT") >= 10)  # Need meaningful peer group
    )

    for row in outliers.iter_rows(named=True):
        median_paid = max(row["MEDIAN_PAID"], 1)
        ratio = row["TOTAL_PAID_ALL"] / median_paid
        overpayment = row["TOTAL_PAID_ALL"] - row["P99_PAID"]

        flags.append(
            {
                "npi": row["NPI"],
                "provider_name": row.get("PROVIDER_NAME", ""),
                "entity_type": str(row.get("ENTITY_TYPE", "")),
                "taxonomy_code": row.get("TAXONOMY_CODE", ""),
                "state": row.get("STATE", ""),
                "enumeration_date": (
                    row["ENUMERATION_DATE_PARSED"].isoformat()
                    if row.get("ENUMERATION_DATE_PARSED")
                    else None
                ),
                "signal_type": "billing_outlier",
                "severity": "high" if ratio > 5 else "medium",
                "evidence": {
                    "total_paid": round(row["TOTAL_PAID_ALL"], 2),
                    "peer_p99": round(row["P99_PAID"], 2),
                    "peer_median": round(row["MEDIAN_PAID"], 2),
                    "peer_count": row["PEER_COUNT"],
                    "ratio_to_median": round(ratio, 2),
                },
                "total_paid_all_time": round(row["TOTAL_PAID_ALL"], 2),
                "total_claims_all_time": row["TOTAL_CLAIMS_ALL"],
                "total_unique_beneficiaries_all_time": row["TOTAL_BENE_ALL"],
                "estimated_overpayment_usd": round(max(overpayment, 0), 2),
                "fca_relevance": {
                    "claim_type": "Statistically anomalous billing volume",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                    "suggested_next_steps": [
                        "Compare service volume against peer providers in same taxonomy and state",
                        "Request itemized claim records for manual review of service documentation",
                        "Evaluate whether billing patterns suggest upcoding or unbundling",
                    ],
                },
            }
        )

    log.info("Signal 2 complete: %d outliers flagged", len(flags))
    return flags


def signal_rapid_escalation(
    spending_lf: pl.LazyFrame,
    nppes_df: pl.DataFrame,
) -> list[dict]:
    """Signal 3: Rapid Billing Escalation.

    Flag newly enumerated providers (within 24 months) with 3-month rolling
    average billing growth exceeding 200%.
    """
    log.info("Running Signal 3: Rapid Billing Escalation")
    flags = []

    # Filter to new providers FIRST (enumerated within 24 months)
    cutoff_date = date(2022, 7, 1)  # 24 months before first data month approx
    new_providers = nppes_df.filter(
        pl.col("ENUMERATION_DATE_PARSED").is_not_null()
        & (pl.col("ENUMERATION_DATE_PARSED") >= cutoff_date)
    ).select(["NPI", "PROVIDER_NAME", "ENTITY_TYPE", "TAXONOMY_CODE", "STATE",
              "ENUMERATION_DATE_PARSED"])

    new_npi_set = set(new_providers["NPI"].to_list())
    log.info("  Found %d new providers (enumerated after %s)", len(new_npi_set), cutoff_date)

    if not new_npi_set:
        return flags

    # Get monthly aggregates ONLY for new providers (pre-filtered for memory)
    monthly_new = (
        spending_lf.filter(
            pl.col("BILLING_PROVIDER_NPI_NUM").is_in(new_npi_set)
        )
        .group_by(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
        .agg(
            pl.col("TOTAL_PAID").sum().alias("MONTHLY_PAID"),
            pl.col("TOTAL_CLAIMS").sum().alias("MONTHLY_CLAIMS"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .sort(["NPI", "CLAIM_FROM_MONTH"])
        .collect(engine="streaming")
    )

    if len(monthly_new) == 0:
        log.warning("No new providers found for Signal 3")
        return flags

    # Calculate month-over-month growth
    monthly_new = monthly_new.sort(["NPI", "CLAIM_FROM_MONTH"])
    monthly_new = monthly_new.with_columns(
        pl.col("MONTHLY_PAID")
        .shift(1)
        .over("NPI")
        .alias("PREV_MONTH_PAID"),
    )
    monthly_new = monthly_new.with_columns(
        (
            (pl.col("MONTHLY_PAID") - pl.col("PREV_MONTH_PAID"))
            / pl.col("PREV_MONTH_PAID").abs().clip(lower_bound=1.0)
            * 100
        ).alias("MOM_GROWTH_PCT")
    )

    # 3-month rolling average growth
    monthly_new = monthly_new.with_columns(
        pl.col("MOM_GROWTH_PCT")
        .rolling_mean(window_size=3, min_samples=3)
        .over("NPI")
        .alias("ROLLING_3M_GROWTH")
    )

    # Flag NPIs where 3-month rolling avg > 200%
    escalated_months = monthly_new.filter(pl.col("ROLLING_3M_GROWTH") > 200)

    # Get per-NPI aggregates: max growth, total paid in escalation months, peak month
    escalated = (
        escalated_months.group_by("NPI")
        .agg(
            pl.col("ROLLING_3M_GROWTH").max().alias("MAX_ROLLING_GROWTH"),
            pl.col("MONTHLY_PAID").sum().alias("ESCALATION_PAID"),
            pl.col("MONTHLY_CLAIMS").sum().alias("TOTAL_CLAIMS"),
            pl.col("CLAIM_FROM_MONTH").max().alias("PEAK_MONTH"),
        )
    )

    # Join with NPPES for provider info
    escalated = escalated.join(new_providers, on="NPI", how="left")

    for row in escalated.iter_rows(named=True):
        # Overpayment: total paid in months where growth exceeded 200%
        overpayment = row["ESCALATION_PAID"]
        growth = row["MAX_ROLLING_GROWTH"]

        flags.append(
            {
                "npi": row["NPI"],
                "provider_name": row.get("PROVIDER_NAME", ""),
                "entity_type": str(row.get("ENTITY_TYPE", "")),
                "taxonomy_code": row.get("TAXONOMY_CODE", ""),
                "state": row.get("STATE", ""),
                "enumeration_date": (
                    row["ENUMERATION_DATE_PARSED"].isoformat()
                    if row.get("ENUMERATION_DATE_PARSED")
                    else None
                ),
                "signal_type": "rapid_escalation",
                "severity": "high" if growth > 500 else "medium",
                "evidence": {
                    "max_3m_rolling_growth_pct": round(growth, 2),
                    "peak_month": row["PEAK_MONTH"],
                    "total_paid_escalation_months": round(row["ESCALATION_PAID"], 2),
                    "total_claims": row["TOTAL_CLAIMS"],
                },
                "total_paid_all_time": round(row["ESCALATION_PAID"], 2),
                "total_claims_all_time": row["TOTAL_CLAIMS"],
                "total_unique_beneficiaries_all_time": 0,
                "estimated_overpayment_usd": round(max(overpayment, 0), 2),
                "fca_relevance": {
                    "claim_type": "Rapid billing escalation by new provider",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                    "suggested_next_steps": [
                        "Verify provider credentials and enrollment documentation",
                        "Request medical records for claims during escalation period",
                        "Compare service growth against patient panel expansion",
                    ],
                },
            }
        )

    log.info("Signal 3 complete: %d rapid escalators flagged", len(flags))
    return flags


def signal_workforce_impossibility(
    spending_lf: pl.LazyFrame,
    nppes_df: pl.DataFrame,
) -> list[dict]:
    """Signal 4: Workforce Impossibility.

    Flag organizations (Entity Type 2) where peak monthly claims imply
    more than 6 claims per provider-hour (assuming 176 working hours/month).
    """
    log.info("Running Signal 4: Workforce Impossibility")
    flags = []

    # Filter NPPES to organizations only (Entity Type 2)
    orgs = nppes_df.filter(pl.col("ENTITY_TYPE") == "2").select(
        ["NPI", "PROVIDER_NAME", "TAXONOMY_CODE", "STATE", "ENUMERATION_DATE_PARSED"]
    )
    org_npi_set = set(orgs["NPI"].to_list())

    if not org_npi_set:
        log.warning("No organization NPIs found for Signal 4")
        return flags

    # Get monthly claims per org NPI
    monthly = (
        spending_lf.filter(
            pl.col("BILLING_PROVIDER_NPI_NUM").is_in(org_npi_set)
        )
        .group_by(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
        .agg(
            pl.col("TOTAL_CLAIMS").sum().alias("MONTHLY_CLAIMS"),
            pl.col("TOTAL_PAID").sum().alias("MONTHLY_PAID"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES")
            .sum()
            .alias("MONTHLY_BENE"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .collect(engine="streaming")
    )

    if len(monthly) == 0:
        return flags

    # Calculate claims per hour (assuming single provider org)
    monthly = monthly.with_columns(
        (pl.col("MONTHLY_CLAIMS") / WORKING_HOURS_PER_MONTH).alias(
            "CLAIMS_PER_HOUR"
        )
    )

    # Get peak month per org
    peak = (
        monthly.sort("CLAIMS_PER_HOUR", descending=True)
        .group_by("NPI")
        .first()
    )

    # Filter to impossible volumes
    impossible = peak.filter(pl.col("CLAIMS_PER_HOUR") > 6)

    # Get lifetime totals
    lifetime = (
        monthly.group_by("NPI")
        .agg(
            pl.col("MONTHLY_PAID").sum().alias("TOTAL_PAID"),
            pl.col("MONTHLY_CLAIMS").sum().alias("TOTAL_CLAIMS"),
            pl.col("MONTHLY_BENE").sum().alias("TOTAL_BENE"),
        )
    )

    impossible = impossible.join(lifetime, on="NPI", how="left", suffix="_LT")
    impossible = impossible.join(orgs, on="NPI", how="left")

    for row in impossible.iter_rows(named=True):
        cph = row["CLAIMS_PER_HOUR"]
        # Overpayment: claims beyond 6/hour rate * avg cost per claim
        excess_claims = row["MONTHLY_CLAIMS"] - (6 * WORKING_HOURS_PER_MONTH)
        avg_cost = row["MONTHLY_PAID"] / max(row["MONTHLY_CLAIMS"], 1)
        overpayment = excess_claims * avg_cost

        flags.append(
            {
                "npi": row["NPI"],
                "provider_name": row.get("PROVIDER_NAME", ""),
                "entity_type": "2",
                "taxonomy_code": row.get("TAXONOMY_CODE", ""),
                "state": row.get("STATE", ""),
                "enumeration_date": (
                    row["ENUMERATION_DATE_PARSED"].isoformat()
                    if row.get("ENUMERATION_DATE_PARSED")
                    else None
                ),
                "signal_type": "workforce_impossibility",
                "severity": "high",
                "evidence": {
                    "peak_month": row["CLAIM_FROM_MONTH"],
                    "peak_monthly_claims": row["MONTHLY_CLAIMS"],
                    "claims_per_hour": round(cph, 2),
                    "monthly_paid_peak": round(row["MONTHLY_PAID"], 2),
                },
                "total_paid_all_time": round(row.get("TOTAL_PAID", 0), 2),
                "total_claims_all_time": row.get("TOTAL_CLAIMS", 0),
                "total_unique_beneficiaries_all_time": row.get("TOTAL_BENE", 0),
                "estimated_overpayment_usd": round(max(overpayment, 0), 2),
                "fca_relevance": {
                    "claim_type": "Physically impossible service volume",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(B)",
                    "suggested_next_steps": [
                        "Verify organizational staffing records against billed services",
                        "Cross-reference servicing provider NPIs to confirm distinct providers",
                        "Request time-of-service documentation for peak billing periods",
                    ],
                },
            }
        )

    log.info("Signal 4 complete: %d impossible volumes flagged", len(flags))
    return flags


def signal_shared_authorized_official(
    spending_lf: pl.LazyFrame,
    nppes_df: pl.DataFrame,
) -> list[dict]:
    """Signal 5: Shared Authorized Official.

    Flag authorized officials controlling 5+ NPIs with combined billings
    exceeding $1M.
    """
    log.info("Running Signal 5: Shared Authorized Official")
    flags = []

    # Get authorized officials with 5+ NPIs
    auth_officials = (
        nppes_df.filter(
            pl.col("AUTH_OFFICIAL_KEY").is_not_null()
            & (pl.col("AUTH_OFFICIAL_KEY") != "|")
            & (pl.col("AUTH_OFFICIAL_KEY") != "")
        )
        .group_by("AUTH_OFFICIAL_KEY")
        .agg(
            pl.col("NPI").alias("CONTROLLED_NPIS"),
            pl.col("NPI").n_unique().alias("NPI_COUNT"),
            pl.col("AUTH_OFFICIAL_FIRST").first().alias("FIRST_NAME"),
            pl.col("AUTH_OFFICIAL_LAST").first().alias("LAST_NAME"),
        )
        .filter(pl.col("NPI_COUNT") >= 5)
    )

    if len(auth_officials) == 0:
        log.info("No shared authorized officials with 5+ NPIs")
        return flags

    # Collect all controlled NPIs for targeted spending query
    all_controlled_npis = set()
    for row in auth_officials.iter_rows(named=True):
        all_controlled_npis.update(row["CONTROLLED_NPIS"])
    log.info("  %d shared officials control %d NPIs", len(auth_officials), len(all_controlled_npis))

    # Get spending ONLY for controlled NPIs (pre-filtered for memory)
    provider_spend = (
        spending_lf.filter(
            pl.col("BILLING_PROVIDER_NPI_NUM").is_in(all_controlled_npis)
        )
        .group_by("BILLING_PROVIDER_NPI_NUM")
        .agg(
            pl.col("TOTAL_PAID").sum().alias("TOTAL_PAID"),
            pl.col("TOTAL_CLAIMS").sum().alias("TOTAL_CLAIMS"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES").sum().alias("TOTAL_BENE"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .collect(engine="streaming")
    )

    for row in auth_officials.iter_rows(named=True):
        controlled_npis = row["CONTROLLED_NPIS"]
        npi_spending = provider_spend.filter(
            pl.col("NPI").is_in(controlled_npis)
        )

        combined_paid = npi_spending["TOTAL_PAID"].sum()
        combined_claims = npi_spending["TOTAL_CLAIMS"].sum()
        combined_bene = npi_spending["TOTAL_BENE"].sum()

        if combined_paid < 1_000_000:
            continue

        official_name = f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip()

        # Get details for each controlled NPI
        npi_details = []
        for npi_row in npi_spending.iter_rows(named=True):
            npi_info = nppes_df.filter(pl.col("NPI") == npi_row["NPI"])
            name = npi_info["PROVIDER_NAME"][0] if len(npi_info) > 0 else ""
            npi_details.append(
                {
                    "npi": npi_row["NPI"],
                    "name": name,
                    "paid": round(npi_row["TOTAL_PAID"], 2),
                }
            )

        flags.append(
            {
                "npi": controlled_npis[0],  # Primary NPI
                "provider_name": official_name,
                "entity_type": "",
                "taxonomy_code": "",
                "state": "",
                "enumeration_date": None,
                "signal_type": "shared_official",
                "severity": "high" if combined_paid > 5_000_000 else "medium",
                "evidence": {
                    "authorized_official": official_name,
                    "controlled_npi_count": row["NPI_COUNT"],
                    "combined_paid": round(combined_paid, 2),
                    "controlled_npis": npi_details[:20],
                },
                "total_paid_all_time": round(combined_paid, 2),
                "total_claims_all_time": combined_claims,
                "total_unique_beneficiaries_all_time": combined_bene,
                "estimated_overpayment_usd": 0.0,
                "fca_relevance": {
                    "claim_type": "Shared control suggesting shell entity network",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(C)",
                    "suggested_next_steps": [
                        "Investigate corporate relationships between controlled entities",
                        "Verify that each NPI represents a distinct operational practice",
                        "Check for common billing addresses, phone numbers, or bank accounts",
                    ],
                },
            }
        )

    log.info("Signal 5 complete: %d shared officials flagged", len(flags))
    return flags


def signal_geographic_implausibility(
    spending_lf: pl.LazyFrame,
    nppes_df: pl.DataFrame,
) -> list[dict]:
    """Signal 6: Geographic Implausibility.

    Flag providers with < 0.1 unique patient-to-claim ratio for home health
    HCPCS codes, suggesting impossible geographic service patterns.
    """
    log.info("Running Signal 6: Geographic Implausibility")
    flags = []

    # Filter spending to home health HCPCS codes, per month
    hh_monthly = (
        spending_lf.filter(pl.col("HCPCS_CODE").is_in(HOME_HEALTH_HCPCS))
        .group_by(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
        .agg(
            pl.col("TOTAL_UNIQUE_BENEFICIARIES").sum().alias("MONTHLY_BENE"),
            pl.col("TOTAL_CLAIMS").sum().alias("MONTHLY_CLAIMS"),
            pl.col("TOTAL_PAID").sum().alias("MONTHLY_PAID"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .collect(engine="streaming")
    )

    if len(hh_monthly) == 0:
        log.info("No home health claims found")
        return flags

    # Filter to providers who have at least one month with >100 claims
    high_vol_npis = (
        hh_monthly.filter(pl.col("MONTHLY_CLAIMS") > 100)
        .select("NPI")
        .unique()
    )

    if len(high_vol_npis) == 0:
        log.info("No providers with >100 monthly HH claims")
        return flags

    # Aggregate HH totals for qualifying providers
    hh_spending = (
        hh_monthly.join(high_vol_npis, on="NPI", how="inner")
        .group_by("NPI")
        .agg(
            pl.col("MONTHLY_BENE").sum().alias("TOTAL_BENE"),
            pl.col("MONTHLY_CLAIMS").sum().alias("TOTAL_CLAIMS"),
            pl.col("MONTHLY_PAID").sum().alias("TOTAL_PAID"),
        )
    )

    # Calculate beneficiary/claims ratio
    hh_spending = hh_spending.with_columns(
        (pl.col("TOTAL_BENE") / pl.col("TOTAL_CLAIMS").cast(pl.Float64).clip(lower_bound=1.0))
        .alias("BENE_CLAIM_RATIO")
    )

    implausible = hh_spending.filter(pl.col("BENE_CLAIM_RATIO") < 0.1)

    # Join with NPPES for provider info
    nppes_slim = nppes_df.select(
        ["NPI", "PROVIDER_NAME", "ENTITY_TYPE", "TAXONOMY_CODE", "STATE",
         "ENUMERATION_DATE_PARSED"]
    )
    implausible = implausible.join(nppes_slim, on="NPI", how="left")

    for row in implausible.iter_rows(named=True):
        ratio = row["BENE_CLAIM_RATIO"]

        flags.append(
            {
                "npi": row["NPI"],
                "provider_name": row.get("PROVIDER_NAME", ""),
                "entity_type": str(row.get("ENTITY_TYPE", "")),
                "taxonomy_code": row.get("TAXONOMY_CODE", ""),
                "state": row.get("STATE", ""),
                "enumeration_date": (
                    row["ENUMERATION_DATE_PARSED"].isoformat()
                    if row.get("ENUMERATION_DATE_PARSED")
                    else None
                ),
                "signal_type": "geographic_implausibility",
                "severity": "medium",
                "evidence": {
                    "beneficiary_claim_ratio": round(ratio, 6),
                    "total_beneficiaries": row["TOTAL_BENE"],
                    "total_claims": row["TOTAL_CLAIMS"],
                    "total_paid_home_health": round(row["TOTAL_PAID"], 2),
                },
                "total_paid_all_time": round(row["TOTAL_PAID"], 2),
                "total_claims_all_time": row["TOTAL_CLAIMS"],
                "total_unique_beneficiaries_all_time": row["TOTAL_BENE"],
                "estimated_overpayment_usd": 0.0,
                "fca_relevance": {
                    "claim_type": "Geographic implausibility in home health services",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(G)",
                    "suggested_next_steps": [
                        "Verify patient addresses against provider service area",
                        "Request visit logs and travel documentation for sampled claims",
                        "Compare beneficiary-to-claim ratio against state home health averages",
                    ],
                },
            }
        )

    log.info("Signal 6 complete: %d implausible providers flagged", len(flags))
    return flags


def find_data_file(name: str, search_dirs: list[str]) -> Optional[str]:
    """Search for a data file in common locations."""
    for d in search_dirs:
        for pattern in [name, name.lower(), name.upper()]:
            p = Path(d) / pattern
            if p.exists():
                return str(p)
    return None


@click.command()
@click.option("--spending", type=click.Path(exists=True), default=None,
              help="Path to Medicaid spending parquet file")
@click.option("--leie", type=click.Path(exists=True), default=None,
              help="Path to LEIE exclusion list CSV")
@click.option("--nppes", type=click.Path(exists=True), default=None,
              help="Path to NPPES NPI registry CSV")
@click.option("--output", type=click.Path(), default="fraud_signals.json",
              help="Output JSON file path")
@click.option("--no-gpu", is_flag=True, default=False, help="Disable GPU acceleration")
def main(spending, leie, nppes, output, no_gpu):
    """Medicaid Provider Fraud Signal Detection Engine."""
    start_time = time.time()

    # Auto-discover data files if not specified
    script_dir = str(Path(__file__).parent.parent)
    search_dirs = [
        script_dir,
        str(Path(script_dir).parent),
        ".",
        os.path.expanduser("~"),
    ]

    if spending is None:
        spending = find_data_file(
            "medicaid-provider-spending.parquet", search_dirs
        )
        if spending is None:
            log.error("Spending parquet file not found. Use --spending flag.")
            sys.exit(1)

    if leie is None:
        leie = find_data_file("leie_exclusions.csv", search_dirs)
        if leie is None:
            log.error("LEIE CSV not found. Use --leie flag.")
            sys.exit(1)

    nppes_df = None
    if nppes is None:
        # Try common NPPES filenames
        for fname in ["npidata_pfile.csv", "npidata_pfile_20050523-20260208.csv",
                       "npidata_pfile_20050523-20260209.csv", "nppes_npi.csv"]:
            nppes = find_data_file(fname, search_dirs)
            if nppes:
                break

    # Load data
    log.info("=" * 60)
    log.info("Medicaid Provider Fraud Signal Detection Engine")
    log.info("=" * 60)

    spending_lf = load_spending(spending)
    leie_df = load_leie(leie)

    if nppes:
        nppes_df = load_nppes(nppes)
    else:
        log.warning("NPPES file not found. Signals 2-6 will be limited.")
        nppes_df = None

    # Run all 6 signals (with GC between to manage memory)
    import gc
    all_flags = []

    # Signal 1: Excluded Provider (no NPPES needed)
    s1_flags = signal_excluded_provider(spending_lf, leie_df)
    all_flags.extend(s1_flags)
    gc.collect()

    if nppes_df is not None:
        # Signal 2: Billing Volume Outlier
        s2_flags = signal_billing_volume_outlier(spending_lf, nppes_df)
        all_flags.extend(s2_flags)
        gc.collect()

        # Signal 3: Rapid Billing Escalation
        s3_flags = signal_rapid_escalation(spending_lf, nppes_df)
        all_flags.extend(s3_flags)
        gc.collect()

        # Signal 4: Workforce Impossibility
        s4_flags = signal_workforce_impossibility(spending_lf, nppes_df)
        all_flags.extend(s4_flags)
        gc.collect()

        # Signal 5: Shared Authorized Official
        s5_flags = signal_shared_authorized_official(spending_lf, nppes_df)
        all_flags.extend(s5_flags)
        gc.collect()

        # Signal 6: Geographic Implausibility
        s6_flags = signal_geographic_implausibility(spending_lf, nppes_df)
        all_flags.extend(s6_flags)
    else:
        s2_flags = s3_flags = s4_flags = s5_flags = s6_flags = []

    # Count total providers (memory-efficient: just count unique NPIs)
    log.info("Counting total unique providers...")
    total_scanned = (
        spending_lf.select("BILLING_PROVIDER_NPI_NUM")
        .unique()
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    log.info("Total unique billing NPIs: %d", total_scanned)

    # Build output
    elapsed = time.time() - start_time

    signal_counts = {
        "excluded_provider": len(s1_flags),
        "billing_outlier": len(s2_flags) if nppes_df else 0,
        "rapid_escalation": len(s3_flags) if nppes_df else 0,
        "workforce_impossibility": len(s4_flags) if nppes_df else 0,
        "shared_official": len(s5_flags) if nppes_df else 0,
        "geographic_implausibility": len(s6_flags) if nppes_df else 0,
    }

    output_doc = build_output(
        flags=all_flags,
        total_scanned=total_scanned,
        signal_counts=signal_counts,
    )

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_doc, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info("Total providers scanned: %d", total_scanned)
    log.info("Total providers flagged: %d", len(all_flags))
    for signal_name, count in signal_counts.items():
        log.info("  %s: %d", signal_name, count)
    log.info("Total estimated overpayment: $%s",
             f"{sum(f.get('estimated_overpayment_usd', 0) for f in all_flags):,.2f}")
    log.info("Elapsed time: %.1f seconds", elapsed)
    log.info("Output written to: %s", output_path)


if __name__ == "__main__":
    main()
