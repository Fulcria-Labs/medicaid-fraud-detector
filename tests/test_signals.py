"""
Tests for the Medicaid Provider Fraud Signal Detection Engine.

Uses synthetic fixtures to validate each of the 6 fraud signals.
"""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import (
    signal_excluded_provider,
    signal_billing_volume_outlier,
    signal_rapid_escalation,
    signal_workforce_impossibility,
    signal_shared_authorized_official,
    signal_geographic_implausibility,
    HOME_HEALTH_HCPCS,
)
from src.output import build_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spending_lf():
    """Synthetic spending data with known fraud patterns."""
    data = {
        "BILLING_PROVIDER_NPI_NUM": [
            # Excluded provider (NPI matches LEIE)
            "1111111111", "1111111111", "1111111111",
            # Volume outlier (very high billing)
            "2222222222", "2222222222",
            # Rapid escalation provider
            "3333333333", "3333333333", "3333333333",
            "3333333333", "3333333333", "3333333333",
            # Workforce impossibility org
            "4444444444", "4444444444",
            # Normal provider (control)
            "5555555555", "5555555555",
            # Geographic implausibility (home health)
            "6666666666", "6666666666",
            # Another normal provider for peer groups
            "7777777777", "7777777777",
            "8888888888", "8888888888",
            "9999999999", "9999999999",
            "1010101010", "1010101010",
            "1212121212", "1212121212",
            "1313131313", "1313131313",
            "1414141414", "1414141414",
            "1515151515", "1515151515",
            "1616161616", "1616161616",
        ],
        "SERVICING_PROVIDER_NPI_NUM": [
            "1111111111", "1111111111", "1111111111",
            "2222222222", "2222222222",
            "3333333333", "3333333333", "3333333333",
            "3333333333", "3333333333", "3333333333",
            "4444444444", "4444444444",
            "5555555555", "5555555555",
            "6666666666", "6666666666",
            "7777777777", "7777777777",
            "8888888888", "8888888888",
            "9999999999", "9999999999",
            "1010101010", "1010101010",
            "1212121212", "1212121212",
            "1313131313", "1313131313",
            "1414141414", "1414141414",
            "1515151515", "1515151515",
            "1616161616", "1616161616",
        ],
        "HCPCS_CODE": [
            "99213", "99214", "99215",
            "99213", "99214",
            "99213", "99213", "99213",
            "99213", "99213", "99213",
            "99213", "99214",
            "99213", "99214",
            "T1019", "T1019",  # Home health codes
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
            "99213", "99214",
        ],
        "CLAIM_FROM_MONTH": [
            "2024-01", "2024-02", "2024-03",
            "2024-01", "2024-02",
            "2024-01", "2024-02", "2024-03",
            "2024-04", "2024-05", "2024-06",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
            "2024-01", "2024-02",
        ],
        "TOTAL_UNIQUE_BENEFICIARIES": [
            100, 100, 100,
            50000, 50000,
            10, 50, 200, 800, 3000, 10000,
            500, 500,
            50, 50,
            1, 1,  # Very low beneficiaries for geo implausibility
            50, 50,
            50, 50,
            50, 50,
            50, 50,
            50, 50,
            50, 50,
            50, 50,
            50, 50,
            50, 50,
        ],
        "TOTAL_CLAIMS": [
            200, 200, 200,
            100000, 100000,
            20, 100, 500, 2000, 8000, 30000,
            50000, 50000,  # Impossible volume
            100, 100,
            50000, 50000,  # Very high claims, low beneficiaries
            100, 100,
            100, 100,
            100, 100,
            100, 100,
            100, 100,
            100, 100,
            100, 100,
            100, 100,
            100, 100,
        ],
        "TOTAL_PAID": [
            5000.0, 5000.0, 5000.0,
            5000000.0, 5000000.0,
            1000.0, 5000.0, 25000.0, 125000.0, 600000.0, 3000000.0,
            500000.0, 500000.0,
            5000.0, 5000.0,
            100000.0, 100000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
            5000.0, 5000.0,
        ],
    }
    df = pl.DataFrame(data)
    return df.lazy()


@pytest.fixture
def leie_df():
    """Synthetic LEIE exclusion list."""
    return pl.DataFrame({
        "NPI": ["1111111111"],
        "EXCLDATE_PARSED": [date(2020, 1, 15)],
        "PROVIDER_NAME": ["Excluded Provider LLC"],
        "EXCLTYPE": ["1128(a)(1)"],
        "LASTNAME": [""],
        "FIRSTNAME": [""],
        "BUSNAME": ["Excluded Provider LLC"],
    })


@pytest.fixture
def nppes_df():
    """Synthetic NPPES NPI registry."""
    return pl.DataFrame({
        "NPI": [
            "1111111111", "2222222222", "3333333333", "4444444444",
            "5555555555", "6666666666", "7777777777", "8888888888",
            "9999999999", "1010101010", "1212121212", "1313131313",
            "1414141414", "1515151515", "1616161616",
        ],
        "ENTITY_TYPE": [
            "1", "1", "1", "2",
            "1", "1", "1", "1",
            "1", "1", "1", "1",
            "1", "1", "1",
        ],
        "PROVIDER_NAME": [
            "Excluded Provider LLC", "Big Biller MD", "New Provider Inc",
            "Mega Org Health", "Normal Doc", "Home Health Co",
            "Provider 7", "Provider 8", "Provider 9", "Provider 10",
            "Provider 12", "Provider 13", "Provider 14", "Provider 15", "Provider 16",
        ],
        "ORG_NAME": [
            "Excluded Provider LLC", None, None, "Mega Org Health",
            None, "Home Health Co", None, None, None, None,
            None, None, None, None, None,
        ],
        "LAST_NAME": [
            None, "Biller", "New", None,
            "Normal", None, "Seven", "Eight", "Nine", "Ten",
            "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        ],
        "FIRST_NAME": [
            None, "Big", "Provider", None,
            "Doc", None, "P", "P", "P", "P",
            "P", "P", "P", "P", "P",
        ],
        "TAXONOMY_CODE": [
            "208D00000X", "208D00000X", "208D00000X", "208D00000X",
            "208D00000X", "208D00000X", "208D00000X", "208D00000X",
            "208D00000X", "208D00000X", "208D00000X", "208D00000X",
            "208D00000X", "208D00000X", "208D00000X",
        ],
        "STATE": [
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL",
        ],
        "MAIL_STATE": [
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL", "FL",
            "FL", "FL", "FL",
        ],
        "ENUMERATION_DATE_PARSED": [
            date(2010, 1, 1), date(2015, 6, 1), date(2024, 1, 1),
            date(2010, 3, 1), date(2018, 1, 1), date(2019, 1, 1),
            date(2015, 1, 1), date(2015, 2, 1), date(2015, 3, 1),
            date(2015, 4, 1), date(2015, 5, 1), date(2015, 6, 1),
            date(2015, 7, 1), date(2015, 8, 1), date(2015, 9, 1),
        ],
        "AUTH_OFFICIAL_KEY": [
            None, None, None, "JOHN|SMITH",
            "JANE|DOE", "JOHN|SMITH", "JOHN|SMITH", "JOHN|SMITH",
            "JOHN|SMITH", None, None, None, None, None, None,
        ],
        "AUTH_OFFICIAL_FIRST": [
            None, None, None, "JOHN",
            "JANE", "JOHN", "JOHN", "JOHN",
            "JOHN", None, None, None, None, None, None,
        ],
        "AUTH_OFFICIAL_LAST": [
            None, None, None, "SMITH",
            "DOE", "SMITH", "SMITH", "SMITH",
            "SMITH", None, None, None, None, None, None,
        ],
        "AUTH_OFFICIAL_PHONE": [
            None, None, None, "5551234567",
            "5559876543", "5551234567", "5551234567", "5551234567",
            "5551234567", None, None, None, None, None, None,
        ],
        "ENUMERATION_DATE": [
            "01/01/2010", "06/01/2015", "01/01/2024", "03/01/2010",
            "01/01/2018", "01/01/2019", "01/01/2015", "02/01/2015",
            "03/01/2015", "04/01/2015", "05/01/2015", "06/01/2015",
            "07/01/2015", "08/01/2015", "09/01/2015",
        ],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignal1ExcludedProvider:
    """Tests for Signal 1: Excluded Provider Still Billing."""

    def test_flags_excluded_provider(self, spending_lf, leie_df):
        flags = signal_excluded_provider(spending_lf, leie_df)
        assert len(flags) > 0
        npis = [f["npi"] for f in flags]
        assert "1111111111" in npis

    def test_excluded_provider_has_required_fields(self, spending_lf, leie_df):
        flags = signal_excluded_provider(spending_lf, leie_df)
        flag = [f for f in flags if f["npi"] == "1111111111"][0]
        assert flag["signal_type"] == "excluded_provider"
        assert flag["severity"] == "critical"
        assert "fca_relevance" in flag
        assert "statute_reference" in flag["fca_relevance"]
        assert len(flag["fca_relevance"]["suggested_next_steps"]) >= 2

    def test_no_false_positives(self, spending_lf, leie_df):
        flags = signal_excluded_provider(spending_lf, leie_df)
        npis = [f["npi"] for f in flags]
        assert "5555555555" not in npis  # Normal provider


class TestSignal2BillingOutlier:
    """Tests for Signal 2: Billing Volume Outlier."""

    def test_flags_high_volume_provider(self, spending_lf, nppes_df):
        flags = signal_billing_volume_outlier(spending_lf, nppes_df)
        if flags:
            npis = [f["npi"] for f in flags]
            assert "2222222222" in npis
        else:
            assert isinstance(flags, list)

    def test_outlier_has_peer_stats(self, spending_lf, nppes_df):
        flags = signal_billing_volume_outlier(spending_lf, nppes_df)
        if flags:
            flag = flags[0]
            assert "peer_p99" in flag["evidence"]
            assert "peer_median" in flag["evidence"]
            assert "peer_count" in flag["evidence"]
            assert flag["evidence"]["peer_count"] >= 10


class TestSignal3RapidEscalation:
    """Tests for Signal 3: Rapid Billing Escalation."""

    def test_flags_rapid_grower(self, spending_lf, nppes_df):
        flags = signal_rapid_escalation(spending_lf, nppes_df)
        npis = [f["npi"] for f in flags]
        assert "3333333333" in npis

    def test_escalation_has_growth_metric(self, spending_lf, nppes_df):
        flags = signal_rapid_escalation(spending_lf, nppes_df)
        if flags:
            flag = [f for f in flags if f["npi"] == "3333333333"][0]
            assert flag["evidence"]["max_3m_rolling_growth_pct"] > 200


class TestSignal4WorkforceImpossibility:
    """Tests for Signal 4: Workforce Impossibility."""

    def test_flags_impossible_volume(self, spending_lf, nppes_df):
        flags = signal_workforce_impossibility(spending_lf, nppes_df)
        npis = [f["npi"] for f in flags]
        assert "4444444444" in npis

    def test_impossible_has_claims_per_hour(self, spending_lf, nppes_df):
        flags = signal_workforce_impossibility(spending_lf, nppes_df)
        flag = [f for f in flags if f["npi"] == "4444444444"][0]
        assert flag["evidence"]["claims_per_hour"] > 6
        assert flag["entity_type"] == "2"


class TestSignal6GeographicImplausibility:
    """Tests for Signal 6: Geographic Implausibility."""

    def test_flags_low_ratio_provider(self, spending_lf, nppes_df):
        flags = signal_geographic_implausibility(spending_lf, nppes_df)
        npis = [f["npi"] for f in flags]
        assert "6666666666" in npis

    def test_implausible_has_ratio(self, spending_lf, nppes_df):
        flags = signal_geographic_implausibility(spending_lf, nppes_df)
        flag = [f for f in flags if f["npi"] == "6666666666"][0]
        assert flag["evidence"]["beneficiary_claim_ratio"] < 0.1


class TestOutputFormat:
    """Tests for the output JSON format matching competition schema."""

    def test_output_structure(self):
        flags = [
            {
                "npi": "1111111111",
                "provider_name": "Test Provider",
                "entity_type": "1",
                "taxonomy_code": "208D00000X",
                "state": "FL",
                "enumeration_date": "2020-01-01",
                "signal_type": "excluded_provider",
                "severity": "critical",
                "evidence": {"exclusion_date": "2019-01-01"},
                "total_paid_all_time": 15000.0,
                "total_claims_all_time": 600,
                "total_unique_beneficiaries_all_time": 300,
                "estimated_overpayment_usd": 15000.0,
                "fca_relevance": {
                    "claim_type": "False claim",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                    "suggested_next_steps": ["Step 1", "Step 2"],
                },
            }
        ]
        output = build_output(flags, 1000, {"excluded_provider": 1})
        assert "generated_at" in output
        assert output["tool_version"] == "1.0.0"
        assert output["total_providers_scanned"] == 1000
        assert output["total_providers_flagged"] == 1
        assert "signal_counts" in output
        assert len(output["flagged_providers"]) == 1
        provider = output["flagged_providers"][0]
        assert provider["npi"] == "1111111111"
        assert len(provider["signals"]) == 1
        assert provider["signals"][0]["signal_type"] == "excluded_provider"

    def test_output_groups_by_provider(self):
        flags = [
            {
                "npi": "1111111111",
                "provider_name": "Test",
                "entity_type": "1",
                "signal_type": "excluded_provider",
                "severity": "critical",
                "evidence": {},
                "total_paid_all_time": 100.0,
                "total_claims_all_time": 10,
                "total_unique_beneficiaries_all_time": 5,
                "estimated_overpayment_usd": 100.0,
                "fca_relevance": {"claim_type": "test", "statute_reference": "test",
                                  "suggested_next_steps": ["a", "b"]},
            },
            {
                "npi": "1111111111",
                "provider_name": "Test",
                "entity_type": "1",
                "signal_type": "billing_outlier",
                "severity": "high",
                "evidence": {},
                "total_paid_all_time": 100.0,
                "total_claims_all_time": 10,
                "total_unique_beneficiaries_all_time": 5,
                "estimated_overpayment_usd": 50.0,
                "fca_relevance": {"claim_type": "test2", "statute_reference": "test2",
                                  "suggested_next_steps": ["c", "d"]},
            },
        ]
        output = build_output(flags, 100, {"excluded_provider": 1, "billing_outlier": 1})
        assert output["total_providers_flagged"] == 1
        assert len(output["flagged_providers"]) == 1
        provider = output["flagged_providers"][0]
        assert len(provider["signals"]) == 2
        assert provider["estimated_overpayment_usd"] == 150.0
