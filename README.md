# Medicaid Provider Fraud Signal Detection Engine

A CLI tool that ingests the HHS Medicaid Provider Spending dataset (2.9 GB, 227M rows), cross-references providers against the OIG LEIE exclusion list and NPPES NPI registry, and outputs structured JSON fraud signal reports usable by qui tam / FCA lawyers.

## Requirements

- Python 3.11+
- Ubuntu 22.04+ or macOS 14+ (Apple Silicon supported)
- 16 GB+ RAM recommended (streams 227M rows via polars lazy evaluation)

## Quick Start

```bash
# 1. Setup (downloads data + installs deps)
chmod +x setup.sh run.sh
./setup.sh

# 2. Run
./run.sh

# Or with explicit data paths:
SPENDING_PATH=/path/to/medicaid-provider-spending.parquet \
LEIE_PATH=/path/to/leie_exclusions.csv \
NPPES_PATH=/path/to/npidata_pfile.csv \
./run.sh
```

Output: `fraud_signals.json`

## Signals Detected

| # | Signal | Type Key | Severity | FCA Statute | Description |
|---|--------|----------|----------|-------------|-------------|
| 1 | Excluded Provider Still Billing | `excluded_provider` | Critical | 3729(a)(1)(A) | Providers on OIG LEIE still submitting Medicaid claims |
| 2 | Billing Volume Outlier | `billing_outlier` | High if >5x median | 3729(a)(1)(A) | Providers above 99th percentile in taxonomy+state peer group |
| 3 | Rapid Billing Escalation | `rapid_escalation` | High if >500% growth | 3729(a)(1)(A) | New providers (<24 months) with >200% 3-month rolling growth |
| 4 | Workforce Impossibility | `workforce_impossibility` | High | 3729(a)(1)(B) | Organizations with >6 claims/provider-hour in peak month |
| 5 | Shared Authorized Official | `shared_official` | High if >$5M | 3729(a)(1)(C) | Officials controlling 5+ NPIs with >$1M combined billing |
| 6 | Geographic Implausibility | `geographic_implausibility` | Medium | 3729(a)(1)(G) | Home health providers with <0.1 beneficiary-to-claim ratio |

## Estimated Overpayment Formulas

| Signal | Formula |
|--------|---------|
| 1 | Total paid after exclusion date |
| 2 | provider_total - peer_99th_percentile (floored at 0) |
| 3 | Total paid in months where 3-month rolling growth exceeded 200% |
| 4 | (peak_claims - 6*8*22) * (peak_paid / peak_claims), floored at 0 |
| 5 | 0 (not estimated) |
| 6 | 0 (not estimated) |

## Output Schema

```json
{
  "generated_at": "ISO8601",
  "tool_version": "1.0.0",
  "total_providers_scanned": 617503,
  "total_providers_flagged": 9000,
  "signal_counts": {
    "excluded_provider": 1240,
    "billing_outlier": 5027,
    "rapid_escalation": 2948,
    "workforce_impossibility": 500,
    "shared_official": 100,
    "geographic_implausibility": 50
  },
  "flagged_providers": [
    {
      "npi": "1234567890",
      "provider_name": "Provider Name",
      "entity_type": "individual|organization",
      "taxonomy_code": "208D00000X",
      "state": "FL",
      "enumeration_date": "2020-01-15",
      "total_paid_all_time": 500000.00,
      "total_claims_all_time": 10000,
      "total_unique_beneficiaries_all_time": 500,
      "signals": [
        {
          "signal_type": "excluded_provider",
          "severity": "critical",
          "evidence": { "...signal-specific evidence..." }
        }
      ],
      "estimated_overpayment_usd": 500000.00,
      "fca_relevance": {
        "claim_type": "False claim by excluded entity",
        "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
        "suggested_next_steps": ["Step 1", "Step 2", "Step 3"]
      }
    }
  ]
}
```

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

13 tests covering all 6 signals plus output format validation.

## Architecture

```
submission/
  README.md              - This file
  requirements.txt       - Python dependencies (pip installable)
  setup.sh               - Downloads data + installs deps
  run.sh                 - Produces fraud_signals.json
  src/
    __init__.py          - Package init
    ingest.py            - Data loading (polars lazy evaluation)
    signals.py           - All 6 signal implementations + CLI
    output.py            - JSON report generation (groups by provider)
  tests/
    test_signals.py      - Unit tests with synthetic fixtures
  fraud_signals.json     - Sample output from actual data run
```

## Technical Notes

- Uses **polars** lazy evaluation with streaming engine for memory-efficient processing of the 227M-row dataset
- NPPES loading uses column selection (11 of 329 columns) to minimize memory footprint
- Each signal runs independently with `gc.collect()` between for memory management
- Supports `--no-gpu` flag (CPU-only by default, no GPU required)
- All file paths configurable via CLI flags or environment variables

## Data Sources

- **HHS Medicaid Provider Spending**: Centers for Medicare & Medicaid Services (2.9 GB parquet, Jan 2018 - Dec 2024)
- **OIG LEIE**: Office of Inspector General List of Excluded Individuals/Entities (82K records)
- **NPPES NPI**: National Plan and Provider Enumeration System (9.4M providers)
