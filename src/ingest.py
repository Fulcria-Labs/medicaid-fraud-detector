"""
Data ingestion module for the Medicaid Provider Fraud Signal Detection Engine.

Loads and pre-processes:
- HHS Medicaid Provider Spending dataset (2.9GB parquet, ~227M rows)
- OIG LEIE exclusion list (CSV)
- NPPES NPI registry (CSV)

Uses polars lazy evaluation for memory-efficient processing.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

log = logging.getLogger("fraud_detector.ingest")


def load_spending(path: str) -> pl.LazyFrame:
    """Load Medicaid Provider Spending parquet as a lazy frame.

    Expected columns:
        BILLING_PROVIDER_NPI_NUM, SERVICING_PROVIDER_NPI_NUM,
        HCPCS_CODE, CLAIM_FROM_MONTH (YYYY-MM),
        TOTAL_UNIQUE_BENEFICIARIES, TOTAL_CLAIMS, TOTAL_PAID
    """
    log.info("Loading spending data from %s", path)
    lf = pl.scan_parquet(path)
    log.info("Spending schema: %s", lf.collect_schema())
    return lf


def load_leie(path: str) -> pl.DataFrame:
    """Load OIG LEIE exclusion list.

    Returns DataFrame with columns: NPI (str), EXCLDATE (Date), provider name fields.
    Filters to rows that have a valid NPI.
    """
    log.info("Loading LEIE exclusion list from %s", path)
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", " "],
    )
    # Keep only rows with valid NPI
    df = df.filter(
        pl.col("NPI").is_not_null()
        & (pl.col("NPI").cast(pl.Utf8).str.len_chars() == 10)
    )
    # Parse EXCLDATE from integer YYYYMMDD to date
    df = df.with_columns(
        pl.col("NPI").cast(pl.Utf8).alias("NPI"),
        pl.col("EXCLDATE")
        .cast(pl.Utf8)
        .str.to_date("%Y%m%d", strict=False)
        .alias("EXCLDATE_PARSED"),
    )
    # Build provider name
    df = df.with_columns(
        pl.when(pl.col("BUSNAME").is_not_null() & (pl.col("BUSNAME") != ""))
        .then(pl.col("BUSNAME"))
        .otherwise(
            pl.concat_str(
                [pl.col("FIRSTNAME"), pl.col("LASTNAME")],
                separator=" ",
                ignore_nulls=True,
            )
        )
        .alias("PROVIDER_NAME")
    )
    log.info("LEIE loaded: %d excluded providers with valid NPI", len(df))
    return df


def load_nppes(path: str) -> pl.DataFrame:
    """Load NPPES NPI registry CSV.

    Returns DataFrame with key columns for fraud signals:
    - NPI, Entity Type Code, Provider Organization Name, Provider Last Name,
      Provider First Name, Provider Business Practice Location Address State,
      Healthcare Provider Taxonomy Code_1, Enumeration Date,
      Authorized Official Last Name, Authorized Official First Name
    """
    log.info("Loading NPPES NPI registry from %s", path)

    # Only load columns we need to minimize memory
    needed_cols = [
        "NPI",
        "Entity Type Code",
        "Provider Organization Name (Legal Business Name)",
        "Provider Last Name (Legal Name)",
        "Provider First Name",
        "Provider Business Mailing Address State Name",
        "Provider Business Practice Location Address State Name",
        "Healthcare Provider Taxonomy Code_1",
        "Provider Enumeration Date",
        "Authorized Official Last Name",
        "Authorized Official First Name",
        "Authorized Official Telephone Number",
    ]

    df = pl.read_csv(
        path,
        columns=needed_cols,
        infer_schema_length=10000,
        null_values=["", " "],
        dtypes={
            "NPI": pl.Utf8,
            "Authorized Official Telephone Number": pl.Utf8,
            "Entity Type Code": pl.Utf8,
        },
    )

    # Rename for easier access
    rename_map = {
        "Entity Type Code": "ENTITY_TYPE",
        "Provider Organization Name (Legal Business Name)": "ORG_NAME",
        "Provider Last Name (Legal Name)": "LAST_NAME",
        "Provider First Name": "FIRST_NAME",
        "Provider Business Practice Location Address State Name": "STATE",
        "Provider Business Mailing Address State Name": "MAIL_STATE",
        "Healthcare Provider Taxonomy Code_1": "TAXONOMY_CODE",
        "Provider Enumeration Date": "ENUMERATION_DATE",
        "Authorized Official Last Name": "AUTH_OFFICIAL_LAST",
        "Authorized Official First Name": "AUTH_OFFICIAL_FIRST",
        "Authorized Official Telephone Number": "AUTH_OFFICIAL_PHONE",
    }
    df = df.rename(rename_map)

    # Parse enumeration date
    df = df.with_columns(
        pl.col("ENUMERATION_DATE")
        .str.to_date("%m/%d/%Y", strict=False)
        .alias("ENUMERATION_DATE_PARSED"),
    )

    # Build provider name
    df = df.with_columns(
        pl.when(pl.col("ORG_NAME").is_not_null() & (pl.col("ORG_NAME") != ""))
        .then(pl.col("ORG_NAME"))
        .otherwise(
            pl.concat_str(
                [pl.col("FIRST_NAME"), pl.col("LAST_NAME")],
                separator=" ",
                ignore_nulls=True,
            )
        )
        .alias("PROVIDER_NAME"),
        # Build authorized official key for Signal 5
        pl.concat_str(
            [
                pl.col("AUTH_OFFICIAL_FIRST").str.to_uppercase(),
                pl.col("AUTH_OFFICIAL_LAST").str.to_uppercase(),
            ],
            separator="|",
            ignore_nulls=True,
        ).alias("AUTH_OFFICIAL_KEY"),
    )

    # Use practice state, fall back to mailing state
    df = df.with_columns(
        pl.when(pl.col("STATE").is_not_null())
        .then(pl.col("STATE"))
        .otherwise(pl.col("MAIL_STATE"))
        .alias("STATE")
    )

    log.info("NPPES loaded: %d providers", len(df))
    return df


def aggregate_provider_totals(spending_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate spending data to per-NPI lifetime totals.

    Returns lazy frame with columns:
        NPI, TOTAL_PAID_ALL_TIME, TOTAL_CLAIMS_ALL_TIME,
        TOTAL_UNIQUE_BENEFICIARIES_ALL_TIME
    """
    # Aggregate on billing NPI
    return (
        spending_lf.group_by("BILLING_PROVIDER_NPI_NUM")
        .agg(
            pl.col("TOTAL_PAID").sum().alias("TOTAL_PAID_ALL_TIME"),
            pl.col("TOTAL_CLAIMS").sum().alias("TOTAL_CLAIMS_ALL_TIME"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES")
            .sum()
            .alias("TOTAL_UNIQUE_BENEFICIARIES_ALL_TIME"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
    )


def aggregate_monthly(spending_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate spending data per NPI per month.

    Returns lazy frame with columns:
        NPI, CLAIM_FROM_MONTH, MONTHLY_PAID, MONTHLY_CLAIMS, MONTHLY_BENEFICIARIES
    """
    return (
        spending_lf.group_by(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
        .agg(
            pl.col("TOTAL_PAID").sum().alias("MONTHLY_PAID"),
            pl.col("TOTAL_CLAIMS").sum().alias("MONTHLY_CLAIMS"),
            pl.col("TOTAL_UNIQUE_BENEFICIARIES")
            .sum()
            .alias("MONTHLY_BENEFICIARIES"),
        )
        .rename({"BILLING_PROVIDER_NPI_NUM": "NPI"})
        .sort(["NPI", "CLAIM_FROM_MONTH"])
    )
