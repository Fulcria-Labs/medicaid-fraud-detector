#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# Activate venv
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "=== Medicaid Provider Fraud Signal Detection Engine ==="
echo "Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Auto-discover data files
SPENDING_PATH="${SPENDING_PATH:-}"
LEIE_PATH="${LEIE_PATH:-}"
NPPES_PATH="${NPPES_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/fraud_signals.json}"

# Try data/ directory first, then parent directory
if [ -z "$SPENDING_PATH" ]; then
    if [ -f "$DATA_DIR/medicaid-provider-spending.parquet" ]; then
        SPENDING_PATH="$DATA_DIR/medicaid-provider-spending.parquet"
    fi
fi

if [ -z "$LEIE_PATH" ]; then
    if [ -f "$DATA_DIR/leie_exclusions.csv" ]; then
        LEIE_PATH="$DATA_DIR/leie_exclusions.csv"
    elif [ -f "$DATA_DIR/UPDATED.csv" ]; then
        LEIE_PATH="$DATA_DIR/UPDATED.csv"
    fi
fi

if [ -z "$NPPES_PATH" ]; then
    for f in "$DATA_DIR"/npidata_pfile*.csv; do
        if [ -f "$f" ]; then
            NPPES_PATH="$f"
            break
        fi
    done
fi

# Build CLI args
ARGS=()
if [ -n "$SPENDING_PATH" ]; then ARGS+=(--spending "$SPENDING_PATH"); fi
if [ -n "$LEIE_PATH" ]; then ARGS+=(--leie "$LEIE_PATH"); fi
if [ -n "$NPPES_PATH" ]; then ARGS+=(--nppes "$NPPES_PATH"); fi
ARGS+=(--output "$OUTPUT_PATH")

# Pass through any additional CLI args
ARGS+=("$@")

python3 -m src.signals "${ARGS[@]}"

echo ""
echo "Completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Output: $OUTPUT_PATH"
