#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

echo "=== Medicaid Provider Fraud Signal Detection Engine - Setup ==="

# Check Python version
PYTHON=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11+ required but not found."
    exit 1
fi
echo "Using Python: $PYTHON ($($PYTHON --version))"

# Create virtual environment if not exists
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate and install deps
source "$VENV_DIR/bin/activate"
echo "Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Download data files
mkdir -p "$DATA_DIR"

echo ""
echo "Downloading data files..."

# 1. Medicaid Provider Spending (2.9 GB parquet)
SPENDING="$DATA_DIR/medicaid-provider-spending.parquet"
if [ ! -f "$SPENDING" ]; then
    echo "  Downloading Medicaid Provider Spending dataset (~2.9 GB)..."
    curl -fSL --progress-bar \
        "https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet" \
        -o "$SPENDING"
else
    echo "  Medicaid Provider Spending: already downloaded"
fi

# 2. OIG LEIE Exclusion List
LEIE="$DATA_DIR/leie_exclusions.csv"
if [ ! -f "$LEIE" ]; then
    echo "  Downloading OIG LEIE exclusion list..."
    curl -fSL --progress-bar \
        "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv" \
        -o "$LEIE"
else
    echo "  LEIE exclusion list: already downloaded"
fi

# 3. NPPES NPI Registry (~1 GB zip)
NPPES_ZIP="$DATA_DIR/nppes.zip"
NPPES_CSV=""
# Check if any npidata_pfile CSV already exists
for f in "$DATA_DIR"/npidata_pfile*.csv; do
    if [ -f "$f" ]; then
        NPPES_CSV="$f"
        break
    fi
done

if [ -z "$NPPES_CSV" ]; then
    echo "  Downloading NPPES NPI Registry (~1 GB)..."
    curl -fSL --progress-bar \
        "https://download.cms.gov/nppes/NPPES_Data_Dissemination_February_2026_V2.zip" \
        -o "$NPPES_ZIP"
    echo "  Extracting NPPES CSV..."
    unzip -o -j "$NPPES_ZIP" "npidata_pfile_*.csv" -d "$DATA_DIR" 2>/dev/null || \
    unzip -o "$NPPES_ZIP" -d "$DATA_DIR" 2>/dev/null
    rm -f "$NPPES_ZIP"
    echo "  NPPES extracted."
else
    echo "  NPPES NPI Registry: already downloaded ($NPPES_CSV)"
fi

echo ""
echo "Setup complete."
echo "  Data directory: $DATA_DIR"
echo "  Run ./run.sh to execute the fraud detection engine."
