#!/usr/bin/env bash
# Rebuild the current journal documents from the consolidated scientific source.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/build_scientific_reports.py
cp scientific-reports/scientific-reports-manuscript-inline.pdf paper.pdf
