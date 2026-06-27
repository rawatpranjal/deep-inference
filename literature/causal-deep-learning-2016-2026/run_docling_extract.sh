#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p markdown_docling

extract downloads/*.pdf \
  -o markdown_docling \
  --backend docling \
  --split none \
  --jobs 1 \
  --timeout 600 \
  --memory-mb 4096 \
  2>&1 | tee docling_extract_20260627.txt
