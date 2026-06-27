#!/usr/bin/env python3
"""Build a catalog for searchable Markdown and TeX artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.csv"
MARKDOWN_DIR = ROOT / "markdown_docling"
TEX_DIR = ROOT / "tex_sources" / "extracted"
OUT_CSV = ROOT / "searchable_catalog.csv"
OUT_JSON = ROOT / "searchable_catalog.json"
OUT_README = ROOT / "SEARCH.md"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_lookup() -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not MARKDOWN_DIR.exists():
        return out
    for path in MARKDOWN_DIR.glob("*.md"):
        out[normalize(path.stem)] = path
    return out


def match_markdown(row: dict[str, str], lookup: dict[str, Path]) -> Path | None:
    candidates = []
    pdf_path = row.get("pdf_path", "")
    if pdf_path:
        candidates.append(normalize(Path(pdf_path).stem))
    candidates.append(normalize(row.get("title", "")))
    candidates.append(normalize(row.get("key", "")))
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for key, path in lookup.items():
        if any(candidate and (candidate in key or key in candidate) for candidate in candidates):
            return path
    return None


def tex_stats(arxiv_id: str) -> tuple[str, int, int]:
    if not arxiv_id:
        return "", 0, 0
    source_dir = TEX_DIR / arxiv_id
    if not source_dir.exists():
        return "", 0, 0
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    tex_files = [path for path in files if path.suffix.lower() in {".tex", ".ltx", ".latex"}]
    return str(source_dir.relative_to(ROOT)), len(files), len(tex_files)


def main() -> int:
    rows = read_manifest()
    md_lookup = markdown_lookup()
    catalog = []
    for row in rows:
        markdown_path = match_markdown(row, md_lookup)
        tex_dir, source_file_count, tex_file_count = tex_stats(row.get("arxiv_id", ""))
        catalog.append(
            {
                "key": row["key"],
                "year": row["year"],
                "title": row["title"],
                "arxiv_id": row["arxiv_id"],
                "doi": row["doi"],
                "pdf_path": row["pdf_path"],
                "markdown_path": str(markdown_path.relative_to(ROOT)) if markdown_path else "",
                "tex_dir": tex_dir,
                "tex_file_count": tex_file_count,
                "source_file_count": source_file_count,
                "retrieval_status": row["retrieval_status"],
                "tags": row["tags"],
            }
        )

    fields = list(catalog[0].keys()) if catalog else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(catalog)
    OUT_JSON.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_count = sum(1 for row in catalog if row["markdown_path"])
    tex_count = sum(1 for row in catalog if row["tex_file_count"])
    lines = [
        "# Search",
        "",
        "Search the extracted paper text with `rg`:",
        "",
        "```bash",
        "rg -n \"overlap\" markdown_docling tex_sources/extracted",
        "rg -n \"Riesz\" markdown_docling tex_sources/extracted",
        "rg -n \"instrumental variable\" markdown_docling tex_sources/extracted",
        "```",
        "",
        "## Catalog",
        "",
        f"- Manifest records: {len(catalog)}",
        f"- Records with Docling Markdown: {md_count}",
        f"- Records with TeX sources: {tex_count}",
        f"- CSV: `searchable_catalog.csv`",
        f"- JSON: `searchable_catalog.json`",
        "",
        "The PDFs remain in `downloads/`; the searchable text lives in `markdown_docling/` and `tex_sources/extracted/`.",
        "",
    ]
    OUT_README.write_text("\n".join(lines), encoding="utf-8")
    print(f"catalog rows={len(catalog)} markdown={md_count} tex={tex_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
