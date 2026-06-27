#!/usr/bin/env python3
"""Fetch public arXiv source bundles for manifest entries.

Only text-like source files are kept.  Binary figures and raw archives are not
stored because the goal is searchable TeX, not a full mirror of arXiv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.csv"
OUT_DIR = ROOT / "tex_sources"
EXTRACTED = OUT_DIR / "extracted"
TEXT_SUFFIXES = {
    ".tex",
    ".ltx",
    ".latex",
    ".bib",
    ".bbl",
    ".bst",
    ".cls",
    ".sty",
    ".def",
    ".cfg",
    ".clo",
    ".txt",
    ".md",
}
MAX_TEXT_FILE_BYTES = 5_000_000
USER_AGENT = "deep-inference-literature-pack/1.0 (arxiv source retrieval)"


@dataclass
class SourceStatus:
    key: str
    title: str
    arxiv_id: str
    status: str
    text_file_count: int = 0
    tex_file_count: int = 0
    bytes_written: int = 0
    source_dir: str = ""
    error: str = ""


def safe_name(value: str) -> str:
    value = value.replace("\\", "/").strip("/")
    parts = [part for part in value.split("/") if part and part not in {".", ".."}]
    cleaned = []
    for part in parts:
        cleaned.append(re.sub(r"[^A-Za-z0-9._+=,@ -]+", "_", part)[:160] or "file")
    return "/".join(cleaned)


def is_text_source(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch_bytes(arxiv_id: str, timeout: int) -> bytes:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def write_text_file(out_root: Path, relative_name: str, body: bytes) -> tuple[int, bool]:
    safe_relative = safe_name(relative_name)
    if not safe_relative:
        safe_relative = "source.tex"
    out_path = out_root / safe_relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return len(body), out_path.suffix.lower() in {".tex", ".ltx", ".latex"}


def extract_tar(body: bytes, out_root: Path) -> tuple[int, int, int]:
    text_count = 0
    tex_count = 0
    bytes_written = 0
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not is_text_source(member.name):
                continue
            if member.size > MAX_TEXT_FILE_BYTES:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            payload = stream.read()
            written, is_tex = write_text_file(out_root, member.name, payload)
            text_count += 1
            tex_count += int(is_tex)
            bytes_written += written
    return text_count, tex_count, bytes_written


def extract_single(body: bytes, arxiv_id: str, out_root: Path) -> tuple[int, int, int, str]:
    payload = body
    try:
        payload = gzip.decompress(body)
    except OSError:
        pass
    if payload.startswith(b"%PDF"):
        return 0, 0, 0, "pdf-returned"
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        try:
            payload.decode("latin-1")
        except UnicodeDecodeError:
            return 0, 0, 0, "binary-source"
    written, is_tex = write_text_file(out_root, f"{arxiv_id}.tex", payload)
    return 1, int(is_tex), written, "extracted"


def extract_sources(body: bytes, arxiv_id: str, out_root: Path) -> tuple[str, int, int, int]:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        text_count, tex_count, bytes_written = extract_tar(body, out_root)
        status = "extracted" if tex_count else "no-tex-files"
        return status, text_count, tex_count, bytes_written
    except tarfile.TarError:
        text_count, tex_count, bytes_written, status = extract_single(body, arxiv_id, out_root)
        return status, text_count, tex_count, bytes_written


def process(row: dict[str, str], args: argparse.Namespace) -> SourceStatus:
    arxiv_id = row["arxiv_id"].strip()
    status = SourceStatus(
        key=row["key"],
        title=row["title"],
        arxiv_id=arxiv_id,
        status="skipped",
    )
    if not arxiv_id:
        status.status = "no-arxiv-id"
        return status

    source_dir = EXTRACTED / arxiv_id
    status.source_dir = str(source_dir.relative_to(ROOT))
    if source_dir.exists() and any(source_dir.rglob("*")) and not args.force:
        tex_files = [path for path in source_dir.rglob("*") if path.suffix.lower() in {".tex", ".ltx", ".latex"}]
        text_files = [path for path in source_dir.rglob("*") if path.is_file()]
        status.status = "already-extracted"
        status.text_file_count = len(text_files)
        status.tex_file_count = len(tex_files)
        status.bytes_written = sum(path.stat().st_size for path in text_files)
        return status

    try:
        body = fetch_bytes(arxiv_id, args.timeout)
        extracted_status, text_count, tex_count, bytes_written = extract_sources(body, arxiv_id, source_dir)
        status.status = extracted_status
        status.text_file_count = text_count
        status.tex_file_count = tex_count
        status.bytes_written = bytes_written
    except HTTPError as exc:
        status.status = "http-error"
        status.error = f"HTTP {exc.code}: {exc.reason}"
    except (URLError, TimeoutError, OSError) as exc:
        status.status = "request-error"
        status.error = str(exc)
    return status


def write_outputs(statuses: list[SourceStatus]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(statuses[0]).keys()) if statuses else list(SourceStatus("", "", "", "").__dict__)
    with (OUT_DIR / "source_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for status in statuses:
            writer.writerow(asdict(status))
    (OUT_DIR / "source_status.json").write_text(
        json.dumps([asdict(status) for status in statuses], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status.status] = counts.get(status.status, 0) + 1
    lines = [
        "# arXiv TeX Sources",
        "",
        "Fetched with `fetch_arxiv_sources.py` from public `https://arxiv.org/e-print/<id>` endpoints.",
        "Only text-like files are stored; raw archives and binary figures are discarded.",
        "",
        "## Status Counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines.extend(["", "## Rebuild", "", "```bash", "python3 fetch_arxiv_sources.py", "```", ""])
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between arXiv source requests")
    parser.add_argument("--timeout", type=int, default=45, help="Per-request timeout")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many arXiv rows")
    parser.add_argument("--force", action="store_true", help="Refetch existing source directories")
    args = parser.parse_args()

    rows = [row for row in read_manifest() if row.get("arxiv_id")]
    if args.limit:
        rows = rows[: args.limit]
    statuses: list[SourceStatus] = []
    for index, row in enumerate(rows, start=1):
        status = process(row, args)
        statuses.append(status)
        print(f"{index:03d}/{len(rows):03d} {status.arxiv_id} {status.status} tex={status.tex_file_count}")
        if index < len(rows) and args.sleep > 0:
            time.sleep(args.sleep)
    write_outputs(statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
