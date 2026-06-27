# Search

Search the extracted paper text with `rg`:

```bash
rg -n "overlap" markdown_docling tex_sources/extracted
rg -n "Riesz" markdown_docling tex_sources/extracted
rg -n "instrumental variable" markdown_docling tex_sources/extracted
```

## Catalog

- Manifest records: 86
- Records with Docling Markdown: 79
- Records with TeX sources: 75
- CSV: `searchable_catalog.csv`
- JSON: `searchable_catalog.json`

The PDFs remain in `downloads/`; the searchable text lives in `markdown_docling/` and `tex_sources/extracted/`.
