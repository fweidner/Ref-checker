# Ref Checker

Extract and verify all references from a research paper PDF against live academic databases.

Optimised for IEEE-style bracket references (`[1]`, `[2]`, ...).

## Usage

```bash
python check_refs.py paper.pdf
python check_refs.py paper.pdf --output results.csv --delay 3
```

**Options:**
- `--output / -o` — save results to CSV
- `--delay / -d` — seconds between API calls (default: 2)
- `--verbose / -v` — print raw text, query, and lookup detail per reference

## How it works

1. Extracts text via `pdftotext` (poppler) or falls back to `pdfplumber`
2. Locates the References section and parses `[N]`-style entries
3. For each reference, looks up in order:
   - **Semantic Scholar** — DOI lookup (most reliable)
   - **Semantic Scholar** — title search
   - **CrossRef** — title/bibliographic search
   - **Google Scholar** — final fallback (skipped once blocked)

## Setup

```bash
pip install -r requirements.txt
```

For best results, also install [poppler](https://poppler.freedesktop.org/) so `pdftotext` is available.

Optional: install `refextract` for improved structured field parsing.

## Output

Each reference is reported as `✓ FOUND`, `✗ NOT FOUND`, or `? UNKNOWN`, with the source database and matched title. A summary is printed at the end.
