#!/usr/bin/env python3
"""
check_refs.py -- Extract and verify references from a research paper PDF.

Optimised for IEEE-style and abbrv-doi BibTeX bracket references:
  [N] Authors. Title. Venue, Year. doi: 10.xxx/yyy

Extraction: uses pdftotext -raw (poppler-utils) for clean word spacing.
Falls back to pdfplumber if pdftotext is not available.

Lookup order:
  1. Semantic Scholar DOI lookup  (deterministic, most reliable)
  2. Semantic Scholar title search (fallback for refs without DOI)
  3. CrossRef title search         (last resort)
  4. Google Scholar via scholarly  (final fallback; skipped once blocked)

Usage:
  python check_refs.py paper.pdf
  python check_refs.py paper.pdf --output results.csv --delay 3

Developed with the assistance of Claude (Anthropic) via Claude Code.
"""

import sys
import re
import time
import csv
import subprocess
import argparse
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 1.  PDF text extraction
# ---------------------------------------------------------------------------

# Lines matching these patterns are PDF headers/footers, not reference content.
JUNK_LINES = re.compile(
    r"©\s*20\d\d\s+IEEE"
    r"|author.s version"
    r"|final version of this record"
    r"|^\s*\d{1,2}\s*$"         # lone page numbers
    r"|\f",                     # form-feed characters
    re.IGNORECASE,
)


def extract_text_pdftotext(pdf_path: str) -> str:
    """Use pdftotext -raw for clean, properly spaced extraction."""
    result = subprocess.run(
        ["pdftotext", "-raw", pdf_path, "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr[:200]}")
    return result.stdout


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Fallback: pdfplumber column-crop extraction."""
    import pdfplumber
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            mid = page.width / 2
            for col in [
                page.crop((0, 0, mid, page.height)),
                page.crop((mid, 0, page.width, page.height)),
            ]:
                t = col.extract_text()
                if t:
                    parts.append(t)
    return "\n".join(parts)


def extract_text(pdf_path: str) -> str:
    try:
        return extract_text_pdftotext(pdf_path)
    except (FileNotFoundError, RuntimeError):
        print("[warn] pdftotext not found; falling back to pdfplumber (spacing may be imperfect).",
              file=sys.stderr)
        return extract_text_pdfplumber(pdf_path)


# ---------------------------------------------------------------------------
# 2.  Reference section isolation
# ---------------------------------------------------------------------------

SECTION_HEADERS = [
    r"^\s*REFERENCES\s*$",
    r"^\s*Bibliography\s*$",
    r"^\s*Works\s+Cited\s*$",
    r"^\s*Literature\s+Cited\s*$",
    r"^\s*\d+\.\s+References\s*$",
]


def find_references_section(text: str) -> str:
    """
    Return text from the LAST occurrence of a References header onward.
    Using the last occurrence avoids matching mentions of 'REFERENCES'
    earlier in the body text.
    """
    lines = text.split("\n")
    last_start = None
    for i, line in enumerate(lines):
        for pat in SECTION_HEADERS:
            if re.match(pat, line, re.IGNORECASE):
                last_start = i
                break

    if last_start is None:
        print("[warn] Could not locate a References section; scanning full text.", file=sys.stderr)
        return text

    return "\n".join(lines[last_start:])


# ---------------------------------------------------------------------------
# 3.  Reference parsing  (strict [N] bracket format)
# ---------------------------------------------------------------------------

REF_START = re.compile(r"^\[(\d{1,3})\]\s+")


def parse_references(ref_section: str) -> list[dict]:
    """
    Returns list of dicts: {"number": int, "raw": str}

    - Only matches lines starting with [N] (strict bracket format).
    - Enforces monotonically increasing numbers to reject stray in-text
      citations that may bleed into the extracted section.
    - Skips PDF header/footer junk lines between references.
    """
    lines = ref_section.split("\n")
    refs = []
    current = None
    last_num = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip known junk lines (page footers, copyright headers)
        if JUNK_LINES.search(line):
            continue

        m = REF_START.match(line)
        if m:
            num = int(m.group(1))
            # Reject out-of-sequence numbers (stray in-text citations)
            if num != last_num + 1:
                if last_num > 0:
                    if current:
                        current["raw"] += " " + line
                    continue
            if current:
                refs.append(current)
            last_num = num
            current = {"number": num, "raw": line[m.end():]}
        elif current:
            # Continuation line — rejoin soft hyphens from PDF line breaks.
            # When a word is hyphenated at a line break the raw ends with '-'
            # and the next line starts with the rest of the word.
            # e.g. "di-" + "rect volume" → "direct volume"
            #      "Self-" + "Supervised" → "Self-Supervised"
            # Heuristic: always strip the trailing hyphen and join without
            # space; real compound hyphens are preserved inside the word.
            prev = current["raw"]
            if prev.endswith('-') and line and line[0].isalpha():
                current["raw"] = prev[:-1] + line
            else:
                current["raw"] += " " + line

    if current:
        refs.append(current)

    if not refs:
        print("[warn] No [N]-style references found. Check the PDF.", file=sys.stderr)

    return refs


# ---------------------------------------------------------------------------
# 4.  DOI and title extraction
# ---------------------------------------------------------------------------

def _fix_split_dois(raw: str) -> str:
    """
    Rejoin DOIs that were split across PDF line breaks.

    Three patterns observed in two-column IEEE PDFs after line joining:
      1. "doi: 10. 1109/..."   -- split right after "10."
      2. "doi: 10.xxx/yyy. NNN" -- split after a period mid-DOI
      3. "doi: 10.xxx/yyy .NNN" -- split before a period mid-DOI
    """
    # Case 1: prefix "10." separated from the rest of the DOI
    raw = re.sub(r'(doi:\s*10\.)\s+(\d)', r'\1\2', raw, flags=re.IGNORECASE)
    # Case 2: trailing period before whitespace + alphanumeric continuation
    raw = re.sub(r'(10\.\d{4,}/\S+)\.\s+(\w{3,})', r'\1.\2', raw, flags=re.IGNORECASE)
    # Case 3: space before leading period + alphanumeric continuation
    raw = re.sub(r'(10\.\d{4,}/\S+)\s+\.(\w)', r'\1.\2', raw, flags=re.IGNORECASE)
    return raw


def extract_doi(raw: str) -> str | None:
    """Pull DOI from 'doi: 10.xxx/yyy' pattern, handling line-break splits."""
    raw = _fix_split_dois(raw)
    m = re.search(r'\bdoi:\s*(10\.\d{4,}/\S+)', raw, re.IGNORECASE)
    if not m:
        return None
    doi = m.group(1).rstrip('.,;)')
    # Strip trailing in-paper page-reference tokens (e.g. "...0086 5" or "...0086 4, 5")
    doi = re.sub(r'\s+[\d,\s]+$', '', doi).strip()
    return doi


VENUE_PATTERNS = re.compile(
    r'\b(vol\.|pp\.|no\.|ed\.|Trans\.|Proc\.|Conf\.|Lett\.|Int\.|Symp\.|arXiv)\b'
    r'|^In\b',
    re.IGNORECASE,
)


def extract_title(raw: str) -> str:
    """
    Extract title from a reference formatted by IEEE or abbrv-doi BibTeX styles.

    Both styles use period-separated blocks:
      Authors. Title. Venue, Year. doi: ...

    abbrv specifics that require special handling:
      - Abbreviated first names (F.~Smith) create extra '. ' split points
      - This yields author-fragment segments like "Jones, and G" (3 words,
        0 initials, uppercase-starting) that must be filtered
      - Titles may be shorter than 4 words (sentence-case, no quotes)
      - Venue blocks contain abbreviations like "IEEE Trans." or start with "In"
    """
    # Strip doi and everything after
    clean = re.sub(r'\bdoi:\s*\S.*$', '', raw, flags=re.IGNORECASE).strip()
    # Strip trailing stray numbers: " 5", ", 5" or ". 5" (PDF citation artefacts)
    # Use \d{1,2} to avoid stripping legitimate 4-digit years
    clean = re.sub(r'(?:[\s,]|\.)\s*\d{1,2}\s*$', '', clean).strip()
    # Strip URLs
    clean = re.sub(r'https?://\S+', '', clean).strip()

    _MONTHS = (r'Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?'
               r'|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?'
               r'|Nov(?:ember)?|Dec(?:ember)?')

    for seg in re.split(r'\.\s+', clean):
        words = seg.split()
        if len(words) < 3:
            continue
        if not seg[0:1].isupper():
            continue
        # Skip author-list fragments:
        #   >= 2 abbreviated initials (e.g. "F. Smith, G. Jones")
        initials = sum(1 for w in words if re.match(r'^[A-Z]\.$', w))
        if initials >= 2:
            continue
        #   ends with a bare capital — cut-off first initial of next author
        #   e.g. "Jones, and G" or "Smith and B"
        if re.search(r'\s[A-Z]$', seg):
            continue
        # Skip venue/metadata segments
        if VENUE_PATTERNS.search(seg):
            continue
        # Strip trailing date that isn't a separate block: ", January 2003" or ", 2003"
        seg = re.sub(rf',\s+(?:{_MONTHS})\.?\s+\d{{4}}\s*$', '', seg, flags=re.IGNORECASE).strip()
        seg = re.sub(r',\s+\d{4}\s*$', '', seg).strip()
        return seg

    return clean[:180]


def build_query(raw: str) -> str:
    return extract_title(raw)[:200]


# ---------------------------------------------------------------------------
# 4b. refextract parsing (primary structured field extraction)
# ---------------------------------------------------------------------------

def refextract_parse(raw: str) -> dict:
    """
    Parse a raw reference string with refextract.

    Returns a dict with keys: doi, title, authors, year — all may be None.
    Returns an empty dict if refextract is not installed or parsing fails,
    so callers can always fall back to the regex-based helpers.
    """
    try:
        from refextract import extract_references_from_string
    except ImportError:
        return {}
    try:
        results = extract_references_from_string(raw)
        if not results:
            return {}
        rec = results[0]
        doi    = (rec.get("doi")    or [None])[0] or None
        titles = rec.get("title", [])
        title  = (titles[0].get("title") if titles else None) or None
        authors = rec.get("author") or None
        year   = (rec.get("year")   or [None])[0] or None
        return {"doi": doi, "title": title, "authors": authors, "year": year}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 5.  Semantic Scholar lookups
# ---------------------------------------------------------------------------

SS_BASE    = "https://api.semanticscholar.org/graph/v1/paper"
SS_SEARCH  = f"{SS_BASE}/search"
SS_FIELDS  = "title,authors,year,externalIds"
SS_HEADERS = {"User-Agent": "ref-checker/1.0"}


def _ss_request(url: str, params: dict) -> tuple[bool | None, object]:
    try:
        r = requests.get(url, params=params, headers=SS_HEADERS, timeout=12)
        if r.status_code == 429:
            return None, "SS: rate-limited"
        if r.status_code == 404:
            return False, "SS: not found (404)"
        if r.status_code != 200:
            return None, f"SS: HTTP {r.status_code}"
        return True, r.json()
    except requests.exceptions.Timeout:
        return None, "SS: timeout"
    except Exception as e:
        return None, f"SS: error ({str(e)[:60]})"


def check_ss_doi(doi: str) -> tuple[bool | None, str]:
    ok, payload = _ss_request(f"{SS_BASE}/DOI:{doi}", {"fields": SS_FIELDS})
    if ok is not True:
        return ok, str(payload)
    title = payload.get("title", "?")
    year  = payload.get("year", "?")
    return True, f"SS-DOI: {title[:80]} ({year})"


def check_ss_title(query: str) -> tuple[bool | None, str]:
    ok, payload = _ss_request(SS_SEARCH, {"query": query, "limit": 1, "fields": SS_FIELDS})
    if ok is not True:
        return ok, str(payload)
    data = payload.get("data", [])
    if not data:
        return False, "SS: no results"
    title = data[0].get("title", "?")
    year  = data[0].get("year", "?")
    return True, f"SS-title: {title[:80]} ({year})"


# ---------------------------------------------------------------------------
# 6.  CrossRef (last resort)
# ---------------------------------------------------------------------------

CR_SEARCH = "https://api.crossref.org/works"
CR_HEADERS = {"User-Agent": "ref-checker/1.0 (mailto:user@example.com)"}


def check_crossref(query: str, authors: list | None = None, year: str | None = None) -> tuple[bool | None, str]:
    # Build the richest query possible from available metadata
    if authors or year:
        bib_parts = [query]
        if authors:
            bib_parts.append(", ".join(authors[:3]))
        if year:
            bib_parts.append(year)
        params = {"query.bibliographic": " ".join(bib_parts), "rows": 1,
                  "select": "title,author,published,score"}
    else:
        params = {"query.title": query, "rows": 1,
                  "select": "title,author,published,score"}
    try:
        r = requests.get(
            CR_SEARCH,
            params=params,
            headers=CR_HEADERS,
            timeout=12,
        )
        if r.status_code == 429:
            return None, "CR: rate-limited"
        if r.status_code != 200:
            return None, f"CR: HTTP {r.status_code}"
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return False, "CR: no results"
        item = items[0]
        title = (item.get("title") or ["?"])[0]
        year = (item.get("published", {}).get("date-parts") or [["?"]])[0][0]
        score = item.get("score", 0)
        if score < 50:
            return False, f"CR: low confidence ({score:.0f}) — {title[:80]}"
        return True, f"CR: {title[:80]} ({year})"
    except requests.exceptions.Timeout:
        return None, "CR: timeout"
    except Exception as e:
        return None, f"CR: error ({str(e)[:60]})"


# ---------------------------------------------------------------------------
# 6b. Google Scholar (final last-resort fallback)
# ---------------------------------------------------------------------------

def check_google_scholar(query: str, timeout: float = 15.0) -> tuple[bool | None, str]:
    """
    Query Google Scholar via the `scholarly` library.

    Rate-limiting / blocking safeguards:
      - Entire call (generator creation + first result) runs in a worker
        thread so the timeout can fire even if scholarly hangs internally.
      - Any response that looks like a block (HTTP 429, captcha, timeout)
        returns a detail string containing "blocked" so the caller can set
        gs_blocked=True and skip all further GS calls for this run.
    """
    try:
        from scholarly import scholarly as _scholarly
    except ImportError:
        return None, "GS: scholarly not installed"
    import threading

    result_box = [None]   # [value] or [StopIteration] or [Exception]

    def _run():
        try:
            result_box[0] = next(_scholarly.search_pubs(query))
        except StopIteration:
            result_box[0] = StopIteration
        except Exception as e:
            result_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Thread still blocked (GS hanging) — abandon it and move on
        return None, "GS: blocked (timeout — likely rate-limited)"

    val = result_box[0]
    if val is StopIteration:
        return False, "GS: no results"
    if isinstance(val, Exception):
        msg = str(val)
        is_blocked = any(w in msg.lower() for w in ("blocked", "captcha", "429", "too many"))
        return None, f"GS: {'blocked' if is_blocked else 'error'} ({msg[:80]})"
    if val is None:
        return None, "GS: no response"
    title = val.get("bib", {}).get("title", "?")
    return True, f"GS: {title[:80]}"


# ---------------------------------------------------------------------------
# 7.  Orchestration
# ---------------------------------------------------------------------------

STATUS_SYMBOLS = {True: "✓", False: "✗", None: "?"}
STATUS_LABELS  = {True: "FOUND", False: "NOT FOUND", None: "UNKNOWN"}


def check_reference(ref: dict, gs_blocked: bool) -> tuple[dict, bool]:
    rx    = refextract_parse(ref["raw"])
    doi   = rx.get("doi")   or extract_doi(ref["raw"])
    query = rx.get("title") or build_query(ref["raw"])
    query = query[:200]

    result = {
        "number": ref["number"],
        "raw":    ref["raw"][:120],
        "doi":    doi or "",
        "query":  query[:80],
        "found":  None,
        "source": "",
        "detail": "",
    }

    # 1. DOI lookup (deterministic)
    if doi:
        found, detail = check_ss_doi(doi)
        if found is True:
            result.update(found=True, source="SS-DOI", detail=detail)
            return result, gs_blocked
        result["detail"] = detail

    # 2. Title search
    if query:
        found, detail = check_ss_title(query)
        if found is True:
            result.update(found=True, source="SS-title", detail=detail)
            return result, gs_blocked
        if found is False:
            result.update(found=False, source="SS-title", detail=detail)
        else:
            result["detail"] += f" | {detail}"

    # 3. CrossRef
    if query:
        found, detail = check_crossref(query, authors=rx.get("authors"), year=rx.get("year"))
        if found is True:
            result.update(found=True, source="CR", detail=detail)
            return result, gs_blocked
        if result["found"] is None:
            result.update(
                found=found,
                source="CR" if found is not None else "",
                detail=detail,
            )

    # 4. Google Scholar — whenever not yet positively confirmed; skip once blocked
    if query and result["found"] is not True and not gs_blocked:
        found, detail = check_google_scholar(query)
        if "blocked" in detail.lower():
            gs_blocked = True
            print("[warn] Google Scholar blocked — skipping GS for remaining references.",
                  file=sys.stderr)
        if found is True:
            result.update(found=True, source="GS", detail=detail)
            return result, gs_blocked
        if result["found"] is None:
            result.update(
                found=found,
                source="GS" if found is not None else "",
                detail=detail,
            )

    return result, gs_blocked


def run(pdf_path: str, output_csv: str | None, delay: float, verbose: bool):
    print(f"[+] Extracting text from: {pdf_path}")
    text = extract_text(pdf_path)

    print("[+] Isolating references section...")
    ref_section = find_references_section(text)

    print("[+] Parsing references...")
    refs = parse_references(ref_section)
    print(f"    Found {len(refs)} references.\n")

    if not refs:
        print("[!] No references parsed. Use --verbose or check the PDF manually.")
        sys.exit(1)

    results    = []
    gs_blocked = False
    col_w      = 55

    header = f"{'#':>4}  {'Status':<10}  {'Via':<9}  {'DOI / Title query (truncated)'}"
    print(header)
    print("-" * (len(header) + 10))

    for i, ref in enumerate(refs):
        res, gs_blocked = check_reference(ref, gs_blocked)

        sym     = STATUS_SYMBOLS[res["found"]]
        lbl     = STATUS_LABELS[res["found"]]
        via     = res["source"] or "-"
        display = (f"doi:{res['doi']}" if res["doi"] else res["query"])[:col_w]

        print(f"{res['number']:>4}  {sym} {lbl:<8}  {via:<9}  {display}")

        if verbose:
            print(f"      raw:    {res['raw'][:90]}")
            print(f"      query:  {res['query']}")
            print(f"      detail: {res['detail']}")

        results.append(res)

        if i < len(refs) - 1:
            time.sleep(delay)

    found_n   = sum(1 for r in results if r["found"] is True)
    missing_n = sum(1 for r in results if r["found"] is False)
    unknown_n = sum(1 for r in results if r["found"] is None)
    doi_n     = sum(1 for r in results if r["doi"])

    print("\n" + "=" * 60)
    print(f"  Total:       {len(results)}  (with DOI: {doi_n})")
    print(f"  ✓ Found:     {found_n}")
    print(f"  ✗ Not found: {missing_n}")
    print(f"  ? Unknown:   {unknown_n}")
    print("=" * 60)

    if gs_blocked:
        print("\n[!] Google Scholar blocked mid-run.")
        print("    Proxy setup: https://github.com/scholarly-python-package/scholarly#using-proxies")

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["number", "found", "source", "doi", "detail", "raw", "query"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[+] Results saved to: {output_csv}")


# ---------------------------------------------------------------------------
# 8.  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract and verify references from a research paper PDF (IEEE bracket format)."
    )
    parser.add_argument("pdf", default="test2.pdf", help="Path to the input PDF")
    parser.add_argument("--output", "-o", default=None,
                        help="Export results to CSV (optional)")
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                        help="Seconds between lookups (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print raw, query, and detail per reference")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"[!] File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    run(args.pdf, args.output, args.delay, args.verbose)


if __name__ == "__main__":
    main()
