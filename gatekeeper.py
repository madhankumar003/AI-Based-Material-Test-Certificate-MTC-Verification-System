import pdfplumber
from pathlib import Path

def _table_to_html(raw_table: list) -> str:
    rows_html = []
    for row in raw_table:
        cells = "".join(
            f"<td>{str(cell or '').strip()}</td>"
            for cell in row
        )
        rows_html.append(f"<tr>{cells}</tr>")
    return f"<table>{''.join(rows_html)}</table>"

def _extract_pdfplumber_pages(pdf) -> list:
    pages = []
    for page_num, page in enumerate(pdf.pages, 1):
        text       = page.extract_text() or ""
        raw_tables = page.extract_tables() or []

        print(f"\n[PDFPLUMBER] Page {page_num} ─────────────────────────")
        print(f"[PDFPLUMBER] Text length: {len(text)} chars")
        print(f"[PDFPLUMBER] Tables found: {len(raw_tables)}")

        html_tables = []
        for table_idx, t in enumerate(raw_tables):
            if not t:
                continue

            print(f"\n[PDFPLUMBER]   Table {table_idx + 1}: {len(t)} rows")
            if len(t) > 0:
                print(f"[PDFPLUMBER]   First row (headers): {t[0]}")
                if len(t) > 1:
                    print(f"[PDFPLUMBER]   Second row (data):  {t[1]}")

            html = _table_to_html(t)
            html_tables.append(html)
            print(f"[PDFPLUMBER]   HTML output length: {len(html)} chars")

        pages.append({"text": text, "html_tables": html_tables})
        print(f"[PDFPLUMBER] Page {page_num} complete\n")
    return pages

PRIMARY_KEYWORDS = [
    "chemical composition",
    "chemical analysis",
    "mechanical properties",
    "material test",
    "test certificate",
    "mill certificate",
    "inspection certificate",

    "astm", "asme", "en 10204", "iso 10474",
    "sa516", "sa312", "ss316", "ss304",
    "aa7075", "aa6061", "aa2024",
    "grade", "heat no", "heat number",
    "cast no", "batch no",
]

ELEMENT_SYMBOLS = [
    "C", "Mn", "P", "S", "Si",
    "Cr", "Ni", "Mo", "N", "Nb",
    "V",  "Ti", "Cu", "Fe", "Al",
    "Mg", "Zn", "Zr", "W",  "Co",
]

ELEMENT_NAMES = [
    "carbon", "manganese", "phosphorus", "sulfur", "silicon",
    "chromium", "nickel", "molybdenum", "nitrogen", "niobium",
    "vanadium", "titanium", "copper", "iron", "zinc",
    "aluminum", "aluminium",
]

def check_is_valid_mtc(file_path: str) -> dict:
    file_path = Path(file_path)
    suffix    = file_path.suffix.lower()

    if suffix in [".jpg", ".jpeg", ".png"]:
        return {
            "is_valid"        : True,
            "reason"          : "Image file — skipping text pre-scan, "
                               "PaddleOCR will handle directly",
            "found_keywords"  : [],
            "found_elements"  : [],
            "page_count"      : 1,
            "raw_preview"     : "",
            "extraction_path" : "ocr_required",
            "pages_data"      : [],
        }

    if suffix != ".pdf":
        return _fail(f"Unsupported file type: {suffix}")

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            page_count  = len(pdf.pages)
            all_text    = ""

            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"

            pages_data = _extract_pdfplumber_pages(pdf)

        if not all_text.strip():
            return _fail(
                "PDF appears to be a scanned image with no "
                "embedded text — sending to PaddleOCR",
                page_count  = page_count,
                is_valid    = True,
                reason_code = "SCANNED_PDF"
            )

        text_lower      = all_text.lower()
        found_keywords  = [
            kw for kw in PRIMARY_KEYWORDS
            if kw.lower() in text_lower
        ]

        import re
        found_elements = [
            sym for sym in ELEMENT_SYMBOLS
            if re.search(
                rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])",
                all_text
            )
        ]

        found_element_names = [
            name for name in ELEMENT_NAMES
            if name in text_lower
        ]

        has_keyword  = len(found_keywords)      >= 1
        has_element  = (
            len(found_elements)      >= 1 or
            len(found_element_names) >= 1
        )

        raw_preview = all_text[:500].strip()

        if has_keyword and has_element:
            return {
                "is_valid"        : True,
                "reason"          : (
                    f"✅ Valid MTC detected — "
                    f"{len(found_keywords)} keywords + "
                    f"{len(found_elements)} element symbols found"
                ),
                "found_keywords"  : found_keywords,
                "found_elements"  : found_elements + found_element_names,
                "page_count"      : page_count,
                "raw_preview"     : raw_preview,
                "extraction_path" : "pdfplumber",
                "pages_data"      : pages_data,
            }

        elif has_keyword and not has_element:
            return _fail(
                "Document has MTC keywords but NO chemical "
                "element data found — likely not a chemical MTC",
                page_count   = page_count,
                raw_preview  = raw_preview,
                found_keywords = found_keywords,
            )

        elif not has_keyword and has_element:
            return {
                "is_valid"        : True,
                "reason"          : (
                    f"⚠️ No MTC headers found but "
                    f"{len(found_elements)} chemical elements detected "
                    f"— using pdfplumber extraction"
                ),
                "found_keywords"  : [],
                "found_elements"  : found_elements + found_element_names,
                "page_count"      : page_count,
                "raw_preview"     : raw_preview,
                "extraction_path" : "pdfplumber",
                "pages_data"      : pages_data,
            }

        else:
            return _fail(
                "No MTC keywords and no chemical elements found "
                "— this does not appear to be a Material Test Certificate",
                page_count  = page_count,
                raw_preview = raw_preview,
            )

    except Exception as e:
        return {
            "is_valid"        : True,
            "reason"          : f"⚠️ Pre-scan failed ({e}) — "
                               f"allowing PaddleOCR to attempt",
            "found_keywords"  : [],
            "found_elements"  : [],
            "page_count"      : 0,
            "raw_preview"     : "",
            "extraction_path" : "ocr_required",
            "pages_data"      : [],
        }

def _fail(
    reason       : str,
    page_count   : int  = 0,
    raw_preview  : str  = "",
    found_keywords = None,
    is_valid     : bool = False,
    reason_code  : str  = "INVALID",
) -> dict:
    return {
        "is_valid"        : is_valid,
        "reason"          : f"❌ {reason}",
        "found_keywords"  : found_keywords or [],
        "found_elements"  : [],
        "page_count"      : page_count,
        "raw_preview"     : raw_preview,
        "reason_code"     : reason_code,
        "extraction_path" : "ocr_required",
        "pages_data"      : [],
    }

if __name__ == "__main__":
    from pathlib import Path
    import json

    test_dir = Path("data/real_mtcs")
    pdfs     = sorted(test_dir.glob("*.pdf"))[:5]

    for pdf in pdfs:
        result = check_is_valid_mtc(str(pdf))
        print("=" * 55)
        print(f"📄 File    : {pdf.name}")
        print(f"✅ Valid   : {result['is_valid']}")
        print(f"📋 Reason  : {result['reason']}")
        print(f"🔑 Keywords: {result['found_keywords'][:3]}")
        print(f"⚗️  Elements: {result['found_elements'][:5]}")
        print(f"📃 Pages   : {result['page_count']}")
        print()
