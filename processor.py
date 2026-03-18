import re
from compliance_checker import verify_mtc, extract_per_heat_documents

def _html_tables_to_text(html_tables: list) -> str:
    try:
        from bs4 import BeautifulSoup
        parts = []
        for html in (html_tables or []):
            try:
                parts.append(
                    BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
                )
            except Exception:
                pass
        return " ".join(parts)
    except ImportError:
        return ""

_PK_PATTERNS = [
    r"[Hh]eat\s*[Nn]umber\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Hh]eat\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Hh]ea[it]\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Cc]ertificate\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Cc]ert\.?\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Bb]atch\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Cc]ast\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
    r"[Cc]harge\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
]

def extract_primary_key(page_text: str, html_tables: list = None):
    primary_pk = None
    search_text = page_text

    for pattern in _PK_PATTERNS:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            primary_pk = m.group(1).strip().upper()
            break

    if not primary_pk and html_tables:
        table_text = _html_tables_to_text(html_tables)
        search_text = page_text + " " + table_text
        if table_text:
            for pattern in _PK_PATTERNS:
                m = re.search(pattern, table_text, re.IGNORECASE)
                if m:
                    primary_pk = m.group(1).strip().upper()
                    break

    if primary_pk:
        inv = re.search(
            r'INVOICE\s*NO\.?\s*[:/]?\s*(\d+)',
            search_text, re.IGNORECASE
        )
        if inv:
            return f"{primary_pk}_{inv.group(1)}"
        return primary_pk

    return None

def stitch_pages(page_dicts: list) -> list:
    stitched = []

    for page_num, page in enumerate(page_dicts, start=1):
        if isinstance(page, dict):
            text   = page.get("text", "")
            tables = list(page.get("html_tables") or [])
        else:
            text   = str(page)
            tables = []

        pk = extract_primary_key(text, tables)
        print(f"[STITCH] Page {page_num}: PK={pk!r}  text_len={len(text)} chars")

        if not stitched:
            stitched.append({
                "text"        : text,
                "html_tables" : tables,
                "page_numbers": [page_num],
                "primary_key" : pk,
            })
            print(f"[STITCH] -> New doc #1 started (PK={pk!r})")
        else:
            last = stitched[-1]

            if pk is None:
                last["text"]         += "\n" + text
                last["html_tables"]  += tables
                last["page_numbers"].append(page_num)
                print(f"[STITCH] -> Orphan merged into existing doc pages {last['page_numbers']}")

            elif pk == last["primary_key"]:
                last["text"]         += "\n" + text
                last["html_tables"]  += tables
                last["page_numbers"].append(page_num)
                print(f"[STITCH] -> Merged into existing doc (PK={pk!r})")

            elif last["primary_key"] is None:
                last["text"]         += "\n" + text
                last["html_tables"]  += tables
                last["page_numbers"].append(page_num)
                last["primary_key"]   = pk
                print(f"[STITCH] -> Back-fill: orphan doc PK updated to {pk!r}")

            else:
                stitched.append({
                    "text"        : text,
                    "html_tables" : tables,
                    "page_numbers": [page_num],
                    "primary_key" : pk,
                })
                print(f"[STITCH] -> New doc #{len(stitched)} started (PK={pk!r})")

    return stitched

def _detect_mech_duplication(reports: list) -> set:
    MECH_KEYS = ("Yield Strength", "Tensile Strength", "Elongation")
    flagged: set = set()
    for i, r1 in enumerate(reports):
        m1 = tuple(r1["mechanicals"].get(k, "N/A") for k in MECH_KEYS)
        if all(v == "N/A" for v in m1):
            continue
        for j in range(i + 1, len(reports)):
            r2 = reports[j]
            if r1.get("heat_number") == r2.get("heat_number"):
                continue
            m2 = tuple(r2["mechanicals"].get(k, "N/A") for k in MECH_KEYS)
            if m1 == m2:
                flagged.add(i)
                flagged.add(j)
    return flagged

def process_batch(
    stitched_documents: list,
    filename          : str,
    custom_limits     : dict = None,
) -> list:
    batch_results = []

    for doc in stitched_documents:
        pages = doc["page_numbers"]

        if len(pages) == 1:
            page_label = f"p{pages[0]}"
        else:
            page_label = f"p{pages[0]}-{pages[-1]}"

        doc_label = f"{filename} [{page_label}]"
        print(f"\n[BATCH] Processing : {doc_label}")
        print(f"[BATCH]   Pages    : {pages}")
        print(f"[BATCH]   PK       : {doc.get('primary_key')!r}")
        print(f"[BATCH]   Text len : {len(doc['text'])} chars")

        html_tables = doc.get("html_tables") or []

        per_heat = extract_per_heat_documents(html_tables)

        if len(per_heat) >= 2:
            unique_heat_nos = {ph["heat_no"] for ph in per_heat}
            if len(unique_heat_nos) == 1:
                print(
                    f"[BATCH]   Duplicate heat rows: all {len(per_heat)} rows share "
                    f"heat '{per_heat[0]['heat_no']}' - collapsing to 1"
                )
                per_heat = [per_heat[0]]

            print(f"[BATCH]   Multi-heat cert: {len(per_heat)} heat rows detected")
            heat_reports = []
            for i, ph in enumerate(per_heat, 1):
                sub_label = f"{doc_label} [H{i}/{len(per_heat)}]"
                report = verify_mtc(
                    markdown_text = doc["text"],
                    mtc_name      = sub_label,
                    custom_limits = custom_limits,
                    html_tables   = [ph["html"]],
                )
                report["heat_number"] = ph["heat_no"]
                print(f"[BATCH]   H{i} Heat={ph['heat_no']}  Grade={report['grade']}  Verdict={report['verdict']}")
                report["page_numbers"] = pages
                report["primary_key"]  = f"{doc['primary_key']}_H{i}"
                report["raw_text"]     = doc["text"]
                heat_reports.append(report)

            dup_indices = _detect_mech_duplication(heat_reports)
            for idx, report in enumerate(heat_reports):
                report["mech_duplication_warning"] = (idx in dup_indices)
                if idx in dup_indices:
                    print(
                        f"[BATCH]   [WARN] Mech duplication: H{idx+1} "
                        f"({report['heat_number']}) identical mech to another "
                        f"heat — OCR row-copy suspected"
                    )
                batch_results.append(report)
        else:
            report = verify_mtc(
                markdown_text = doc["text"],
                mtc_name      = doc_label,
                custom_limits = custom_limits,
                html_tables   = html_tables,
            )

            print(f"[BATCH]   Grade    : {report['grade']}")
            print(f"[BATCH]   Verdict  : {report['verdict']}")
            report["page_numbers"] = pages
            report["primary_key"]  = doc["primary_key"]
            report["raw_text"]     = doc["text"]

            batch_results.append(report)

        del per_heat

    return batch_results
