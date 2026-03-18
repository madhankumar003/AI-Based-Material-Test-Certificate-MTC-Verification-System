import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import re
def _getval(obj, key, default=''):
    try:
        val = obj[key]
        return val if val is not None else default
    except (KeyError, TypeError, IndexError):
        pass
    return getattr(obj, key, default) or default
import sys
import tempfile
from pathlib import Path

def create_pipeline():
    from paddleocr import PPStructureV3

    print("⏳ Loading PPStructureV3 model...")
    pipeline = PPStructureV3(
        lang                         = "en",
        use_doc_orientation_classify = True,
        use_doc_unwarping            = True,
        use_table_recognition        = True,
        use_seal_recognition         = False,
        use_formula_recognition      = False,
        use_chart_recognition        = False,
        device                       = "cpu",
    )
    print("✅ Model loaded!\n")
    return pipeline

OCR_DEBUG = True

def extract_pages(uploaded_file, pipeline) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in (".pdf", ".jpg", ".jpeg", ".png"):
        raise ValueError(f"Unsupported format: {suffix}")

    stem    = Path(uploaded_file.name).stem
    out_dir = Path("ocr_debug") / stem if OCR_DEBUG else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OCR] Debug output → {out_dir.resolve()}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        return _run_pipeline(
            input_path = tmp_path,
            pipeline   = pipeline,
            filename   = uploaded_file.name,
            out_dir    = out_dir,
        )
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)

def extract_mtc(
    pdf_path  : str,
    pipeline,
    output_dir: str = "outputs",
) -> dict:
    pdf_path = Path(pdf_path)
    out_dir  = Path(output_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Processing: {pdf_path.name}")

    page_results = _run_pipeline(
        input_path = str(pdf_path),
        pipeline   = pipeline,
        filename   = pdf_path.name,
        out_dir    = out_dir,
    )

    full_text   = "\n".join(p["text"]                     for p in page_results)
    html_tables = [t for p in page_results for t in p.get("html_tables", [])]
    markdown    = "\n".join(p.get("_markdown", "")        for p in page_results)

    md_file = out_dir / f"{pdf_path.stem}.md"
    md_file.write_text(markdown, encoding="utf-8")

    print(f"   ✅ Pages       : {len(page_results)}")
    print(f"   ✅ HTML tables : {len(html_tables)}")
    print(f"   ✅ Saved to    : {out_dir}\n")

    return {
        "name"       : pdf_path.stem,
        "text"       : full_text,
        "html_tables": html_tables,
        "markdown"   : markdown,
        "md_path"    : str(md_file),
        "out_dir"    : str(out_dir),
        "pages"      : len(page_results),
    }

_SKIP_LABELS = frozenset({
    'header_image', 'footer_image', 'figure', 'chart',
    'seal', 'formula', 'number', 'footnote',
    'header', 'footer', 'aside_text',
})

def _run_pipeline(input_path, pipeline, filename, out_dir=None) -> list:
    print(f"\n{'='*60}")
    print(f"[OCR] File : {filename}")

    output = pipeline.predict(
        input                                  = input_path,
        use_wireless_table_cells_trans_to_html = True,
    )

    markdown_list   = []
    page_results    = []

    for res in output:
        if out_dir:
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))

        text, html_tables = _extract_page_text(res)

        md_info = res.markdown
        markdown_list.append(md_info)

        pg    = len(page_results) + 1
        pipes = sum(1 for ln in text.splitlines() if '|' in ln)
        print(f"[OCR] --- Page {pg} ---")
        print(f"[OCR]   Chars       : {len(text)}")
        print(f"[OCR]   HTML tables : {len(html_tables)}")
        print(f"[OCR]   Pipe lines  : {pipes}  "
              f"({'pipe table' if pipes > 2 else 'no pipe table'})")
        print(f"[OCR]   Preview     : {text[:300].strip()!r}")

        page_results.append({
            "text"       : text,
            "html_tables": html_tables,
            "_markdown"  : str(md_info) if md_info else "",
        })

    try:
        merged = pipeline.concatenate_markdown_pages(markdown_list)
        md_text = merged if isinstance(merged, str) else str(merged)
        print(f"[OCR] Merged markdown : {len(md_text)} chars")
    except Exception as e:
        print(f"[OCR] concatenate_markdown_pages: {e}")

    print(f"[OCR] Pages total : {len(page_results)}")
    print(f"{'='*60}\n")

    return page_results

def _extract_page_text(res) -> tuple:
    html_tables = []
    text_parts  = []

    try:
        for tbl in (res['table_res_list'] or []):
            html = _getval(tbl, 'pred_html', '')
            if html and html.strip():
                html_tables.append(html)
                print(f"[OCR]   table_res_list → {len(html)} char HTML table")
    except (KeyError, TypeError):
        pass

    got_text = False
    try:
        for region in (res['parsing_res_list'] or []):
            label   = _getval(region, 'block_label', '')
            content = _getval(region, 'block_content', '')

            if label == 'table':
                sub  = _getval(region, 'res', None)
                html = _getval(sub, 'html', '') if sub else ''
                if html and html.strip() and html not in html_tables:
                    html_tables.append(html)
                got_text = True

            elif label not in _SKIP_LABELS and content and content.strip():
                text_parts.append(content)
                got_text = True

    except (KeyError, TypeError):
        pass

    if not got_text:
        print(f"[OCR]   parsing_res_list empty → falling back to overall_ocr_res")
        spatial = _spatial_text_from_ocr(res)
        if spatial.strip():
            text_parts.append(spatial)

    return '\n'.join(text_parts), html_tables

def _spatial_text_from_ocr(res) -> str:
    overall = _getval(res, 'overall_ocr_res', None)
    if overall is None:
        return ""

    rec_texts = _getval(overall, 'rec_texts', []) or []
    dt_polys  = _getval(overall, 'dt_polys',  []) or []

    if not rec_texts:
        return ""

    if not dt_polys or len(dt_polys) != len(rec_texts):
        print(f"[OCR]   overall_ocr_res: {len(rec_texts)} tokens, no polys → flat join")
        return '\n'.join(str(t) for t in rec_texts if str(t).strip())

    items = []
    for tok, poly in zip(rec_texts, dt_polys):
        t = str(tok).strip()
        if not t:
            continue
        try:
            pts = list(poly)
            if isinstance(pts[0], (list, tuple)):
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
            else:
                coords = [float(v) for v in pts]
                xs, ys = coords[0::2], coords[1::2]
            items.append((sum(xs)/len(xs), sum(ys)/len(ys), t))
        except Exception:
            items.append((0.0, float(len(items)), t))

    if not items:
        return ""

    print(f"[OCR]   overall_ocr_res: {len(items)} tokens → spatial column rebuild")

    items.sort(key=lambda i: i[1])
    Y_TOL = 20
    rows, band = [], [items[0]]
    for item in items[1:]:
        if abs(item[1] - band[-1][1]) <= Y_TOL:
            band.append(item)
        else:
            rows.append(sorted(band, key=lambda i: i[0]))
            band = [item]
    rows.append(sorted(band, key=lambda i: i[0]))

    all_cx = [i[0] for i in items]
    x_min  = min(all_cx)
    x_rng  = max(max(all_cx) - x_min, 1.0)
    COLS   = 120

    lines = []
    for row in rows:
        buf = [' '] * (COLS + 60)
        for cx, _, tok in row:
            col = int((cx - x_min) * COLS / x_rng)
            for k, ch in enumerate(tok):
                pos = col + k
                if 0 <= pos < len(buf):
                    buf[pos] = ch
        line = ''.join(buf).rstrip()
        if line.strip():
            lines.append(line)

    return '\n'.join(lines)

if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_path:
        candidates = sorted(Path("data/real_mtcs").glob("*.pdf"))
        if not candidates:
            print("⚠️  Usage: python ocr_engine.py path/to/file.pdf")
            sys.exit(1)
        test_path = str(candidates[0])

    pipe   = create_pipeline()
    result = extract_mtc(test_path, pipe)

    print("=" * 55)
    print("📋 TEXT PREVIEW (first 500 chars):")
    print(result["text"][:500] or "(empty)")
    print("\n📊 HTML TABLES:")
    for i, tbl in enumerate(result["html_tables"], 1):
        print(f"   Table {i}: {len(tbl)} chars — {tbl[:80].strip()}…")
    print(f"\n✅ Markdown → {result['md_path']}")
    print(f"✅ Pages    → {result['pages']}")
