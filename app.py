import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import streamlit as st
import pandas as pd
import tempfile
import re
from pathlib import Path
from ocr_engine import extract_pages

from compliance_checker import verify_mtc
from processor import stitch_pages, process_batch
from standards import (
    ALL_STANDARDS,
    STANDARD_FAMILIES,
    list_all_grades,
    get_family,
)
from gatekeeper import check_is_valid_mtc

st.set_page_config(
    page_title = "AI-Based Material Test Certificate (MTC) Verification System",
    page_icon  = "📜",
    layout     = "wide"
)

@st.cache_resource
def load_pipeline():
    from paddleocr import PPStructureV3
    return PPStructureV3(
        lang                         = "en",
        use_doc_orientation_classify = True,
        use_doc_unwarping            = True,
        use_table_recognition        = True,
        use_seal_recognition         = False,
        use_formula_recognition      = False,
        use_chart_recognition        = False,
        device                       = "cpu"
    )

FAMILY_COLOR = {
    "Aluminum"       : "🔵",
    "Carbon Steel"   : "🔴",
    "Stainless Steel": "⚪",
    "Titanium"       : "🟡",
    "Nickel Alloy"   : "🟠",
    "Copper Alloy"   : "🟤",
    "Unknown"        : "⚫",
}

st.title("📜 AI-Based Material Test Certificate (MTC) Verification System")
st.markdown(
    "**Automated International Standards Compliance Checker "
    "for Material Test Certificates**"
)
st.divider()

with st.spinner("⏳ Loading OCR model..."):
    pipeline = load_pipeline()
st.success("✅ OCR Engine Ready!")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "📄 Upload Material Test Certificate",
        type = ["pdf", "jpg", "jpeg", "png"],
        help = "Supports PDF, JPG, PNG — Temp file auto-deleted after OCR"
    )

with col2:
    st.markdown("### 📋 Supported Standards")
    for family_label, grades in STANDARD_FAMILIES.items():
        if grades:
            with st.expander(family_label, expanded=False):
                st.markdown(
                    "\n".join(f"- `{g}`" for g in grades)
                )
    st.caption(f"✅ Total: **{len(ALL_STANDARDS)}** grades loaded")

st.divider()
st.subheader("⚙️ Custom Limits")

with st.expander(
    "🔧 Set Custom Element Limits",
    expanded=False
):
    st.info(
        "Leave fields empty to use default only. "
        "Fill in any element to override with a stricter limit."
    )

    common_elements = [
        "Carbon", "Manganese", "Phosphorus", "Sulfur",
        "Silicon", "Chromium", "Nickel", "Molybdenum",
        "Nitrogen", "Niobium", "Vanadium", "Titanium",
        "Copper", "Iron", "Zinc", "Aluminum",
    ]

    custom_inputs = {}
    cols = st.columns(3)

    for i, element in enumerate(common_elements):
        with cols[i % 3]:
            st.markdown(f"**{element}**")
            c1, c2 = st.columns(2)
            with c1:
                min_val = st.number_input(
                    f"Min %",
                    key         = f"min_{element}",
                    value       = None,
                    min_value   = 0.0,
                    max_value   = 100.0,
                    step        = 0.001,
                    format      = "%.3f",
                    placeholder = "default",
                    label_visibility = "visible"
                )
            with c2:
                max_val = st.number_input(
                    f"Max %",
                    key         = f"max_{element}",
                    value       = None,
                    min_value   = 0.0,
                    max_value   = 100.0,
                    step        = 0.001,
                    format      = "%.3f",
                    placeholder = "default",
                    label_visibility = "visible"
                )

            if min_val is not None or max_val is not None:
                entry = {"unit": "%"}
                if min_val is not None:
                    entry["min"] = float(min_val)
                if max_val is not None:
                    entry["max"] = float(max_val)
                custom_inputs[element] = entry

    if custom_inputs:
        st.success(
            f"✅ Custom limits set for: "
            f"**{', '.join(custom_inputs.keys())}**"
        )
        st.json(custom_inputs)
    else:
        st.info("ℹ️ No custom limits set - defaults will apply")

if uploaded:
    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.info(
            f"📁 **File:** {uploaded.name}  \n"
            f"📐 **Size:** {uploaded.size / 1024:.1f} KB  \n"
            f"🔧 **Custom Limits:** "
            f"{'✅ Yes (' + str(len(custom_inputs)) + ' elements)' if custom_inputs else '❌ None'}"
        )
    with col_b:
        verify_btn = st.button(
            "🔍 Verify Certificate",
            type                = "primary",
            use_container_width = True
        )

    if verify_btn:

        with st.spinner("🔍 Pre-scanning document (fast check)..."):

            import tempfile
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                uploaded.seek(0)
                tmp.write(uploaded.read())
                gate_tmp = tmp.name

            gate_result = check_is_valid_mtc(gate_tmp)

        if not gate_result["is_valid"]:

            Path(gate_tmp).unlink(missing_ok=True)

            st.error("## ❌ INVALID DOCUMENT — Verification Blocked")
            st.markdown(f"""
            **Reason:** {gate_result['reason']}
            **Action:** Please upload a valid Material Test Certificate (MTC)
            """)

            with st.expander("🔍 Pre-scan Details"):
                st.json({
                    "Keywords Found" : gate_result["found_keywords"],
                    "Elements Found" : gate_result["found_elements"],
                    "Pages Scanned"  : gate_result["page_count"],
                    "Text Preview"   : gate_result["raw_preview"][:300],
                })

            st.stop()

        else:
            st.success(
                f"✅ Pre-scan passed — {gate_result['reason']}"
            )
            with st.expander("🔍 Pre-scan Details", expanded=False):
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Keywords Detected:**")
                    st.write(gate_result["found_keywords"] or "None")
                with col_g2:
                    st.markdown("**Elements Detected:**")
                    st.write(gate_result["found_elements"] or "None")

        extraction_path = gate_result.get("extraction_path", "ocr_required")

        if extraction_path == "pdfplumber":
            page_dicts = gate_result["pages_data"]
            Path(gate_tmp).unlink(missing_ok=True)
            st.success(
                f"✅ Direct PDF extraction — "
                f"{len(page_dicts)} page(s) read via pdfplumber. "
                f"OCR engine bypassed."
            )
        else:
            with st.spinner(
                "⏳ Running OCR page-by-page... "
                "(Temp file created → OCR → Auto-deleted)"
            ):
                try:
                    uploaded.seek(0)
                    page_dicts = extract_pages(uploaded, pipeline)

                    Path(gate_tmp).unlink(missing_ok=True)
                    st.success(
                        f"✅ OCR Complete — "
                        f"{len(page_dicts)} page(s) extracted. "
                        f"Temp file deleted."
                    )
                except Exception as e:
                    Path(gate_tmp).unlink(missing_ok=True)
                    st.error(f"❌ OCR Failed: {e}")
                    st.stop()

        with st.spinner("⏳ Stitching pages and running compliance checks..."):
            stitched_docs = stitch_pages(page_dicts)
            batch_results = process_batch(
                stitched_docs,
                filename      = uploaded.name,
                custom_limits = custom_inputs or None,
            )

        st.divider()
        total_docs    = len(batch_results)
        pass_count    = sum(1 for r in batch_results if r["verdict"] == "PASS")
        fail_count    = sum(1 for r in batch_results if r["verdict"] == "FAIL")
        review_count  = sum(1 for r in batch_results if "MANUAL" in r["verdict"])
        invalid_count = sum(1 for r in batch_results if "INVALID" in r["verdict"])

        summary_msg = (
            f"🗂️ **Batch Complete** — `{uploaded.name}`  \n"
            f"**{total_docs}** certificate(s) detected | "
            f"✅ Passed: **{pass_count}** | "
            f"❌ Failed: **{fail_count}** | "
            f"🔍 Manual Review: **{review_count}** | "
            f"⚠️ Invalid: **{invalid_count}** | "
            f"💾 All results auto-saved to `audit_log.csv`"
        )
        if fail_count > 0:
            st.error(summary_msg)
        elif review_count > 0:
            st.warning(summary_msg)
        else:
            st.success(summary_msg)

        def _cell_color_status(val):
            if "PASS"          in str(val): return "background-color:#d4edda"
            if "MANUAL REVIEW" in str(val): return "background-color:#ffe8a1"
            if "FAIL"          in str(val): return "background-color:#f8d7da"
            if "NOT FOUND"     in str(val): return "background-color:#fff3cd"
            return ""

        def _cell_color_reason(val):
            if "Custom" in str(val): return "color:#856404; font-weight:bold"
            if "ASTM"   in str(val): return "color:#721c24"
            return ""

        for doc_idx, report in enumerate(batch_results, start=1):
            pages       = report["page_numbers"]
            page_label  = (
                f"p{pages[0]}"
                if len(pages) == 1
                else f"p{pages[0]}–{pages[-1]}"
            )
            family_icon = FAMILY_COLOR.get(report["family"], "⚫")
            verdict     = report["verdict"]
            heat        = report.get("heat_number") or "—"
            if heat in ("Not Found", "", None):
                heat = "—"
            mech        = report.get("mechanicals", {})
            df          = pd.DataFrame(report["details"])

            total_e  = len(df)
            passed_e = (len(df[df["Status"] == "[OK] PASS"])
                        if not df.empty else 0)
            failed_e = (len(df[df["Status"] == "[FAIL] FAIL"])
                        if not df.empty else 0)
            nf_e     = (len(df[df["Status"].str.contains("NOT FOUND")])
                        if not df.empty else 0)
            rev_e    = (len(df[df["Status"].str.contains("MANUAL REVIEW")])
                        if not df.empty else 0)
            score_pct = int(100 * passed_e / total_e) if total_e > 0 else 0

            if verdict == "PASS":
                hdr_bg     = "#0f5132"
                hdr_accent = "#198754"
                v_icon     = "✅"
                v_text     = "PASS"
            elif "MANUAL" in verdict:
                hdr_bg     = "#664d03"
                hdr_accent = "#e6a817"
                v_icon     = "🔍"
                v_text     = "MANUAL REVIEW"
            elif "INVALID" in verdict:
                hdr_bg     = "#343a40"
                hdr_accent = "#6c757d"
                v_icon     = "⚠️"
                v_text     = "INVALID MTC"
            else:
                hdr_bg     = "#842029"
                hdr_accent = "#dc3545"
                v_icon     = "❌"
                v_text     = "FAIL — NON-COMPLIANT"

            pk_badge = (
                f"&nbsp;·&nbsp; Key:&nbsp;<strong>{report['primary_key']}</strong>"
                if report.get("primary_key") else ""
            )
            custom_badge = (
                "&nbsp;·&nbsp; ★&nbsp;Custom&nbsp;Limits&nbsp;Applied"
                if report["custom_used"] else ""
            )

            st.markdown(
                f"""<div style="background:linear-gradient(135deg,{hdr_bg} 0%,{hdr_accent} 100%); border-radius:12px; padding:20px 28px; margin:28px 0 10px 0; display:flex; justify-content:space-between; align-items:center; box-shadow:0 4px 14px rgba(0,0,0,0.18);"><div style="display:flex; flex-direction:column; gap:4px;"><div style="font-size:0.75em; color:rgba(255,255,255,0.8); letter-spacing:1px; text-transform:uppercase; font-weight:700;">📄 Certificate {doc_idx} of {total_docs} &nbsp;·&nbsp; Pages: {page_label}{pk_badge}</div><div style="font-size:1.75em; font-weight:800; color:#fff; line-height:1.2;">{v_icon} {v_text}</div><div style="color:rgba(255,255,255,0.9); font-size:1em;">{family_icon} <strong>{report['grade']}</strong> &nbsp;·&nbsp; {report['family']} &nbsp;·&nbsp; Heat: <strong>{heat}</strong> {custom_badge}</div></div><div style="background:rgba(255,255,255,0.2); border-radius:50%; width:85px; height:85px; display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0; box-shadow:inset 0 2px 4px rgba(0,0,0,0.1);"><div style="font-size:1.6em; font-weight:800; color:#fff; line-height:1.1;">{score_pct}%</div><div style="font-size:0.65em; color:rgba(255,255,255,0.9); font-weight:700; text-transform:uppercase; letter-spacing:1px;">Compliant</div></div></div>""",
                unsafe_allow_html=True
            )

            if "MANUAL" in verdict:
                confirmed = st.checkbox(
                    "✅ I have manually verified these values are safe",
                    key=f"hitl_confirm_{report['mtc_name']}_{doc_idx}",
                )
                if confirmed:
                    st.success(
                        "✅ PASS (Manually Overridden) — reviewer confirmed"
                    )
                else:
                    st.warning(
                        "⚠️ Suspiciously high values detected — the OCR may have "
                        "missed a column multiplier (e.g. ×100 or ×10⁻²). "
                        "Tick the checkbox above after reviewing the printed document."
                    )

            if report["custom_used"]:
                st.info(
                    "★ A custom limit was stricter than the standard and was enforced."
                )

            if report.get("mech_duplication_warning"):
                st.warning(
                    "⚠️ **Data Duplication Detected** — This heat's mechanical "
                    "properties (YS / TS / Elongation) are **identical** to another "
                    "heat in this certificate. Exact numerical coincidence across "
                    "different heats is metallurgically near-impossible and strongly "
                    "suggests the OCR engine copied one row's data into another. "
                    "**Manual review of the printed document is required.**"
                )

            col_shift = report.get("column_shift_warning") or []
            if col_shift:
                details = "; ".join(
                    f"{w['extracted_as']}={w['value']} → likely {w['likely_is']}"
                    for w in col_shift
                )
                st.warning(
                    f"⚠️ **Possible OCR Column Displacement** — The system detected "
                    f"that one or more values were placed in the wrong column by the "
                    f"OCR engine and automatically reassigned them based on "
                    f"metallurgical constraints: {details}. "
                    f"Verify against the printed certificate."
                )

            st.markdown("#### 🔖 Traceability & Mechanical Properties")
            t_col, y_col, ts_col, el_col = st.columns(4)
            t_col.metric(
                "🔢 Heat / Cast No.",
                heat,
                delta       = "Found"  if heat != "—" else "Not Found",
                delta_color = "normal" if heat != "—" else "inverse",
            )
            y_col.metric(
                "💪 Yield Strength",
                mech.get("Yield Strength",   "N/A"),
            )
            ts_col.metric(
                "🏗️ Tensile Strength",
                mech.get("Tensile Strength", "N/A"),
            )
            el_col.metric(
                "📐 Elongation",
                mech.get("Elongation",       "N/A"),
            )

            st.markdown("#### 📊 Chemical Composition Analysis")
            if df.empty:
                st.warning("⚠️ No chemical elements could be extracted.")
            else:
                styled = (
                    df.style
                    .map(_cell_color_status, subset=["Status"])
                    .map(_cell_color_reason, subset=["Fail Reason"])
                )
                st.dataframe(
                    styled,
                    use_container_width = True,
                    hide_index          = True,
                )

            st.markdown("#### 📈 Compliance Scorecard")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📌 Elements Checked", total_e)
            c2.metric("✅ Passed",            passed_e)
            c3.metric("❌ Failed",            failed_e)
            c4.metric("⚠️ Not Found",        nf_e)
            c5.metric("🔍 Manual Review",    rev_e)

            st.caption(
                f"🕐 Verified: {report['timestamp']}  |  "
                f"💾 Auto-saved to `audit_log.csv`"
            )

            with st.expander("📄 View Raw Extracted Text", expanded=False):
                raw   = report.get("raw_text", "")
                clean = re.sub(r"<[^>]+>", " ", raw)
                clean = re.sub(r"\s+",     " ", clean)
                st.text(clean[:3000])

            if doc_idx < total_docs:
                st.divider()

st.divider()
st.subheader("📋 Audit Trail")
st.caption("Auto-updated after every verification — no manual save needed")

if Path("audit_log.csv").exists():
    try:
        audit_df = pd.read_csv("audit_log.csv")

        expected_cols = [
            "Timestamp", "MTC Name", "Grade", "Family",
            "Verdict", "Custom Used", "Total Elements",
            "Passed", "Failed", "Not Found"
        ]
        if not all(col in audit_df.columns for col in expected_cols):
            st.warning(
                "⚠️ Old audit log format detected — resetting file."
            )
            Path("audit_log.csv").unlink()
            st.info("🗑️ audit_log.csv reset. Re-verify to start fresh.")
            st.stop()

    except Exception as e:
        st.error(f"❌ audit_log.csv is corrupted: {e}")
        if st.button("🗑️ Reset Audit Log"):
            Path("audit_log.csv").unlink()
            st.success("✅ Audit log reset. Refresh the page.")
        st.stop()

    def color_verdict(val):
        if val == "PASS":                              return "background-color:#d4edda"
        if val == "FAIL":                              return "background-color:#f8d7da"
        if "MANUAL VERIFICATION" in str(val):         return "background-color:#ffe8a1"
        if "INVALID" in str(val):                     return "background-color:#fff3cd"
        return ""

    def color_custom(val):
        if str(val).lower() == "true":
            return "background-color:#fff3cd; font-weight:bold"
        return ""

    styled_audit = (
        audit_df.style
        .map(color_verdict, subset=["Verdict"])
        .map(color_custom,  subset=["Custom Used"])
    )
    st.dataframe(
        styled_audit,
        use_container_width = True,
        hide_index          = True
    )

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Checks",       len(audit_df))
    a2.metric("✅ Total PASS",
              len(audit_df[audit_df["Verdict"] == "PASS"]))
    a3.metric("❌ Total FAIL",
              len(audit_df[audit_df["Verdict"] == "FAIL"]))
    a4.metric("🔧 With Custom Limits",
              len(audit_df[audit_df["Custom Used"] == True]))

    st.download_button(
        label     = "⬇️ Download Audit Log CSV",
        data      = audit_df.to_csv(index=False),
        file_name = "audit_log.csv",
        mime      = "text/csv"
    )
else:
    st.info("No audit records yet — verify a certificate to begin.")
