import re
import pandas as pd
from bs4 import BeautifulSoup
from standards import ALL_STANDARDS, GRADE_KEYWORDS, get_family
from datetime import datetime
from io import StringIO
from rapidfuzz import process, fuzz

_GRADE_ANCHORS = [
    r"[Gg]rade\s*[:/]?\s*",
    r"[Mm]aterial\s*[:/]?\s*",
    r"[Ss]pecification\s*[:/]?\s*",
    r"[Aa]lloy\s*[:/]?\s*",
    r"[Ss]tandard\s*[:/]?\s*",
    r"[Mm]at\.?\s*[Gg]rade\s*[:/]?\s*",
    r"[Pp]roduct\s+[Ss]tandard\s*[:/]?\s*",
    r"[Tt]ype\s*[:/]?\s*",
]
_ANCHOR_RADIUS    = 150                                          
_FUZZY_THRESHOLD  = 85                                                    

_POISON_WORDS = [
    "max", "min", "requirement", "standard",
    "limit", "aim", "specified",
]

_ELEMENT_ALIASES = {
    "carbon"     : "Carbon",      "manganese"  : "Manganese",
    "phosphorus" : "Phosphorus",   "sulfur"     : "Sulfur",
    "sulphur"    : "Sulfur",       "silicon"    : "Silicon",
    "chromium"   : "Chromium",     "nickel"     : "Nickel",
    "molybdenum" : "Molybdenum",   "nitrogen"   : "Nitrogen",
    "niobium"    : "Niobium",      "vanadium"   : "Vanadium",
    "titanium"   : "Titanium",     "tungsten"   : "Tungsten",
    "copper"     : "Copper",       "zinc"       : "Zinc",
    "tin"        : "Tin",          "lead"       : "Lead",
    "iron"       : "Iron",         "oxygen"     : "Oxygen",
    "hydrogen"   : "Hydrogen",     "aluminum"   : "Aluminum",
    "aluminium"  : "Aluminum",
    "c"  : "Carbon",    "mn" : "Manganese",
    "p"  : "Phosphorus","s"  : "Sulfur",
    "si" : "Silicon",   "cr" : "Chromium",
    "ni" : "Nickel",    "mo" : "Molybdenum",
    "n"  : "Nitrogen",  "nb" : "Niobium",
    "v"  : "Vanadium",  "ti" : "Titanium",
    "w"  : "Tungsten",  "cu" : "Copper",
    "zn" : "Zinc",      "sn" : "Tin",
    "pb" : "Lead",      "fe" : "Iron",
    "o"  : "Oxygen",    "h"  : "Hydrogen",
    "al" : "Aluminum",  "mg" : "Mg",
    "n1" : "Nickel",                                                                
    "ai" : "Aluminum",                                                 
    "t1" : "Titanium",                                                              
    "a1" : "Aluminum",                                                              
    "t.s"              : "Tensile Strength",
    "t.s."             : "Tensile Strength",
    "ts"               : "Tensile Strength",
    "uts"              : "Tensile Strength",
    "rm"               : "Tensile Strength",
    "tensile"          : "Tensile Strength",
    "tensile strength" : "Tensile Strength",
    "y.s"              : "Yield Strength",
    "y.s."             : "Yield Strength",
    "ys"               : "Yield Strength",
    "re"               : "Yield Strength",
    "yield"            : "Yield Strength",
    "yield strength"   : "Yield Strength",
    "rp0.2"            : "Yield Strength",
    "e.l"              : "Elongation",
    "e.l."             : "Elongation",
    "el"               : "Elongation",
    "el%"              : "Elongation",
    "el.%"             : "Elongation",
    "a%"               : "Elongation",
    "elongation"       : "Elongation",
    "gl"               : "Gauge Length",
    "gauge length"     : "Gauge Length",
    "ten dir ys"           : "Yield Strength",
    "ten dir uts"          : "Tensile Strength",
    "ten dir el"           : "Elongation",
    "dir ys"               : "Yield Strength",
    "dir uts"              : "Tensile Strength",
    "ce"                   : "Carbon Equivalent",
    "ceq"                  : "Carbon Equivalent",
    "eq"                   : "Carbon Equivalent",                                            
    "c eq"                 : "Carbon Equivalent",                                                   
    "carbon equivalent"    : "Carbon Equivalent",
    "carbon eq"            : "Carbon Equivalent",
    "ce long"              : "Carbon Equivalent",                             
    "ce transv"            : "Carbon Equivalent",                               
}

_ELEM_NATURAL_LIMITS: dict = {
    "Carbon"     : (0.0,  4.0),
    "Manganese"  : (0.0, 30.0),
    "Silicon"    : (0.0, 15.0),
    "Sulfur"     : (0.0,  1.5),
    "Phosphorus" : (0.0,  1.5),
    "Chromium"   : (0.0, 35.0),
    "Nickel"     : (0.0, 80.0),
    "Molybdenum" : (0.0, 30.0),
    "Nitrogen"   : (0.0,  1.5),
    "Niobium"    : (0.0, 10.0),
    "Vanadium"   : (0.0, 10.0),
    "Titanium"   : (0.0, 10.0),
    "Copper"     : (0.0, 30.0),
    "Aluminum"   : (0.0, 15.0),
    "Tungsten"   : (0.0, 35.0),
    "Iron"       : (0.0,100.0),
    "Zinc"       : (0.0, 50.0),
    "Tin"        : (0.0, 15.0),
    "Lead"       : (0.0, 10.0),
    "Oxygen"     : (0.0,  0.5),
    "Cobalt"     : (0.0, 70.0),
    "Carbon Equivalent" : (0.0,  0.60),                                                       
    "Gauge Length"      : (50.0, 1000.0),                                                             
}

def _fuzzy_map_grade(
    text           : str,
    grade_keywords : dict,
    threshold      : int = _FUZZY_THRESHOLD,
) -> tuple:
    pairs = [
        (kw, grade_key)
        for grade_key, keywords in grade_keywords.items()
        for kw in keywords
    ]
    keyword_list = [kw for kw, _ in pairs]
    if not keyword_list:
        return ("SA516-70", "SA516-70")

    result = process.extractOne(
        text, keyword_list, scorer=fuzz.partial_ratio
    )
    if result and result[1] >= threshold:
        matched_kw = result[0]
        for kw, grade_key in pairs:
            if kw == matched_kw:
                return (matched_kw, grade_key)

    return ("SA516-70", "SA516-70")

def _detect_grade_anchored(text: str) -> tuple:
    for anchor_pattern in _GRADE_ANCHORS:
        for anchor_match in re.finditer(anchor_pattern, text):
            anchor_end   = anchor_match.end()
            window_start = max(0, anchor_end - 20)
            window_end   = min(len(text), anchor_end + _ANCHOR_RADIUS)
            window       = text[window_start:window_end]

            best_kw, best_grade = "", ""
            for grade, keywords in GRADE_KEYWORDS.items():
                for kw in keywords:
                    if re.search(re.escape(kw), window, re.IGNORECASE):
                        if len(kw) > len(best_kw):                          
                            best_kw, best_grade = kw, grade
            if best_kw:
                return (best_kw, best_grade)

    best_kw, best_grade = "", ""
    for grade, keywords in GRADE_KEYWORDS.items():
        for kw in keywords:
            if re.search(re.escape(kw), text, re.IGNORECASE):
                if len(kw) > len(best_kw):                                  
                    best_kw, best_grade = kw, grade
    if best_kw:
        return (best_kw, best_grade)

    return _fuzzy_map_grade(text, GRADE_KEYWORDS)

def detect_grade(text: str) -> tuple:
    return _detect_grade_anchored(text)

def _get_patterns(grade: str) -> dict:
    family = get_family(grade)

    al_patterns = {
        "Si"   : r"\bSi\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Fe"   : r"\bFe\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Cu"   : r"\bCu\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Mn"   : r"\bMn\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Mg"   : r"\bMg\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Cr"   : r"\bCr\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Zn"   : r"\bZn\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Ti"   : r"\bTi\b[\s|:]*([0-9]+\.?[0-9]*)",
        "Al"   : r"\bAl\b[\s|:]*([0-9]+\.?[0-9]*)",
    }

    steel_patterns = {
        "Carbon"           : r"[Cc]arbon[\s:|]+([0-9.]+)|\bC\b[\s:|]+([0-9.]+)",
        "Manganese"        : r"[Mm]anganese[\s:|]+([0-9.]+)|\bMn\b[\s:|]+([0-9.]+)",
        "Phosphorus"       : r"[Pp]hosphorus[\s:|]+([0-9.]+)|\bP\b[\s:|]+([0-9.]+)",
        "Sulfur"           : r"[Ss]ulf(?:ur|phur)[\s:|]+([0-9.]+)|\bS\b[\s:|]+([0-9.]+)",
        "Silicon"          : r"[Ss]ilicon[\s:|]+([0-9.]+)|\bSi\b[\s:|]+([0-9.]+)",
        "Chromium"         : r"[Cc]hromium[\s:|]+([0-9.]+)|\bCr\b[\s:|]+([0-9.]+)",
        "Nickel"           : r"[Nn]ickel[\s:|]+([0-9.]+)|\bNi\b[\s:|]+([0-9.]+)",
        "Molybdenum"       : r"[Mm]olybdenum[\s:|]+([0-9.]+)|\bMo\b[\s:|]+([0-9.]+)",
        "Nitrogen"         : r"[Nn]itrogen[\s:|]+([0-9.]+)|\bN\b[\s:|]+([0-9.]+)",
        "Niobium"          : r"[Nn]iobium[\s:|]+([0-9.]+)|\bNb\b[\s:|]+([0-9.]+)",
        "Vanadium"         : r"[Vv]anadium[\s:|]+([0-9.]+)|\bV\b[\s:|]+([0-9.]+)",
        "Yield Strength"   : r"(?:Y\.S\.?|Yield Strength|Yield|Rp0\.2|Re)[\s:|]*([0-9.]+)",
        "Tensile Strength" : r"(?:T\.S\.?|Tensile Strength|Tensile|UTS|Rm)[\s:|]*([0-9.]+)",
        "Elongation"       : r"(?:E\.L\.?|Elongation|Elong\b|A%)[\s:|]*([0-9.]+)",
    }

    ss_patterns = {**steel_patterns,
        "Titanium"   : r"[Tt]itanium[\s:|]+([0-9.]+)|\bTi\b[\s:|]+([0-9.]+)",
        "Copper"     : r"[Cc]opper[\s:|]+([0-9.]+)|\bCu\b[\s:|]+([0-9.]+)",
    }

    ti_patterns = {
        "Iron"     : r"[Ii]ron[\s:|]+([0-9.]+)|\bFe\b[\s:|]+([0-9.]+)",
        "Oxygen"   : r"[Oo]xygen[\s:|]+([0-9.]+)|\bO\b[\s:|]+([0-9.]+)",
        "Carbon"   : r"[Cc]arbon[\s:|]+([0-9.]+)|\bC\b[\s:|]+([0-9.]+)",
        "Nitrogen" : r"[Nn]itrogen[\s:|]+([0-9.]+)|\bN\b[\s:|]+([0-9.]+)",
        "Hydrogen" : r"[Hh]ydrogen[\s:|]+([0-9.]+)|\bH\b[\s:|]+([0-9.]+)",
        "Aluminum" : r"[Aa]luminum[\s:|]+([0-9.]+)|\bAl\b[\s:|]+([0-9.]+)",
        "Vanadium" : r"[Vv]anadium[\s:|]+([0-9.]+)|\bV\b[\s:|]+([0-9.]+)",
        "Ti"       : r"\bTi\b[\s:|]*([0-9]+\.?[0-9]*)",
    }

    ni_patterns = {
        "Nickel"    : r"[Nn]ickel[\s:|]+([0-9.]+)|\bNi\b[\s:|]+([0-9.]+)",
        "Chromium"  : r"[Cc]hromium[\s:|]+([0-9.]+)|\bCr\b[\s:|]+([0-9.]+)",
        "Molybdenum": r"[Mm]olybdenum[\s:|]+([0-9.]+)|\bMo\b[\s:|]+([0-9.]+)",
        "Niobium"   : r"[Nn]iobium[\s:|]+([0-9.]+)|\bNb\b[\s:|]+([0-9.]+)",
        "Iron"      : r"[Ii]ron[\s:|]+([0-9.]+)|\bFe\b[\s:|]+([0-9.]+)",
        "Carbon"    : r"[Cc]arbon[\s:|]+([0-9.]+)|\bC\b[\s:|]+([0-9.]+)",
        "Manganese" : r"[Mm]anganese[\s:|]+([0-9.]+)|\bMn\b[\s:|]+([0-9.]+)",
        "Silicon"   : r"[Ss]ilicon[\s:|]+([0-9.]+)|\bSi\b[\s:|]+([0-9.]+)",
        "Tungsten"  : r"[Tt]ungsten[\s:|]+([0-9.]+)|\bW\b[\s:|]+([0-9.]+)",
        "Copper"    : r"[Cc]opper[\s:|]+([0-9.]+)|\bCu\b[\s:|]+([0-9.]+)",
        "Aluminum"  : r"[Aa]luminum[\s:|]+([0-9.]+)|\bAl\b[\s:|]+([0-9.]+)",
        "Titanium"  : r"[Tt]itanium[\s:|]+([0-9.]+)|\bTi\b[\s:|]+([0-9.]+)",
    }

    cu_patterns = {
        "Copper"    : r"[Cc]opper[\s:|]+([0-9.]+)|\bCu\b[\s:|]+([0-9.]+)",
        "Zinc"      : r"[Zz]inc[\s:|]+([0-9.]+)|\bZn\b[\s:|]+([0-9.]+)",
        "Tin"       : r"[Tt]in[\s:|]+([0-9.]+)|\bSn\b[\s:|]+([0-9.]+)",
        "Lead"      : r"[Ll]ead[\s:|]+([0-9.]+)|\bPb\b[\s:|]+([0-9.]+)",
        "Iron"      : r"[Ii]ron[\s:|]+([0-9.]+)|\bFe\b[\s:|]+([0-9.]+)",
        "Phosphorus": r"[Pp]hosphorus[\s:|]+([0-9.]+)|\bP\b[\s:|]+([0-9.]+)",
        "Oxygen"    : r"[Oo]xygen[\s:|]+([0-9.]+)|\bO\b[\s:|]+([0-9.]+)",
    }

    routing = {
        "Aluminum"       : al_patterns,
        "Carbon Steel"   : steel_patterns,
        "Stainless Steel": ss_patterns,
        "Titanium"       : ti_patterns,
        "Nickel Alloy"   : ni_patterns,
        "Copper Alloy"   : cu_patterns,
    }

    return routing.get(family, steel_patterns)                   

def _parse_dataframe(df: "pd.DataFrame", grade: str,
                     warn_log: list | None = None) -> dict:
    elements = {}
    column_shift_warnings: list = []                                                             

    _MECH_RANGES = {
        "Yield Strength"  : (150.0,  900.0),
        "Tensile Strength": (300.0, 1200.0),
        "Elongation"      : (  5.0,   80.0),
    }
    _CHEM_RANGE = (0.0, 99.99)                                                              

    aliases = _ELEMENT_ALIASES

    def _clean_key(raw: str) -> str:
        raw = re.sub(r'(\d),(\d)', r'\1.\2', str(raw))
        raw = raw.replace('\n', ' ')
        raw = re.sub(r'\(.*?\)', '', raw)                           
        raw = re.sub(r'\s*%\s*$', '', raw)                    
        return raw.strip().lower()

    def _to_float(cell_str: str, lo: float, hi: float):
        s = re.sub(r'(\d),(\d)', r'\1.\2', str(cell_str))
        for tok in re.findall(r'\b\d+(?:\.\d+)?\b', s):
            if re.fullmatch(r'0\d{2,}', tok):
                continue
            try:
                v = float(tok)
                if lo <= v <= hi:
                    return v
            except ValueError:
                pass
        for tok in re.findall(r'\b0\d+\b', s):
            if '.' not in tok:
                candidate = tok[0] + '.' + tok[1:]
                try:
                    v = float(candidate)
                    if lo <= v <= hi:
                        return v
                except ValueError:
                    pass
        return None

    nrows = len(df)
    ncols = len(df.columns)
    if nrows == 0 or ncols == 0:
        return elements

    text_grid = [
        [str(df.iloc[r, c]) for c in range(ncols)]
        for r in range(nrows)
    ]

    _poison = {"max", "min", "requirement", "standard",
               "limit", "aim", "specified",
               "ud", "pcs", "prime", "nominal", "size",
               "mother", "coil", "packet", "bundle",
               "heat", "cast"}

    def _is_poison_row(r: int) -> bool:
        row_words = set(" ".join(text_grid[r]).lower().split())
        return bool(row_words & _poison)

    _AMBIGUOUS = {'w', 'p', 't', 'n', 'v', 's', 'c', 'b'}

    def _is_element_cell(cell: str) -> bool:
        k = cell.strip().lower()
        if aliases.get(k) or aliases.get(_clean_key(cell)):
            return True
        leading = k.split()[0] if k.split() else ''
        return bool(leading and aliases.get(leading))

    def _row_element_count(r: int) -> int:
        return sum(1 for cell in text_grid[r] if _is_element_cell(cell))

    def _col_element_count(c: int) -> int:
        return sum(1 for row in text_grid if _is_element_cell(row[c]))

    _DIR_PREFIXES = ('ten dir ', 'ten direction ', 'dir ys', 'dir uts')

    def _is_dir_compound(raw_text: str) -> bool:
        lower = raw_text.strip().lower()
        return any(lower.startswith(p) for p in _DIR_PREFIXES)

    consumed_value_cols: set = set()

    for r in range(nrows):
        for c in range(ncols):
            raw = text_grid[r][c]
            if not raw or raw in ('nan', 'NaN', ''):
                continue

            key1 = raw.strip().lower()
            key2 = _clean_key(raw)
            element_name = aliases.get(key1) or aliases.get(key2)

            if re.fullmatch(r'[\d.\-]+', raw.strip()):
                continue

            if element_name is None:
                parts = raw.strip().split()
                if parts:
                    leading = parts[0].lower()
                    element_name = aliases.get(leading)

            if element_name is None or element_name in elements:
                continue

            if key1 in _AMBIGUOUS:
                if _row_element_count(r) < 3 and _col_element_count(c) < 3:
                    continue

            lo, hi = (
                _MECH_RANGES.get(element_name)
                or _ELEM_NATURAL_LIMITS.get(element_name, _CHEM_RANGE)
            )
            val = None

            _alt_elem: tuple | None = None
            if key1 == "n" and element_name == "Nitrogen" and "Nickel" not in elements:
                _alt_elem = ("Nickel", _ELEM_NATURAL_LIMITS.get("Nickel", _CHEM_RANGE))

            if _is_dir_compound(raw):
                scan_c = min(c + 1, ncols - 1)
                consumed_value_cols.add(scan_c)
            elif c in consumed_value_cols:
                scan_c = min(c + 1, ncols - 1)
                consumed_value_cols.add(scan_c)                                        
            else:
                scan_c = c

            if not _is_poison_row(r):
                for c2 in range(c + 1, ncols):
                    cell_r = text_grid[r][c2]
                    if cell_r == raw:
                        continue
                    v = _to_float(cell_r, lo, hi)
                    if v is not None:
                        val = v
                        print(
                            f"   [OK] '{element_name}' = {v} "
                            f"(Right [{r},{c2}])"
                        )
                        break

            if val is None:
                for r2 in range(r + 1, nrows):
                    if _is_poison_row(r2):
                        continue
                    cell_d = text_grid[r2][scan_c]
                    v = _to_float(cell_d, lo, hi)
                    if v is None and _alt_elem:
                        en2, (lo2, hi2) = _alt_elem
                        v2 = _to_float(cell_d, lo2, hi2)
                        if v2 is not None and en2 not in elements:
                            elements[en2] = v2
                            print(f"   [OK] '{en2}' = {v2} (Down-alt [{r2},{scan_c}])")
                            break
                    if v is not None:
                        val = v
                        print(
                            f"   [OK] '{element_name}' = {v} "
                            f"(Down [{r2},{scan_c}])"
                        )
                        break

            if val is None and not _is_poison_row(r):
                for c2 in range(c - 1, -1, -1):
                    cell_l = text_grid[r][c2]
                    if cell_l in ("", "nan", "NaN") or cell_l == raw:
                        continue
                    k_l = cell_l.strip().lower()
                    if aliases.get(k_l) or aliases.get(_clean_key(cell_l)):
                        continue
                    v = _to_float(cell_l, lo, hi)
                    if v is None and _alt_elem:
                        en2, (lo2, hi2) = _alt_elem
                        v2 = _to_float(cell_l, lo2, hi2)
                        if v2 is not None and en2 not in elements:
                            elements[en2] = v2
                            print(f"   [OK] '{en2}' = {v2} (Left-alt [{r},{c2}])")
                            break
                    if v is not None:
                        val = v
                        print(f"   [OK] '{element_name}' = {v} (Left [{r},{c2}])")
                        break

            if val is not None and element_name not in elements:
                elements[element_name] = val

    _chem_keys = set(_ELEM_NATURAL_LIMITS.keys())
    chem_found_count = sum(1 for k in elements if k in _chem_keys)

    elem_hdr_row = -1
    elem_hdr_cols: dict = {}                                    
    for r in range(nrows):
        if _row_element_count(r) >= 4:
            tmp = {}
            for c in range(ncols):
                cell = text_grid[r][c]
                k1 = cell.strip().lower()
                k2 = _clean_key(cell)
                en = aliases.get(k1) or aliases.get(k2)
                if en and en in _chem_keys:
                    tmp[c] = en
            if len(tmp) >= 4:
                elem_hdr_row = r
                elem_hdr_cols = tmp
                break

    if elem_hdr_row >= 0 and len(elem_hdr_cols) >= 4:
        first_elem_col = min(elem_hdr_cols.keys())
        data_rows = []
        for r in range(elem_hdr_row + 1, nrows):
            if _is_poison_row(r):
                continue
            dec_count = sum(
                1 for c in range(first_elem_col, ncols)
                if re.search(r'\b\d+\.\d+\b', text_grid[r][c])
            )
            if dec_count >= 3:
                data_rows.append(r)
            if len(data_rows) >= 3:
                break

        _grade_limits = ALL_STANDARDS.get(grade, {})

        def _grade_score(cand: dict) -> int:
            score = 0
            for en, v in cand.items():
                spec = _grade_limits.get(en, {})
                ok = True
                if "max" in spec and v > spec["max"]:
                    ok = False
                if "min" in spec and v < spec["min"]:
                    ok = False
                if spec and ok:
                    score += 1
            return score

        best_shift, best_count, best_grade_score, best_extra = 0, 0, 0, {}

        initial_grade_score = _grade_score(
            {k: v for k, v in elements.items() if k in _chem_keys}
        )
        for shift in range(1, 9):
            candidate: dict = {}
            for elem_col, en in elem_hdr_cols.items():
                nat_lo, nat_hi = _ELEM_NATURAL_LIMITS.get(en, _CHEM_RANGE)
                target_col = elem_col + shift
                if target_col >= ncols:
                    continue
                for dr in data_rows:
                    v = _to_float(text_grid[dr][target_col],
                                  nat_lo, nat_hi)
                    if v is not None:
                        candidate[en] = v
                        break
            nat_count   = len(candidate)
            grade_score = _grade_score(candidate)
            if (grade_score, nat_count) > (best_grade_score, best_count):
                best_count       = nat_count
                best_grade_score = grade_score
                best_shift       = shift
                best_extra       = dict(candidate)

        if best_count >= 3 and (
            best_count > chem_found_count
            or chem_found_count < 3
            or best_grade_score > initial_grade_score
        ):
            print(
                f"   [SHIFT] Column-shift recovery: shift=+{best_shift}, "
                f"{best_count} elements "
                f"(grade-score {initial_grade_score}->{best_grade_score}) "
                f"{list(best_extra.keys())}"
            )
            elements.update(best_extra)
            for en in elem_hdr_cols.values():
                if en in _chem_keys and en not in best_extra and en in elements:
                    del elements[en]
                    print(f"   [SHIFT] Cleared stale '{en}' (not at shifted pos)")

        if data_rows:
            claimed_vals = set(
                round(v, 8) for v in elements.values() if isinstance(v, float)
            )
            grade_chem_elems = sorted(
                en for en in _grade_limits if en in _chem_keys
            )
            consumed_vals    = set()                                            
            freed_for_rescue = set()                                         
            freed_elements   = set()                                               
            column_shift_warnings = []                                                 

            for en in list(grade_chem_elems):
                v    = elements.get(en)
                if v is None:
                    continue
                spec = _grade_limits.get(en, {})
                if not spec:
                    continue
                ok_here = (
                    ("max" not in spec or v <= spec["max"])
                    and ("min" not in spec or v >= spec["min"])
                )
                if ok_here:
                    continue                                             
                passing_targets = []
                for other in grade_chem_elems:
                    if other == en:
                        continue
                    other_spec = _grade_limits.get(other, {})
                    if not other_spec:
                        continue
                    other_v = elements.get(other)
                    if other_v is not None:
                        ok_other = (
                            ("max" not in other_spec or other_v <= other_spec["max"])
                            and ("min" not in other_spec or other_v >= other_spec["min"])
                        )
                        if ok_other:
                            continue
                    if (
                        ("max" not in other_spec or v <= other_spec["max"])
                        and ("min" not in other_spec or v >= other_spec["min"])
                    ):
                        passing_targets.append(other)
                if len(passing_targets) == 1:                                
                    target = passing_targets[0]
                    print(f"   [SHIFT] Pre-rescue: freed '{en}'={v}"
                          f" -> unique target '{target}'")
                    column_shift_warnings.append({
                        "extracted_as" : en,
                        "value"        : v,
                        "likely_is"    : target,
                        "detail"       : (
                            f"Value {v} fails {en} spec but uniquely passes "
                            f"{target} spec — OCR column displacement suspected"
                        ),
                    })
                    del elements[en]
                    rv = round(v, 8)
                    claimed_vals.discard(rv)
                    freed_for_rescue.add(rv)
                    freed_elements.add(en)

            def _scan_rescue(elems_to_try, min_score):
                for en in elems_to_try:
                    current_val = elements.get(en)
                    spec        = _grade_limits.get(en, {})
                    if current_val is not None:
                        if spec:
                            ok_current = (
                                ("max" not in spec or current_val <= spec["max"])
                                and ("min" not in spec or current_val >= spec["min"])
                            )
                        else:
                            ok_current = True                                     
                        if ok_current:
                            continue                                
                    nat_lo, nat_hi = _ELEM_NATURAL_LIMITS.get(en, _CHEM_RANGE)
                    grade_spec     = spec
                    best_rval, best_rscore = None, -1
                    for col in range(ncols):
                        for dr in data_rows:
                            cell = text_grid[dr][col]
                            if re.search(r'[A-Za-z]', cell):
                                continue
                            v = _to_float(cell, nat_lo, nat_hi)
                            if v is None:
                                continue
                            rv = round(v, 8)
                            if rv in consumed_vals:
                                continue                                    
                            if rv in claimed_vals and rv not in freed_for_rescue:
                                continue                          
                            ok = (
                                ("max" not in grade_spec or v <= grade_spec["max"])
                                and ("min" not in grade_spec or v >= grade_spec["min"])
                            )
                            score = 2 if (ok and grade_spec) else 1
                            if score > best_rscore:
                                best_rscore, best_rval = score, v
                    if best_rval is not None and best_rscore >= min_score:
                        if current_val is not None:
                            old_rv = round(current_val, 8)
                            consumed_vals.add(old_rv)
                            claimed_vals.discard(old_rv)
                            print(f"   [RESCUE] Override '{en}': "
                                  f"{current_val} -> {best_rval} "
                                  f"(score={best_rscore})")
                        else:
                            print(f"   [RESCUE] '{en}' = {best_rval} "
                                  f"(col-independent scan, score={best_rscore})")
                        elements[en] = best_rval
                        new_rv = round(best_rval, 8)
                        claimed_vals.add(new_rv)
                        freed_for_rescue.discard(new_rv)

            _scan_rescue(grade_chem_elems, min_score=2)                                 

            no_spec_elems = [en for en in grade_chem_elems if not _grade_limits.get(en)]
            _scan_rescue(no_spec_elems, min_score=1)                                        

            if freed_elements:
                grade_limits_exact = set()
                for lim in _grade_limits.values():
                    if "min" in lim:
                        grade_limits_exact.add(round(lim["min"], 8))
                    if "max" in lim:
                        grade_limits_exact.add(round(lim["max"], 8))

                for en in grade_chem_elems:
                    if en not in freed_elements:
                        continue                                                     
                    if en in elements:
                        continue                                                
                    spec = _grade_limits.get(en, {})
                    if not spec:
                        continue
                    nat_lo, nat_hi = _ELEM_NATURAL_LIMITS.get(en, _CHEM_RANGE)
                    best_val = None
                    for row_idx in range(nrows):                                     
                        for col_idx in range(ncols):
                            cell = text_grid[row_idx][col_idx]
                            if re.search(r'[A-Za-z]', cell):
                                continue
                            v = _to_float(cell, nat_lo, nat_hi)
                            if v is None:
                                continue
                            rv = round(v, 8)
                            if rv in claimed_vals or rv in consumed_vals:
                                continue
                            if rv in grade_limits_exact:
                                continue                                               
                            ok = (
                                ("max" not in spec or v <= spec["max"])
                                and ("min" not in spec or v >= spec["min"])
                            )
                            if ok and best_val is None:
                                best_val = v                                       
                    if best_val is not None:
                        elements[en] = best_val
                        claimed_vals.add(round(best_val, 8))
                        print(f"   [RESCUE-EXT] '{en}' = {best_val} "
                              f"(full-grid scan, freed element)")

    min_mech_col = 0
    if elem_hdr_row >= 0 and best_shift > 0:
        for _c in range(ncols):
            _raw = text_grid[elem_hdr_row][_c]
            _en  = aliases.get(_raw.strip().lower()) or aliases.get(_clean_key(_raw))
            if _en == "Yield Strength":
                min_mech_col = _c + best_shift
                break

    _mech_triplets = [
        ("Yield Strength",   150.0,  900.0),
        ("Tensile Strength", 300.0, 1200.0),
        ("Elongation",         5.0,   80.0),
    ]
    if not all(mk in elements for mk, *_ in _mech_triplets):
        found_triplet = False
        for r in range(nrows):
            if _is_poison_row(r) or found_triplet:
                continue
            row_cells = text_grid[r]
            for c_start in range(min_mech_col, ncols - 2):
                y = _to_float(row_cells[c_start],     150.0,  900.0)
                if y is None:
                    continue
                u = _to_float(row_cells[c_start + 1], 300.0, 1200.0)
                if u is None:
                    continue
                e = _to_float(row_cells[c_start + 2],   5.0,   80.0)
                if e is None:
                    continue
                if "Yield Strength"   not in elements:
                    elements["Yield Strength"]   = y
                if "Tensile Strength" not in elements:
                    elements["Tensile Strength"] = u
                if "Elongation"       not in elements:
                    elements["Elongation"]       = e
                print(
                    f"   [OK] Mech triplet: YS={y} UTS={u} El={e} "
                    f"(row {r}, cols {c_start}-{c_start+2})"
                )
                found_triplet = True
                break

    if warn_log is not None and column_shift_warnings:
        warn_log.extend(column_shift_warnings)
    return elements

def _parse_regex(text: str, grade: str) -> dict:
    elements = {}
    patterns = _get_patterns(grade)

    _MECH_R = {
        "Yield Strength"  : (150.0,  900.0),
        "Tensile Strength": (300.0, 1200.0),
        "Elongation"      : (  5.0,   80.0),
    }
    _CHEM_R = (0.0001, 99.99)

    for element, pattern in patterns.items():
        lo, hi = _MECH_R.get(element, _CHEM_R)
        for m in re.finditer(pattern, text):
            val = next((g for g in m.groups() if g), None)
            if val:
                try:
                    v = float(val)
                    if lo <= v <= hi:
                        elements[element] = v
                        break                                     
                except ValueError:
                    continue

    print(f"[REGEX]  Matched elements: {list(elements.keys())}")
    return elements

def _markdown_to_dataframes(md_text: str) -> list:
    tables = []
    block  = []

    def _flush(blk):
        if len(blk) < 2:
            return None
        data_lines = [
            l for l in blk
            if not re.fullmatch(r'[\|\s\-:]+', l.strip())
        ]
        if not data_lines:
            return None
        rows = []
        for line in data_lines:
            stripped = line.strip().strip('|')
            cells    = [c.strip() for c in stripped.split('|')]
            rows.append(cells)
        max_cols = max(len(r) for r in rows)
        rows     = [r + [''] * (max_cols - len(r)) for r in rows]
        try:
            df = pd.DataFrame(rows)
            return df if not df.empty else None
        except Exception:
            return None

    for line in md_text.splitlines():
        if '|' in line:
            block.append(line)
        else:
            if block:
                df = _flush(block)
                if df is not None:
                    tables.append(df)
                block = []

    if block:                                              
        df = _flush(block)
        if df is not None:
            tables.append(df)

    print(f"[PATH-A] Pipe-table blocks found: {len(tables)}")
    return tables

_ELEMENT_TOKENS = frozenset({
    'c', 'mn', 'p', 's', 'si', 'cr', 'ni', 'mo', 'n', 'nb', 'v', 'ti',
    'w', 'cu', 'zn', 'sn', 'pb', 'fe', 'o', 'h', 'al', 'mg',
    'carbon', 'manganese', 'phosphorus', 'sulfur', 'sulphur', 'silicon',
    'chromium', 'nickel', 'molybdenum', 'nitrogen', 'niobium', 'vanadium',
    'titanium', 'tungsten', 'copper', 'zinc', 'iron', 'aluminum', 'aluminium',
    'oxygen', 'hydrogen', 'lead', 'tin',
    'yield', 'tensile', 'elongation',
})

def _parse_column_blocks(text: str) -> list:
    def _tokenize(line):
        return [(m.group(), m.start()) for m in re.finditer(r'\S+', line)]

    def _is_element_token(tok: str) -> bool:
        t = re.sub(r'[^a-zA-Z0-9]', '', tok).lower()
        return bool(t) and t in _ELEMENT_TOKENS

    def _is_numeric(tok: str) -> bool:
        return bool(re.fullmatch(r'\d+\.?\d*', tok.strip()))

    lines  = text.splitlines()
    tables = []
    used   = set()                                                    

    for i, line in enumerate(lines):
        if i in used:
            continue
        tokens = _tokenize(line)
        if not tokens:
            continue

        elem_hits = sum(1 for t, _ in tokens if _is_element_token(t))
        if elem_hits < 3:
            continue

        header_tokens = tokens                                      
        header_names  = [t for t, _ in header_tokens]
        header_cols   = [c for _, c in header_tokens]
        print(f"[PATH-A2] Header detected  : {header_names}")

        value_rows = []
        for j in range(i + 1, min(i + 9, len(lines))):
            if j in used:
                continue
            vt = _tokenize(lines[j])
            if sum(1 for t, _ in vt if _is_numeric(t)) >= 3:
                value_rows.append((j, vt))

        if not value_rows:
            continue

        rows = [header_names]
        for j, vt in value_rows:
            row = [''] * len(header_names)
            for tok, col in vt:
                if _is_numeric(tok):
                    dists   = [abs(col - hc) for hc in header_cols]
                    closest = dists.index(min(dists))
                    if row[closest] == '':                             
                        row[closest] = tok
            rows.append(row)
            used.add(j)

        used.add(i)
        try:
            df = pd.DataFrame(rows)
            if not df.empty:
                tables.append(df)
                print(f"[PATH-A2]   Values mapped  : {rows[1:]}")
        except Exception:
            pass

    print(f"[PATH-A2] Column-block tables found: {len(tables)}")
    return tables

def _html_to_grid(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    raw_rows = table.find_all("tr")
    if not raw_rows:
        return []

    occupied = {}                 

    for ri, row in enumerate(raw_rows):
        ci = 0
        for cell in row.find_all(["td", "th"]):
            while (ri, ci) in occupied:
                ci += 1
            txt = cell.get_text(" ", strip=True)
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    pos = (ri + dr, ci + dc)
                    if pos not in occupied:
                        occupied[pos] = txt
            ci += cs

    if not occupied:
        return []

    num_rows = max(r for r, _ in occupied) + 1
    num_cols = max(c for _, c in occupied) + 1
    grid = [
        [occupied.get((r, c), "") for c in range(num_cols)]
        for r in range(num_rows)
    ]
    return grid

def _parse_html_table_bs4(html: str, grade: str,
                          warn_log: list | None = None) -> dict:
    elements = {}
    try:
        grid = _html_to_grid(html)
        if not grid:
            return elements
        ncols = max(len(row) for row in grid)
        grid  = [row + [""] * (ncols - len(row)) for row in grid]
        df    = pd.DataFrame(grid)
        elements = _parse_dataframe(df, grade, warn_log=warn_log)
        print(f"[BS4]    HTML-grid {len(grid)}x{ncols}, found {len(elements)} elements")
    except Exception as e:
        print(f"[BS4]    _parse_html_table_bs4 failed: {e}")
    return elements

def parse_chemical_values(
    text       : str,
    grade      : str,
    html_tables: list = None,
    warn_log   : list | None = None,
) -> dict:
    elements = {}
    print(f"\n[EXTRACT] Grade = {grade!r}")

    if html_tables:
        for html in html_tables:
            partial = _parse_html_table_bs4(html, grade, warn_log=warn_log)
            elements.update(partial)
        print(f"[EXTRACT] PATH 0  (HTML tables)   -> {len(elements)} elements: {list(elements.keys())}")

        grade_spec = ALL_STANDARDS.get(grade, {})
        missing    = [en for en in grade_spec if en not in elements]
        if missing:
            supplement = _parse_regex(text, grade)
            filled = [k for k, v in supplement.items() if k not in elements]
            elements.update({k: supplement[k] for k in filled})
            if filled:
                print(f"[EXTRACT] PATH 0+B supplement   -> filled {len(filled)} via regex: {filled}")

    if not elements:
        for df in _markdown_to_dataframes(text):
            partial = _parse_dataframe(df, grade, warn_log=warn_log)
            elements.update(partial)
        print(f"[EXTRACT] PATH A  (pipe tables)   -> {len(elements)} elements: {list(elements.keys())}")

    if not elements:
        for df in _parse_column_blocks(text):
            partial = _parse_dataframe(df, grade, warn_log=warn_log)
            elements.update(partial)
        print(f"[EXTRACT] PATH A2 (column blocks) -> {len(elements)} elements: {list(elements.keys())}")

    if not elements:
        elements = _parse_regex(text, grade)
        print(f"[EXTRACT] PATH B  (regex)         -> {len(elements)} elements: {list(elements.keys())}")

    return elements

def _merge_limits(
    astm_limits: dict,
    custom_limits: dict
) -> dict:
    merged = {}

    for element, limit in astm_limits.items():
        merged[element] = dict(limit)        

    for element, custom in custom_limits.items():
        if element not in merged:
            merged[element] = dict(custom)
        else:
            base = merged[element]

            if "max" in custom and custom["max"] is not None:
                if "max" not in base or base["max"] is None:
                    base["max"] = custom["max"]
                else:
                    base["max"] = min(base["max"], custom["max"])

            if "min" in custom and custom["min"] is not None:
                if "min" not in base or base["min"] is None:
                    base["min"] = custom["min"]
                else:
                    base["min"] = max(base["min"], custom["min"])

            merged[element] = base

    return merged

def verify_mtc(
    markdown_text : str,
    mtc_name      : str,
    custom_limits : dict = None,                        
    html_tables   : list = None,                               
) -> dict:
    if isinstance(markdown_text, dict):
        text = markdown_text.get('markdown_texts', str(markdown_text))
    else:
        text = str(markdown_text)

    if html_tables:
        from bs4 import BeautifulSoup
        html_text_parts = []
        for html in html_tables:
            try:
                soup = BeautifulSoup(html, "html.parser")
                html_text_parts.append(soup.get_text(" "))
            except Exception:
                pass
        if html_text_parts:
            text = "\n".join(html_text_parts) + "\n" + text

    text = re.sub(r'(\d),(\d)', r'\1.\2', text)
    text = text.replace('%', ' % ')

    multiplier_match = re.search(
        r'\b[Xx]10\b|\b[Xx]100\b|\b[Xx]1000\b', text
    )
    has_multiplier = bool(multiplier_match)

    raw_grade, mapped_grade = detect_grade(text)
    print(f"\n[VERIFY] {'='*55}")
    print(f"[VERIFY] MTC      : {mtc_name}")
    print(f"[VERIFY] Grade    : raw={raw_grade!r}  mapped={mapped_grade!r}")
    astm_limits  = ALL_STANDARDS.get(mapped_grade, ALL_STANDARDS["SA516-70"])

    if custom_limits:
        for elem, lim in custom_limits.items():
            if "unit" not in lim:
                lim["unit"] = "%"
        active_limits = _merge_limits(astm_limits, custom_limits)
    else:
        active_limits = astm_limits

    shift_log  = []                                                          
    elements   = parse_chemical_values(text, mapped_grade,
                                       html_tables=html_tables, warn_log=shift_log)

    heat_no = _extract_heat_from_html(html_tables) or extract_traceability(text)
    packet_no = _extract_packet_from_html(html_tables)                                   
    print(f"[VERIFY] Heat No  : {heat_no}")
    print(f"[VERIFY] Packet No: {packet_no}")       
    print(f"[VERIFY] Elements : {elements}")

    _mech_keys = ("Yield Strength", "Tensile Strength", "Elongation")
    has_mech   = any(k in elements for k in _mech_keys)

    mech_props = {"Yield Strength": "N/A", "Tensile Strength": "N/A", "Elongation": "N/A"}

    if not elements and heat_no == "Not Found" and not has_mech:
        report = {
            "mtc_name"     : mtc_name,
            "grade"        : raw_grade,
            "family"       : get_family(mapped_grade),
            "verdict"      : "[WARN] INVALID MTC",
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details"      : [],
            "custom_used"  : bool(custom_limits),
            "custom_limits": custom_limits or {},
            "heat_number"  : heat_no,
            "mechanicals"  : mech_props,
        }
        _save_audit(report)
        return report
    results = []
    overall = "PASS"

    for element, limit in active_limits.items():
        extracted = elements.get(element)
        max_val   = limit.get("max")
        min_val   = limit.get("min")

        astm_limit  = astm_limits.get(element, {})
        astm_max    = astm_limit.get("max")
        astm_min    = astm_limit.get("min")

        custom_limit = (custom_limits or {}).get(element, {})
        custom_max   = custom_limit.get("max")
        custom_min   = custom_limit.get("min")

        if extracted is None:
            status       = "[WARN] NOT FOUND"
            fail_reason  = "-"
        else:
            failed      = False
            suspect_ocr = False                                                
            fail_reason = []

            if max_val is not None and extracted > max_val:
                if extracted > (max_val * 5):
                    suspect_ocr = True
                    fail_reason.append(
                        f"Suspect OCR: {extracted} >> limit {max_val} "
                        f"(possible missed multiplier)"
                    )
                else:
                    failed = True
                    if custom_max is not None and extracted > custom_max:
                        fail_reason.append("Custom max exceeded")
                    elif astm_max is not None and extracted > astm_max:
                        fail_reason.append("International Standard max exceeded")

            if min_val is not None and extracted < min_val:
                failed = True
                if custom_min is not None and extracted < custom_min:
                    fail_reason.append("Custom min not met")
                elif astm_min is not None and extracted < astm_min:
                    fail_reason.append("International Standards min not met")

            if suspect_ocr:
                status  = "[WARN] MANUAL REVIEW"
                if overall != "FAIL":                                       
                    overall = "[WARN] MANUAL VERIFICATION REQUIRED"
            elif failed:
                status  = "[FAIL] FAIL"
                overall = "FAIL"
            else:
                status  = "[OK] PASS"

            fail_reason = ", ".join(fail_reason) if fail_reason else "-"

        active_max = limit.get("max", "-")
        active_min = limit.get("min", "-")

        display_max = (
            f"★ {active_max}"
            if custom_max is not None
            and astm_max is not None
            and custom_max < astm_max
            else active_max
        )
        display_min = (
            f"★ {active_min}"
            if custom_min is not None
            and astm_min is not None
            and custom_min > astm_min
            else active_min
        )

        results.append({
            "Element"    : element,
            "Extracted"  : extracted,
            "Min Limit"  : str(display_min) if display_min is not None else "-",
            "Max Limit"  : str(display_max) if display_max is not None else "-",
            "Unit"       : limit.get("unit", "%"),
            "Status"     : status,
            "Fail Reason": fail_reason,
        })

    valid_extractions = sum(1 for r in results if r["Status"] != "[WARN] NOT FOUND")
    if valid_extractions == 0 and overall == "PASS":
        overall = "[WARN] MANUAL VERIFICATION REQUIRED"

    if overall == "PASS":
        for r in results:
            if r["Status"] == "[WARN] NOT FOUND":
                el_limit = active_limits.get(r["Element"], {})
                if el_limit.get("min") is not None:
                    overall = "[WARN] MANUAL VERIFICATION REQUIRED"
                    break

    if has_multiplier and overall != "FAIL":
        overall = "[WARN] MANUAL VERIFICATION REQUIRED"

    mech_props = {
        d["Element"]: (
            f"{d['Extracted']} {d['Unit']}" if d["Extracted"] is not None else "N/A"
        )
        for d in results
        if d["Element"] in _mech_keys
    }
    for k in _mech_keys:
        mech_props.setdefault(k, "N/A")

    report = {
        "mtc_name"     : mtc_name,
        "grade"        : raw_grade,                                   
        "family"       : get_family(mapped_grade),
        "verdict"      : overall,
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details"      : results,
        "custom_used"  : bool(custom_limits),                          
        "custom_limits": custom_limits or {},
        "heat_number"  : heat_no,
        "packet_number": packet_no,                               
        "mechanicals"  : mech_props,
        "column_shift_warning"    : shift_log,                                                 
        "mech_duplication_warning": False,                                                               
    }

    print(f"[VERIFY] Verdict  : {overall}")
    print(f"[VERIFY] {'='*55}\n")
    _save_audit(report)
    return report

_TRACE_FALSE_POSITIVES = {"size", "packet", "nominal", "mother", "n/a", "none"}

_BANNED_HEAT_KEYWORDS = {
    "chief", "inspector", "signature", "officer", "approved", "witnessed",
    "date:", "date :", "signed", "by :", "by:", "checked", "verified",
    "authorised", "authorized", "department", "division", "section",
}

_HEAT_KEYWORDS = {"heat", "cast", "melt", "batch", "heai", "heai no", "heat no"}

_PACKET_KEYWORDS = {"packet", "coil", "lot", "bundle", "pkt", "coil no", "lot no", "bundle no", "packet no"}

def _is_valid_heat(val: str) -> bool:
    val = val.strip()
    if len(val) < 3 or len(val) > 25:
        return False
    if not re.search(r'\d', val):
        return False
    if val.lower() in _TRACE_FALSE_POSITIVES:
        return False

    coil_indicators = {"mc", "coil", "lot", "bundle", "pkt", "packet"}
    val_lower = val.lower()

    if "/" in val:
        parts = [p.strip() for p in val.split("/")]
        if len(parts) > 1:
            if any(parts[1].lower().startswith(ind) for ind in coil_indicators):
                val = parts[0]
                val_lower = val.lower()

    if any(val_lower.startswith(ind) or f" {ind}" in val_lower for ind in coil_indicators):
        return False                                         

    if any(kw in val_lower for kw in _BANNED_HEAT_KEYWORDS):
        return False
    if re.fullmatch(r'\d+\.\d+', val):
        return False
    return True

def _is_alphanumeric_heat(val: str) -> bool:
    return bool(re.search(r'[A-Za-z]', val) and re.search(r'\d', val))

def _extract_heat_from_html(html_tables: list) -> str:
    if not html_tables:
        return None

    for html in html_tables:
        try:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            all_rows_cells = [
                [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                for row in rows
            ]
            for ri, cell_texts in enumerate(all_rows_cells):
                for ci, ct in enumerate(cell_texts):
                    if not any(kw in ct.lower() for kw in _HEAT_KEYWORDS):
                        continue

                    print(f"[HEAT] >>> FOUND HEADER '{ct}' at row {ri}, col {ci}")

                    down_candidates = []
                    max_row = min(ri + 6, len(all_rows_cells))                             
                    print(f"[HEAT]   Scanning DOWN: rows {ri + 1} to {max_row - 1}, col {ci}")
                    for ri2 in range(ri + 1, max_row):
                        row2 = all_rows_cells[ri2]
                        if ci < len(row2):
                            val = row2[ci].strip()
                            is_valid = _is_valid_heat(val)
                            print(f"[HEAT]     [{ri2},{ci}] '{val}' -> valid={is_valid}")
                            if is_valid:
                                down_candidates.append(val)
                    for ci2 in range(ci + 1, min(ci + 4, len(cell_texts))):
                        for ri2 in range(ri + 1, max_row):
                            row2 = all_rows_cells[ri2]
                            if ci2 < len(row2):
                                val = row2[ci2].strip()
                                if _is_valid_heat(val) and _is_alphanumeric_heat(val):
                                    down_candidates.append(val)
                                    print(f"[HEAT]     [{ri2},{ci2}] '{val}' -> alpha=True (added)")
                    print(f"[HEAT]   DOWN candidates: {down_candidates}")
                    alpha = min((v for v in down_candidates if _is_alphanumeric_heat(v)), key=len, default=None)
                    if alpha:
                        print(f"[HEAT]  >>> SELECTED (shortest): {alpha!r}")
                        return alpha.upper()
                    if down_candidates:
                        print(f"[HEAT]  >>> SELECTED (first): {down_candidates[0]!r}")
                        return down_candidates[0].upper()

                    print(f"[HEAT]   Scanning RIGHT: row {ri}, cols {ci + 1} to end")
                    right_candidates = []
                    for j in range(ci + 1, len(cell_texts)):
                        val = cell_texts[j].strip()
                        is_valid = _is_valid_heat(val)
                        if is_valid:
                            right_candidates.append(val)
                            print(f"[HEAT]     [{ri},{j}] '{val}' -> valid=True")
                    print(f"[HEAT]   RIGHT candidates: {right_candidates}")
                    alpha = min((v for v in right_candidates if _is_alphanumeric_heat(v)), key=len, default=None)
                    if alpha:
                        print(f"[HEAT]  >>> SELECTED (shortest): {alpha!r}")
                        return alpha.upper()
                    if right_candidates:
                        print(f"[HEAT]  >>> SELECTED (first): {right_candidates[0]!r}")
                        return right_candidates[0].upper()

        except Exception:
            pass
    return None

def extract_per_heat_documents(html_tables: list) -> list:
    if not html_tables:
        return []

    _MECH_RANGES = {
        "Yield Strength"  : (150.0,  900.0),
        "Tensile Strength": (300.0, 1200.0),
        "Elongation"      : (  5.0,   80.0),
    }

    def _clean_key(raw: str) -> str:
        raw = re.sub(r'(\d),(\d)', r'\1.\2', str(raw))
        raw = raw.replace('\n', ' ')
        raw = re.sub(r'\(.*?\)', '', raw)
        raw = re.sub(r'\s*%\s*$', '', raw)
        return raw.strip().lower()

    def _is_elem_cell(cell: str) -> bool:
        k1 = cell.strip().lower()
        return bool(_ELEMENT_ALIASES.get(k1) or _ELEMENT_ALIASES.get(_clean_key(cell)))

    _poison = {"max", "min", "requirement", "standard", "limit", "aim",
               "specified", "ud", "pcs", "prime", "nominal", "size",
               "mother", "coil", "packet", "bundle"}

    def _is_poison_cells(cells: list) -> bool:
        return bool(set(" ".join(cells).lower().split()) & _poison)

    for html in html_tables:
        try:
            grid = _html_to_grid(html)
            if not grid:
                continue
            nrows = len(grid)
            ncols = max(len(row) for row in grid)
            grid  = [row + [""] * (ncols - len(row)) for row in grid]

            elem_hdr_row = -1
            heat_col     = -1
            for r in range(nrows):
                elem_count = sum(1 for cell in grid[r] if _is_elem_cell(cell))
                if elem_count < 4:
                    continue
                keyword_col = -1
                for scan_r in [r] + [r + d for d in (-1, 1, 2) if 0 <= r + d < nrows]:
                    for c in range(ncols):
                        if any(kw in grid[scan_r][c].lower() for kw in _HEAT_KEYWORDS):
                            keyword_col = c
                            break
                    if keyword_col >= 0:
                        break

                def _heat_counts_at(col: int, start_row: int):
                    alpha, total = 0, 0
                    for dr in range(start_row + 1, nrows):
                        if _is_poison_cells(grid[dr]):
                            continue
                        if col < ncols:
                            v = grid[dr][col].strip()
                            if _is_valid_heat(v):
                                total += 1
                                if _is_alphanumeric_heat(v):
                                    alpha += 1
                    return alpha, total

                h_col = -1
                alpha_k, total_k = 0, 0
                if keyword_col >= 0:
                    alpha_k, total_k = _heat_counts_at(keyword_col, r)
                    if alpha_k >= 1:
                        h_col = keyword_col

                if h_col < 0:
                    best_alpha, best_total, best_col = 0, 0, -1
                    for c in range(ncols):
                        a, t = _heat_counts_at(c, r)
                        if a > best_alpha or (a == best_alpha and t > best_total):
                            best_alpha, best_total, best_col = a, t, c
                    if best_alpha >= 2:
                        h_col = best_col
                        print(f"[MULTI-HEAT] heat_col fallback (alpha): keyword at col "
                              f"{keyword_col}, alphanumeric heats at col {h_col}")
                    elif keyword_col >= 0 and best_alpha == 0 and total_k >= 2:
                        h_col = keyword_col
                        print(f"[MULTI-HEAT] heat_col fallback (int): keyword col {h_col}")

                if h_col >= 0:
                    elem_hdr_row = r
                    heat_col     = h_col
                    break

            if elem_hdr_row < 0 or heat_col < 0:
                continue                                                    

            data_heats = []                                                   
            for r in range(elem_hdr_row + 1, nrows):
                if _is_poison_cells(grid[r]):
                    continue
                dec_count = sum(
                    1 for c in range(ncols)
                    if re.search(r'\b\d+\.\d+\b', grid[r][c])
                )
                if dec_count < 3:
                    continue
                raw_heat = grid[r][heat_col].strip() if heat_col < ncols else ""
                if not _is_valid_heat(raw_heat):
                    for dc in [-1, 1, 2, 3]:
                        alt = heat_col + dc
                        if 0 <= alt < ncols:
                            v = grid[r][alt].strip()
                            if _is_valid_heat(v):
                                raw_heat = v
                                break
                if raw_heat and _is_valid_heat(raw_heat):
                    data_heats.append(raw_heat.upper())

            if len(data_heats) < 2:
                continue                                             

            soup    = BeautifulSoup(html, "html.parser")
            tr_tags = soup.find_all("tr")

            hdr_idx = -1
            for tr_idx, tr in enumerate(tr_tags):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if sum(1 for cell in cells if _is_elem_cell(cell)) >= 4:
                    hdr_idx = tr_idx
                    break

            if hdr_idx < 0:
                continue

            first_heat_no     = data_heats[0]
            first_heat_tr_idx = -1
            for i_tr, tr in enumerate(tr_tags):
                if i_tr <= hdr_idx:                                                           
                    continue
                if first_heat_no in tr.get_text().upper():
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                    if not _is_poison_cells(cells):
                        first_heat_tr_idx = i_tr
                        break

            if first_heat_tr_idx > hdr_idx + 1:
                header_html = "".join(str(tr_tags[i]) for i in range(first_heat_tr_idx))
            else:
                header_html = "".join(str(tr_tags[i]) for i in range(hdr_idx + 1))

            results = []
            for heat_no in data_heats:
                matching_tr_html = None
                for tr in tr_tags[hdr_idx + 1:]:
                    tr_text = tr.get_text()
                    if heat_no in tr_text.upper():
                        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                        if not _is_poison_cells(cells):
                            matching_tr_html = str(tr)
                            break
                if matching_tr_html:
                    snippet = f"<table>{header_html}{matching_tr_html}</table>"
                    results.append({"heat_no": heat_no, "html": snippet})

            if len(results) >= 2:
                print(f"[MULTI-HEAT] Detected {len(results)} heat rows")
                for rr in results:
                    print(f"[MULTI-HEAT]   {rr['heat_no']}")
                return results

        except Exception as e:
            print(f"[MULTI-HEAT] Exception: {e}")
            continue

    return []

def extract_traceability(text: str) -> str:
    if not text:
        return None

    patterns = [
        r"[Hh]eat\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
        r"[Cc]ast\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
        r"[Cc]ertificate\s*[Nn]o\.?\s*[:/]?\s*([A-Z0-9\-]+)",
        r"[Hh]eat\s*[Nn]umber\s*[:/]?\s*([A-Z0-9\-]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().upper()
            if _is_valid_heat(val):
                return val

    return None

def _extract_packet_from_html(html_tables: list) -> str:
    if not html_tables:
        return None

    for html in html_tables:
        try:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            all_rows_cells = [
                [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                for row in rows
            ]
            for ri, cell_texts in enumerate(all_rows_cells):
                for ci, ct in enumerate(cell_texts):
                    if not any(kw in ct.lower() for kw in _PACKET_KEYWORDS):
                        continue

                    down_candidates = []
                    max_row = min(ri + 6, len(all_rows_cells))
                    for ri2 in range(ri + 1, max_row):
                        row2 = all_rows_cells[ri2]
                        if ci < len(row2):
                            val = row2[ci].strip()
                            if len(val) >= 3 and len(val) <= 25 and re.search(r'\d', val):
                                if not re.fullmatch(r'\d+\.\d+', val):
                                    down_candidates.append(val)

                    if down_candidates:
                        result = down_candidates[0].upper()
                        print(f"[PACKET]  HTML DOWN match: {result!r}")
                        return result

                    right_candidates = []
                    for j in range(ci + 1, len(cell_texts)):
                        val = cell_texts[j].strip()
                        if len(val) >= 3 and len(val) <= 25 and re.search(r'\d', val):
                            if not re.fullmatch(r'\d+\.\d+', val):
                                right_candidates.append(val)

                    if right_candidates:
                        result = right_candidates[0].upper()
                        print(f"[PACKET]  HTML RIGHT match: {result!r}")
                        return result

        except Exception:
            pass
    return None

    clean_text = re.sub(r'</?(?:td|tr|th|table|tbody|div|span)[^>]*>', ' ', text, flags=re.IGNORECASE)

    pattern = re.compile(
        r'\b(?:heat|cast|melt|batch)\b\s*(?:no\.?|number|#)?'
        r'[\s\S]{0,80}?'
        r'\b([A-Z0-9\-\/]*\d[A-Z0-9\-\/]*)\b',
        re.IGNORECASE
    )
    candidates = []
    for m in pattern.finditer(clean_text):
        candidate = m.group(1).strip()
        if len(candidate) >= 3 and candidate.lower() not in _TRACE_FALSE_POSITIVES:
            candidates.append(candidate)

    if not candidates:
        return "Not Found"
    alpha = next((c for c in candidates if _is_alphanumeric_heat(c)), None)
    return (alpha or candidates[0])
    return "Not Found"

def _save_audit(report: dict):
    row = {
        "Timestamp"    : report["timestamp"],
        "MTC Name"     : report["mtc_name"],
        "Grade"        : report["grade"],
        "Family"       : report["family"],
        "Verdict"      : report["verdict"],
        "Custom Used"  : report["custom_used"],
        "Total Elements": len(report["details"]),
        "Passed"       : sum(
            1 for d in report["details"]
            if d["Status"] == "[OK] PASS"
        ),
        "Failed"       : sum(
            1 for d in report["details"]
            if d["Status"] == "[FAIL] FAIL"
        ),
        "Not Found"    : sum(
            1 for d in report["details"]
            if "NOT FOUND" in d["Status"]
        ),
    }
    df = pd.DataFrame([row])
    df.to_csv(
        "audit_log.csv",
        mode   = "a",
        header = not pd.io.common.file_exists("audit_log.csv"),
        index  = False
    )

if __name__ == "__main__":
    from pathlib import Path

    md_files = sorted(Path("outputs").rglob("*.md"))
    md_files = [f for f in md_files if not f.stem.endswith("_0")]
    print(f"📂 Found {len(md_files)} MTCs to check\n")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        report = verify_mtc(text, md_file.stem)

        print("=" * 55)
        print(f"📄 MTC      : {report['mtc_name']}")
        print(f"📋 Grade    : {report['grade']}")
        print(f"🏭 Family   : {report['family']}")
        print(f"⚖️  Verdict  : {report['verdict']}")
        print("-" * 55)
        df = pd.DataFrame(report["details"])
        print(df.to_string(index=False))
        print()