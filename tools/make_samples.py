"""
Generate realistic sample land-record documents for the demo.

Why this exists: PS 26018 ships with no dataset (has_dataset: false), so the
system needs its own corpus. These are synthetic records modelled on the real
layout of Bhulekh khatauni / khasra extracts and Maharashtra 7/12 extracts.

Every record is a genuine PDF with a real text layer, plus a few deliberately
degraded raster "scans", so the pipeline does actual extraction rather than
replaying a canned result. Edge cases are planted on purpose so the validation
engine and the human-review queue both have something to show.

Run:  python3 tools/make_samples.py
"""

from __future__ import annotations

import os
import random
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "samples")
os.makedirs(OUT, exist_ok=True)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

DEVA_CANDIDATES = [
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
]

DEVA_OK = False
for path in DEVA_CANDIDATES:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont("Deva", path))
        DEVA_OK = True
        break
if not DEVA_OK:
    print("  ! No Devanagari font found - Hindi samples will render as boxes.")

# The Devanagari font carries no Latin glyphs, and Helvetica carries no
# Devanagari glyphs, so bilingual lines have to be drawn run by run or one
# script silently renders as NUL boxes. That is exactly the kind of mixed-script
# text a real Bhulekh extract contains, so it has to be right.
DEVA_RE = re.compile(r"[\u0900-\u097F]")


def _runs(text: str):
    runs, cur, mode = [], "", None
    for ch in text:
        if DEVA_RE.match(ch):
            m = "d"
        elif ch.isascii() and ch.isalnum():
            m = "l"
        else:
            m = mode           # punctuation/space inherits the current script
        if m is None:
            m = "l"
        if mode is None or m == mode:
            cur += ch
            mode = m
        else:
            runs.append((cur, mode))
            cur, mode = ch, m
    if cur:
        runs.append((cur, mode))
    return runs


def _font_for(mode: str, bold: bool) -> str:
    if mode == "d" and DEVA_OK:
        return "Deva"
    return "Helvetica-Bold" if bold else "Helvetica"


def text_width(s: str, size: float, bold: bool = False) -> float:
    return sum(pdfmetrics.stringWidth(r, _font_for(m, bold), size)
               for r, m in _runs(s))


def dtext(c, x: float, y: float, s: str, size: float, bold: bool = False) -> float:
    for r, m in _runs(s):
        f = _font_for(m, bold)
        c.setFont(f, size)
        c.drawString(x, y, r)
        x += pdfmetrics.stringWidth(r, f, size)
    return x


def dcentre(c, xc: float, y: float, s: str, size: float, bold: bool = False) -> None:
    dtext(c, xc - text_width(s, size, bold) / 2.0, y, s, size, bold)

W, H = A4


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def draw_record(path: str, header: list, rows: list, footer: list,
                seed: int = 0, noise: bool = True) -> None:
    """Render one land record as a government-form-style PDF page."""
    rnd = random.Random(seed)
    c = rl_canvas.Canvas(path, pagesize=A4)
    c.setTitle(os.path.basename(path))

    margin = 18 * mm
    y = H - margin

    # Header block
    for i, line in enumerate(header):
        dcentre(c, W / 2, y, line, 13 if i == 0 else (11 if i == 1 else 10), bold=True)
        y -= 6.2 * mm

    y -= 2 * mm
    c.setLineWidth(1.1)
    c.line(margin, y, W - margin, y)
    y -= 9 * mm

    # Field rows: "label : value", the shape almost every RoR extract uses
    label_x = margin + 2 * mm
    value_x = margin + 68 * mm
    for label, value in rows:
        if label is None:                      # spacer / section rule
            y -= 3 * mm
            c.setLineWidth(0.4)
            c.setStrokeGray(0.65)
            c.line(margin, y, W - margin, y)
            c.setStrokeGray(0)
            y -= 6 * mm
            continue

        jitter = (rnd.uniform(-0.35, 0.35) if noise else 0)
        dtext(c, label_x, y + jitter, f"{label}", 10.5)
        dtext(c, value_x - 4 * mm, y + jitter, ":", 10.5)
        dtext(c, value_x, y + jitter, str(value), 10.5, bold=True)
        y -= 7.4 * mm

        if y < margin + 40 * mm:
            c.showPage()
            y = H - margin

    y -= 4 * mm
    c.setLineWidth(0.8)
    c.line(margin, y, W - margin, y)
    y -= 8 * mm

    for line in footer:
        dtext(c, label_x, y, line, 9)
        y -= 5.6 * mm

    # Signature / seal area, like the real scans
    _sig = "(हस्ताक्षर / Signature)"
    dtext(c, W - margin - text_width(_sig, 9), margin + 20 * mm, _sig, 9)
    c.setStrokeGray(0.4)
    c.setLineWidth(0.7)
    c.line(W - margin - 55 * mm, margin + 24 * mm, W - margin, margin + 24 * mm)
    c.circle(margin + 22 * mm, margin + 24 * mm, 13 * mm, stroke=1, fill=0)
    dcentre(c, margin + 22 * mm, margin + 25 * mm, "राजस्व विभाग", 6.5)
    dcentre(c, margin + 22 * mm, margin + 21 * mm, "REVENUE DEPT", 6.5)
    c.setStrokeGray(0)

    c.showPage()
    c.save()
    print(f"  + {os.path.relpath(path, ROOT)}")


UP_HEADER = [
    "उत्तर प्रदेश शासन - राजस्व परिषद",
    "भूलेख / खतौनी नकल (अंश)",
    "Government of Uttar Pradesh - Board of Revenue",
]
MP_HEADER = [
    "मध्य प्रदेश शासन - राजस्व विभाग",
    "खसरा / भू-अधिकार अभिलेख",
]
MH_HEADER = [
    "महाराष्ट्र शासन - महसूल विभाग",
    "गाव नमुना सात/बारा (7/12 Extract)",
]
RJ_HEADER = [
    "राजस्थान सरकार - राजस्व मंडल",
    "जमाबंदी नकल / Record of Rights",
]

FOOTER_STD = [
    "यह अभिलेख कंप्यूटरीकृत प्रति है। मूल अभिलेख तहसील कार्यालय में उपलब्ध है।",
    "Computer-generated copy. Original record retained at the Tehsil office.",
]


# ---------------------------------------------------------------------------
# The corpus. Each entry plants a specific validation scenario.
# ---------------------------------------------------------------------------

SAMPLES = [
    # 1. Clean, fully valid, bilingual -> should sail through to auto-approved
    dict(
        name="sample_01_khatauni_up_clean.pdf", header=UP_HEADER, seed=11,
        rows=[
            ("खाता संख्या / Khata Number", "1428"),
            ("खसरा संख्या / Khasra Number", "237/4"),
            ("ULPIN", "UP0912237004"),
            (None, None),
            ("खातेदार का नाम / Owner Name", "रामप्रसाद वर्मा"),
            ("पिता का नाम / Father's Name", "श्री जगदीश वर्मा"),
            ("अंश / Share", "1/1"),
            (None, None),
            ("क्षेत्रफल / Area", "1.2540 हेक्टेयर"),
            ("भूमि श्रेणी / Land Classification", "सिंचित कृषि भूमि"),
            (None, None),
            ("ग्राम / Village", "नरहरपुर"),
            ("तहसील / Tehsil", "Sadar"),
            ("जनपद / District", "Lucknow"),
            ("राज्य / State", "Uttar Pradesh"),
            (None, None),
            ("नामांतरण संख्या / Mutation Number", "MUT-2023-004182"),
            ("नामांतरण दिनांक / Mutation Date", "14/03/2023"),
            ("पंजीकरण संख्या / Registration Number", "REG-2019-77120"),
            ("पंजीकरण दिनांक / Registration Date", "09/07/2019"),
        ],
        footer=FOOTER_STD,
    ),

    # 2. Hindi-only labels -> proves multilingual label matching works
    dict(
        name="sample_02_khasra_mp_hindi.pdf", header=MP_HEADER, seed=22,
        rows=[
            ("खाता क्रमांक", "९०७३"),
            ("खसरा क्रमांक", "512/1"),
            (None, None),
            ("कृषक का नाम", "सुनीता बाई"),
            ("पति का नाम", "श्री देवीलाल पटेल"),
            ("हिस्सा", "1/2"),
            (None, None),
            ("रकबा", "0.8090 हेक्टेयर"),
            ("भूमि का प्रकार", "असिंचित"),
            (None, None),
            ("ग्राम", "बरखेड़ी"),
            ("तहसील", "Huzur"),
            ("जिला", "Bhopal"),
            ("राज्य", "Madhya Pradesh"),
            (None, None),
            ("नामांतरण क्रमांक", "MP/MUT/2024/8891"),
            ("नामांतरण दिनांक", "२२ जनवरी २०२४"),
        ],
        footer=FOOTER_STD,
    ),

    # 3. Duplicate parcel with a DIFFERENT owner -> DUPLICATE_CONFLICT
    dict(
        name="sample_03_duplicate_conflict.pdf", header=UP_HEADER, seed=33,
        rows=[
            ("खाता संख्या / Khata Number", "1428"),
            ("खसरा संख्या / Khasra Number", "237/4"),
            (None, None),
            ("खातेदार का नाम / Owner Name", "मोहन लाल गुप्ता"),
            ("पिता का नाम / Father's Name", "श्री बनवारी लाल"),
            ("अंश / Share", "1/1"),
            (None, None),
            ("क्षेत्रफल / Area", "1.2540 हेक्टेयर"),
            ("भूमि श्रेणी / Land Classification", "सिंचित"),
            (None, None),
            ("ग्राम / Village", "नरहरपुर"),
            ("तहसील / Tehsil", "Sadar"),
            ("जनपद / District", "Lucknow"),
            ("राज्य / State", "Uttar Pradesh"),
            (None, None),
            ("नामांतरण संख्या / Mutation Number", "MUT-2024-119045"),
            ("नामांतरण दिनांक / Mutation Date", "02/08/2024"),
        ],
        footer=["संदिग्ध प्रविष्टि - सत्यापन आवश्यक।"] + FOOTER_STD,
    ),

    # 4. Regional unit (bigha/biswa) -> AREA_REGIONAL_UNIT, needs conversion
    dict(
        name="sample_04_bigha_biswa_rajasthan.pdf", header=RJ_HEADER, seed=44,
        rows=[
            ("खाता संख्या", "316"),
            ("खसरा संख्या", "88"),
            (None, None),
            ("काश्तकार का नाम", "हरिराम चौधरी"),
            ("पिता का नाम", "श्री भंवर लाल"),
            ("हिस्सा", "2/3"),
            (None, None),
            ("रकबा", "2 बीघा 10 बिस्वा"),
            ("किस्म भूमि", "बारानी"),
            (None, None),
            ("ग्राम", "खेड़ला"),
            ("तहसील", "Sanganer"),
            ("जिला", "Jaipur"),
            ("राज्य", "Rajasthan"),
            (None, None),
            ("जमाबंदी वर्ष", "2022-2023"),
            ("पंजीकरण संख्या", "RJ-REG-2016-3341"),
            ("पंजीकरण दिनांक", "18/11/2016"),
        ],
        footer=FOOTER_STD,
    ),

    # 5. Area with no unit + no land class -> AREA_UNIT_MISSING, CLASS_UNMAPPED
    dict(
        name="sample_05_missing_unit.pdf", header=MP_HEADER, seed=55,
        rows=[
            ("खाता क्रमांक", "2210"),
            ("खसरा क्रमांक", "47"),
            (None, None),
            ("कृषक का नाम", "Anil Kumar Sahu"),
            ("पिता का नाम", "Ram Kishan Sahu"),
            (None, None),
            ("रकबा", "12.5"),
            ("भूमि का प्रकार", "मिश्रित प्रकार"),
            (None, None),
            ("ग्राम", "Pipariya"),
            ("तहसील", "Hoshangabad"),
            ("जिला", "Hoshangabad"),
            ("राज्य", "Madhya Pradesh"),
        ],
        footer=["अपूर्ण प्रविष्टि / Incomplete entry."],
    ),

    # 6. Impossible dates -> DATE_FUTURE + DATE_ORDER
    dict(
        name="sample_06_bad_dates.pdf", header=UP_HEADER, seed=66,
        rows=[
            ("खाता संख्या / Khata Number", "7781"),
            ("खसरा संख्या / Khasra Number", "1024/2"),
            (None, None),
            ("खातेदार का नाम / Owner Name", "Shabana Khatoon"),
            ("पिता का नाम / Father's Name", "Abdul Rashid"),
            ("अंश / Share", "1/4"),
            (None, None),
            ("क्षेत्रफल / Area", "3400 वर्ग मीटर"),
            ("भूमि श्रेणी / Land Classification", "आबादी"),
            (None, None),
            ("ग्राम / Village", "Bhadohi Khurd"),
            ("तहसील / Tehsil", "Varanasi"),
            ("जनपद / District", "Varanasi"),
            ("राज्य / State", "Uttar Pradesh"),
            (None, None),
            ("नामांतरण दिनांक / Mutation Date", "11/06/2029"),
            ("पंजीकरण दिनांक / Registration Date", "27/02/2031"),
        ],
        footer=FOOTER_STD,
    ),

    # 7. District absent from the LGD master -> DISTRICT_UNKNOWN + suggestion
    dict(
        name="sample_07_unknown_district.pdf", header=UP_HEADER, seed=77,
        rows=[
            ("खाता संख्या / Khata Number", "559"),
            ("खसरा संख्या / Khasra Number", "390"),
            (None, None),
            ("खातेदार का नाम / Owner Name", "Devendra Singh"),
            ("पिता का नाम / Father's Name", "Rajendra Singh"),
            ("अंश / Share", "1/1"),
            (None, None),
            ("क्षेत्रफल / Area", "0.4050 हेक्टेयर"),
            ("भूमि श्रेणी / Land Classification", "सिंचित"),
            (None, None),
            ("ग्राम / Village", "Kishanpur"),
            ("तहसील / Tehsil", "Bilaspur"),
            ("जनपद / District", "Kanpurr Nagar"),
            ("राज्य / State", "Uttar Pradesh"),
            (None, None),
            ("पंजीकरण संख्या / Registration Number", "REG-2021-55018"),
            ("पंजीकरण दिनांक / Registration Date", "03/12/2021"),
        ],
        footer=FOOTER_STD,
    ),

    # 8. Owner == father, share above unity -> ownership consistency errors
    dict(
        name="sample_08_ownership_conflict.pdf", header=RJ_HEADER, seed=88,
        rows=[
            ("खाता संख्या", "1902"),
            ("खसरा संख्या", "66/1"),
            (None, None),
            ("काश्तकार का नाम", "Suresh Chand Meena"),
            ("पिता का नाम", "Suresh Chand Meena"),
            ("हिस्सा", "7/5"),
            (None, None),
            ("रकबा", "1.6 हेक्टेयर"),
            ("किस्म भूमि", "चाही"),
            (None, None),
            ("ग्राम", "Dausa Kalan"),
            ("तहसील", "Dausa"),
            ("जिला", "Dausa"),
            ("राज्य", "Rajasthan"),
            (None, None),
            ("नामांतरण संख्या", "RJ/MUT/2020/771"),
            ("नामांतरण दिनांक", "30/09/2020"),
        ],
        footer=FOOTER_STD,
    ),

    # 9. Maharashtra 7/12 -> survey number instead of khasra, state mismatch
    dict(
        name="sample_09_maharashtra_712.pdf", header=MH_HEADER, seed=99,
        rows=[
            ("खाते क्रमांक / Khata Number", "4471"),
            ("सर्वे क्रमांक / Survey Number", "142/2B"),
            ("खसरा क्रमांक", "142"),
            (None, None),
            ("भोगवटादार / Owner Name", "Vitthal Bhau Pawar"),
            ("वडिलांचे नाव / Father's Name", "Bhau Sakharam Pawar"),
            ("हिस्सा / Share", "1/3"),
            (None, None),
            ("क्षेत्र / Area", "0.6070 hectare"),
            ("जमिनीचा प्रकार / Land Classification", "जिरायत"),
            (None, None),
            ("गाव / Village", "Shirur Kasar"),
            ("तालुका / Tehsil", "Haveli"),
            ("जिल्हा / District", "Pune"),
            ("राज्य / State", "Maharashtra"),
            (None, None),
            ("फेरफार क्रमांक / Mutation Number", "MH-FER-2022-20641"),
            ("फेरफार दिनांक / Mutation Date", "07/05/2022"),
        ],
        footer=["संगणकीकृत नक्कल / Computerised extract."],
    ),

    # 10. Historical record, pre-independence date -> DATE_TOO_OLD (info)
    dict(
        name="sample_10_historical_1938.pdf", header=UP_HEADER, seed=101,
        rows=[
            ("खाता संख्या / Khata Number", "12"),
            ("खसरा संख्या / Khasra Number", "5"),
            (None, None),
            ("खातेदार का नाम / Owner Name", "Ram Dayal"),
            ("पिता का नाम / Father's Name", "Bhola Nath"),
            ("अंश / Share", "1/1"),
            (None, None),
            ("क्षेत्रफल / Area", "4 acre"),
            ("भूमि श्रेणी / Land Classification", "बंजर"),
            (None, None),
            ("ग्राम / Village", "Rampur Bhagan"),
            ("तहसील / Tehsil", "Sadar"),
            ("जनपद / District", "Prayagraj"),
            ("राज्य / State", "Uttar Pradesh"),
            (None, None),
            ("पंजीकरण दिनांक / Registration Date", "21/04/1938"),
        ],
        footer=["ऐतिहासिक अभिलेख / Historical record - manual verification advised."],
    ),
]


# ---------------------------------------------------------------------------
# Degraded raster "scans" - these exercise the image path and, when Tesseract
# is absent, the honest degraded-mode fallback.
# ---------------------------------------------------------------------------

def make_scans(pdf_names: list) -> None:
    try:
        import pymupdf
    except Exception:
        try:
            import fitz as pymupdf  # noqa: N813
        except Exception:
            print("  ! pymupdf unavailable - skipping raster scans.")
            return
    try:
        import cv2
        import numpy as np
    except Exception:
        print("  ! OpenCV unavailable - skipping raster scans.")
        return

    for idx, (src_name, style) in enumerate(pdf_names):
        src = os.path.join(OUT, src_name)
        if not os.path.exists(src):
            continue
        doc = pymupdf.open(src)
        pix = doc[0].get_pixmap(dpi=170)
        raw = os.path.join(OUT, "_tmp_render.png")
        pix.save(raw)
        doc.close()

        img = cv2.imread(raw, cv2.IMREAD_GRAYSCALE)
        rnd = np.random.default_rng(1000 + idx)

        if style == "skew_blur":
            h, w = img.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), 2.4, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderValue=245)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = np.clip(img.astype(np.int16) + rnd.normal(0, 9, img.shape), 0, 255)
        elif style == "faded":
            img = np.clip(img.astype(np.int16) * 0.55 + 112, 0, 255)
            img = np.clip(img + rnd.normal(0, 13, img.shape), 0, 255)
        else:  # heavy - a genuinely bad scan
            h, w = img.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), -3.6, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderValue=240)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = np.clip(img.astype(np.int16) * 0.62 + 95, 0, 255)
            img = np.clip(img + rnd.normal(0, 20, img.shape), 0, 255)

        img = img.astype(np.uint8)
        base = os.path.splitext(src_name)[0].replace("sample_", "")
        out = os.path.join(OUT, f"scan_{base}_{style}.png")
        cv2.imwrite(out, img)
        print(f"  + {os.path.relpath(out, ROOT)}")

    tmp = os.path.join(OUT, "_tmp_render.png")
    if os.path.exists(tmp):
        os.remove(tmp)


def main() -> int:
    print(f"Writing samples to {os.path.relpath(OUT, ROOT)}/")
    for spec in SAMPLES:
        draw_record(
            os.path.join(OUT, spec["name"]),
            spec["header"], spec["rows"], spec["footer"], seed=spec["seed"],
        )
    make_scans([
        ("sample_01_khatauni_up_clean.pdf", "skew_blur"),
        ("sample_02_khasra_mp_hindi.pdf", "faded"),
        ("sample_09_maharashtra_712.pdf", "heavy"),
    ])
    count = len([f for f in os.listdir(OUT) if not f.startswith("_")])
    print(f"Done. {count} sample documents ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
