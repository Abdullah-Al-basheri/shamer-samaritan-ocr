import base64
import os

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# الإعدادات
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get(
    "SHAMER_MODEL", os.path.join("models", "samerocr_yolov8n_p2.onnx")
)
FONT_PATH = os.environ.get(
    "SHAMER_FONT", os.path.join("fonts", "NotoSansSamaritan-Regular.ttf")
)
IMG_SIZE = 640  # احتياطي فقط — القيمة الفعلية تُقرأ تلقائيًا من ملف النموذج
NMS_IOU = 0.5

# --------------------------------------------------------------------------- #
# كل إعدادات العرض في مكان واحد — عدّل من هنا فقط
# --------------------------------------------------------------------------- #
CONFIG = {
    # ---------- الصفحة ----------
    "page_padding": 10,  # الحشوة حول الصفحة (بالبكسل)
    "hide_footer": True,  # إخفاء الشريط السفلي لـ Gradio (True/False)
    # ---------- الترويسة ----------
    "header_text": "Shamer",  # نص العنوان
    "header_font_size": 30,  # حجم خط العنوان
    "header_letter_spacing": 4,  # التباعد بين حروف العنوان
    "header_bold": True,  # عنوان عريض؟ (True/False)
    # ---------- الصور (إدخال / إخراج) ----------
    "image_height": 430,  # ارتفاع خانتي الصور (بالبكسل)
    # ---------- المربع حول كل حرف ----------
    "box_color": (255, 0, 0),  # لون المربع: (أحمر، أخضر، أزرق)
    "box_thickness": 1,  # سماكة المربع بالبكسل — كبّر الرقم تتخن
    # ---------- دقة الرسم ----------
    "render_scale": 2.0,  # دقة الإخراج = دقة الصورة × هذا الرقم
    # (2 = ضعف الدقة: أوضح عند التقريب، وسقف أمان 3200px)
    # ---------- التسمية فوق المربع (الحرف + النسبة) ----------
    "label_show_default": True,  # إظهار التسميات افتراضيًا؟ (True/False)
    "label_font_size": 22,  # الحد الأقصى لمقاس خط الحرف — المقاس الفعلي
    # يُصغَّر تلقائيًا حتى لا تتجاوز التسمية عرض مربعها
    "label_width_ratio": 1.0,  # عرض التسمية = هذه النسبة × عرض المربع (1 = بقد المربع)
    "label_thickness": 0,  # سماكة خط التسمية فقط (0 = أنحف، 1 = مسمّن قليلًا)
    "label_opacity": 1.0,  # وضوح التسمية: 1 = صلب، 0.5 = شفاف
    # ---------- عناصر التحكم ----------
    "default_conf": 0.25,  # أقل نسبة ثقة الافتراضية
    # ---------- خانة النص السامري ----------
    "text_font_size": 30,  # حجم خط النص السامري
    "text_lines": 3,  # عدد أسطر خانة النص
}


def _sanitize_config():
    """يجعل الكود ديناميكيًا: أي قيمة تغيّرها في CONFIG تُضبط تلقائيًا
    ضمن حدود آمنة، فلا يظهر أي خطأ مهما غيّرت."""
    c = CONFIG

    def _num(key, default, lo, hi):
        try:
            v = float(c.get(key, default))
        except (TypeError, ValueError):
            v = float(default)
        c[key] = min(hi, max(lo, v))

    def _int(key, default, lo, hi):
        _num(key, default, lo, hi)
        c[key] = int(round(c[key]))

    _int("page_padding", 10, 0, 60)
    c["header_text"] = str(c.get("header_text", "Shamer"))
    _int("header_font_size", 30, 8, 80)
    _int("header_letter_spacing", 4, 0, 30)
    c["header_bold"] = bool(c.get("header_bold", True))
    _int("image_height", 430, 200, 1200)

    col = c.get("box_color", (0, 214, 110))
    try:
        col = tuple(int(min(255, max(0, v))) for v in list(col)[:3])
    except (TypeError, ValueError):
        col = (0, 214, 110)
    c["box_color"] = col if len(col) == 3 else (0, 214, 110)

    _int("box_thickness", 1, 0, 20)
    _num("render_scale", 2.0, 0.5, 4.0)
    c["label_show_default"] = bool(c.get("label_show_default", True))
    _int("label_font_size", 22, 8, 120)
    _num("label_width_ratio", 1.0, 0.2, 3.0)
    _int("label_thickness", 0, 0, 5)
    _num("label_opacity", 1.0, 0.0, 1.0)
    _num("default_conf", 0.25, 0.05, 0.9)
    _int("text_font_size", 30, 8, 80)
    _int("text_lines", 3, 1, 20)
    c["hide_footer"] = bool(c.get("hide_footer", True))


_sanitize_config()

# الترتيب ثابت ويجب أن يطابق تسميات التدريب.
CLASS_NAMES = [
    "ALAF",
    "BIT",
    "GAMAN",
    "DALAT",
    "IY",
    "BAA",
    "ZEN",
    "IT",
    "TIT",
    "YUT",
    "KAAF",
    "LABAT",
    "MIM",
    "NUN",
    "SINGAAT",
    "IN",
    "FI",
    "TSAADIY",
    "QUF",
    "RISH",
    "SHAN",
    "TAAF",
    "DOT",
]
DOT_ID = CLASS_NAMES.index("DOT")


def class_to_char(class_id: int) -> str:
    """كتلة السامرية في يونيكود تبدأ من U+0800، والنقطة فاصل كلمات (فراغ)."""
    if class_id == DOT_ID:
        return " "
    return chr(0x0800 + class_id)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "ملف النموذج غير موجود: " + MODEL_PATH + "\n"
        "ضع الملف samerocr_yolov8n_p2.onnx داخل مجلد 'models/' "
        "أو حدّد المتغير SHAMER_MODEL."
    )
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(
        "ملف الخط غير موجود: " + FONT_PATH + "\n"
        "ضع الملف NotoSansSamaritan-Regular.ttf داخل مجلد 'fonts/' "
        "أو حدّد المتغير SHAMER_FONT."
    )

# تضمين الخط داخل CSS (base64) ليشتغل في المتصفح على أي نظام بدون مسارات.
with open(FONT_PATH, "rb") as _f:
    FONT_B64 = base64.b64encode(_f.read()).decode("ascii")

# --------------------------------------------------------------------------- #
# النموذج
# --------------------------------------------------------------------------- #
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# مقاس دخل النموذج يُقرأ تلقائيًا من الملف نفسه — ديناميكي بالكامل:
# أيًا كان المقاس الذي صُدّر به النموذج (640 أو 1080 أو غيره)، الكود يتكيف وحده.
_in_shape = session.get_inputs()[0].shape
_model_hw = tuple(_in_shape[2:4])
if (
    all(isinstance(v, int) for v in _model_hw)
    and _model_hw[0] == _model_hw[1]
    and _model_hw[0] > 0
):
    IMG_SIZE = _model_hw[0]
    print(f"[شامر] مقاس دخل النموذج: {IMG_SIZE}×{IMG_SIZE} (مقروء تلقائيًا من الملف)")
else:
    print(f"[شامر] النموذج بمقاس مرن — سيُستخدم IMG_SIZE = {IMG_SIZE}")


# --------------------------------------------------------------------------- #
# المعالجة قبل/بعد الاكتشاف
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, size: int = IMG_SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, left, top


def detect(img_rgb: np.ndarray, min_conf: float):
    """يرجع قائمة الاكتشافات بإحداثيات الصورة الأصلية."""
    H, W = img_rgb.shape[:2]
    canvas, r, left, top = letterbox(img_rgb)

    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    out = session.run(None, {INPUT_NAME: x})[0]  # (1, 4 + classes, N)
    preds = out[0].T  # (N, 4 + classes)

    scores = preds[:, 4:]
    cls_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    keep = confs >= min_conf
    if not keep.any():
        return []

    preds, cls_ids, confs = preds[keep], cls_ids[keep], confs[keep]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

    x1 = np.clip((cx - w / 2 - left) / r, 0, W)
    y1 = np.clip((cy - h / 2 - top) / r, 0, H)
    x2 = np.clip((cx + w / 2 - left) / r, 0, W)
    y2 = np.clip((cy + h / 2 - top) / r, 0, H)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    idx = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), confs.tolist(), float(min_conf), NMS_IOU
    )
    idx = np.array(idx).flatten().astype(int)

    dets = []
    for i in idx:
        dets.append(
            {
                "cls": int(cls_ids[i]),
                "conf": float(confs[i]),
                "x1": float(x1[i]),
                "y1": float(y1[i]),
                "x2": float(x2[i]),
                "y2": float(y2[i]),
                "cx": float((x1[i] + x2[i]) / 2),
                "cy": float((y1[i] + y2[i]) / 2),
                "h": float(y2[i] - y1[i]),
            }
        )
    return dets


def group_lines(dets):
    """يجمّع الاكتشافات في أسطر (من الأعلى للأسفل)، وكل سطر من اليمين لليسار."""
    if not dets:
        return []
    letters = [d for d in dets if d["cls"] != DOT_ID] or dets
    ref_h = float(np.median([d["h"] for d in letters]))

    lines = []
    for d in sorted(dets, key=lambda d: d["cy"]):
        if lines and abs(d["cy"] - np.mean([e["cy"] for e in lines[-1]])) < 0.6 * ref_h:
            lines[-1].append(d)
        else:
            lines.append([d])

    return [sorted(line, key=lambda d: d["cx"], reverse=True) for line in lines]


def draw(img_rgb: np.ndarray, dets, show_labels: bool) -> np.ndarray:
    """يرسم المربعات على الصورة نفسها + (الحرف + النسبة) بجانب كل مربع.

    دقة احترافية: الرسم يتم على نسخة مكبَّرة من الصورة (render_scale)
    فتبقى المربعات والتسميات حادة وواضحة حتى عند تقريب الصورة.
    ترتيب ثابت: كل تسمية فوق مربعها دائمًا (أو داخله من الأعلى لأول سطر)،
    ومقاسها يُصغَّر تلقائيًا حتى لا تتجاوز عرض مربعها — فلا تراكب ولا بعثرة.
    """
    cfg = CONFIG
    box_rgb = cfg["box_color"]

    # --- تكبير الصورة للدقة الاحترافية (مع سقف أمان) ---
    scale = cfg["render_scale"]
    h0, w0 = img_rgb.shape[:2]
    if h0 * scale > 3200:
        scale = 3200 / h0
    W, H = round(w0 * scale), round(h0 * scale)
    img = Image.fromarray(img_rgb).convert("RGBA").resize((W, H), Image.LANCZOS)
    dr = ImageDraw.Draw(img)

    # --- تكبير كل الإحداثيات بنفس المقياس ---
    dets = [
        {
            **d,
            "x1": d["x1"] * scale,
            "y1": d["y1"] * scale,
            "x2": d["x2"] * scale,
            "y2": d["y2"] * scale,
        }
        for d in dets
    ]

    thick = max(1, round(cfg["box_thickness"] * scale))
    for d in dets:
        dr.rectangle(
            [d["x1"], d["y1"], d["x2"], d["y2"]],
            outline=box_rgb + (255,),
            width=thick,
        )

    if show_labels:
        max_fs = cfg["label_font_size"] * scale
        min_fs = 8 * scale
        width_ratio = cfg["label_width_ratio"]
        pct_ratio = 0.75  # النسبة أصغر قليلًا من الحرف
        alpha = round(255 * cfg["label_opacity"])
        lw = max(0, round(cfg["label_thickness"] * scale))

        REF = 100  # مقاس مرجعي لقياس النص قبل تصغيره
        fref = ImageFont.truetype(FONT_PATH, REF)
        fref_pct = ImageFont.truetype(FONT_PATH, round(REF * pct_ratio))
        fonts, fonts_pct = {}, {}

        for d in dets:
            char = class_to_char(d["cls"])
            pct = f"{d['conf'] * 100:.0f}%"
            cb = dr.textbbox((0, 0), char, font=fref)
            pb = dr.textbbox((0, 0), pct, font=fref_pct)
            ref_w = (cb[2] - cb[0]) + REF * 0.15 + (pb[2] - pb[0])
            box_w = max(1.0, d["x2"] - d["x1"])
            fs = round(min(max_fs, max(min_fs, REF * box_w * width_ratio / ref_w)))
            if fs not in fonts:
                fonts[fs] = ImageFont.truetype(FONT_PATH, fs)
                fonts_pct[fs] = ImageFont.truetype(
                    FONT_PATH, max(round(min_fs), round(fs * pct_ratio))
                )
            font, font_pct = fonts[fs], fonts_pct[fs]

            cb = dr.textbbox((0, 0), char, font=font)
            pb = dr.textbbox((0, 0), pct, font=font_pct)
            cw, chh = cb[2] - cb[0], cb[3] - cb[1]
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            gap = max(1, round(fs * 0.15))

            # موضع ثابت دائمًا: فوق المربع، أو داخله من الأعلى لأول سطر
            tx = d["x1"]
            ty = d["y1"] - chh - 2 * scale
            if ty < 0:
                ty = d["y1"] + 2 * scale
            dr.text(
                (tx, ty),
                char,
                font=font,
                fill=box_rgb + (alpha,),
                stroke_width=lw,
                stroke_fill=box_rgb + (alpha,),
            )
            dr.text(
                (tx + cw + gap, ty + chh - ph),
                pct,
                font=font_pct,
                fill=box_rgb + (alpha,),
                stroke_width=lw,
                stroke_fill=box_rgb + (alpha,),
            )

    return np.asarray(img.convert("RGB"))


# --------------------------------------------------------------------------- #
# معالج Gradio
# --------------------------------------------------------------------------- #
def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def run(image, min_conf, show_labels):
    if image is None:
        return None, "⚠️ ارفع صورة أولًا.", "", ""

    img = to_rgb(image)
    dets = detect(img, float(min_conf))

    if not dets:
        return img, "لم يتم اكتشاف أي حروف. جرّب خفض «أقل نسبة ثقة».", "", ""

    info = f"**عدد الحروف المكتشفة:** {len(dets)}"

    lines = group_lines(dets)
    text_lines, flat = [], []
    for line in lines:
        flat.extend(line)
        text = "".join(class_to_char(d["cls"]) for d in line)
        text_lines.append(" ".join(text.split()))

    rows = "".join(
        "<tr>"
        f"<td>{i}</td>"
        f"<td class='sam-cell'>{class_to_char(d['cls'])}</td>"
        f"<td>{CLASS_NAMES[d['cls']]}</td>"
        f"<td>{d['conf'] * 100:.1f}%</td>"
        "</tr>"
        for i, d in enumerate(flat, 1)
    )
    details = (
        "<table class='det'>"
        "<tr><th>#</th><th>الحرف</th><th>الاسم</th><th>نسبة الثقة</th></tr>"
        f"{rows}</table>"
    )

    return draw(img, dets, show_labels), info, "\n".join(text_lines), details


# --------------------------------------------------------------------------- #
# الواجهة
# --------------------------------------------------------------------------- #
CSS_TEMPLATE = """
@font-face {
  font-family: 'NotoSamaritan';
  src: url(data:font/ttf;base64,__FONT_B64__);
}
.gradio-container {
  max-width: 100% !important;
  margin: 0 !important;
  padding: __PAD__px __PAD__px !important;
}
footer { display: __FOOTER__ !important; }
.shamer-header { padding: 2px 6px; margin-bottom: 6px; }
.shamer-header h1 {
  margin: 0; text-align: center; color: #222;
  font-size: __HFS__px; font-weight: __HB__; letter-spacing: __HLS__px;
}
/* الصورة تبقى داخل مكانها المخصص مهما كان حجمها */
.fit-image img { object-fit: contain !important; }
/* النص السامري بالخط الصحيح */
.sam textarea {
  font-family: 'NotoSamaritan', sans-serif !important;
  font-size: __TFS__px !important; line-height: 1.9 !important;
}
.sam-cell { font-family: 'NotoSamaritan', sans-serif !important; font-size: 28px; }
table.det { width: 100%; border-collapse: collapse; }
table.det th, table.det td {
  border: 1px solid #ddd; padding: 5px 10px; text-align: center;
}
table.det th { background: #f3ede4; }
"""

CSS = (
    CSS_TEMPLATE.replace("__FONT_B64__", FONT_B64)
    .replace("__PAD__", str(CONFIG["page_padding"]))
    .replace("__FOOTER__", "none" if CONFIG["hide_footer"] else "block")
    .replace("__HFS__", str(CONFIG["header_font_size"]))
    .replace("__HB__", "800" if CONFIG["header_bold"] else "400")
    .replace("__HLS__", str(CONFIG["header_letter_spacing"]))
    .replace("__TFS__", str(CONFIG["text_font_size"]))
)

HEADER_HTML = f"<div class='shamer-header'><h1>{CONFIG['header_text']}</h1></div>"

with gr.Blocks(title=CONFIG["header_text"], css=CSS) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(
                type="numpy",
                label="١) صورة الإدخال",
                height=CONFIG["image_height"],
                elem_classes=["fit-image"],
                sources=["upload"],
                show_download_button=False,
                show_fullscreen_button=False,
            )
            conf = gr.Slider(
                0.05, 0.9, value=CONFIG["default_conf"], step=0.05, label="أقل نسبة ثقة"
            )
            labels = gr.Checkbox(
                value=CONFIG["label_show_default"],
                label="إظهار الحرف والنسبة فوق المربعات",
            )
            btn = gr.Button("بدء التعرف", variant="primary")
        with gr.Column():
            out_img = gr.Image(
                label="٢) الصورة مع المربعات",
                height=CONFIG["image_height"],
                elem_classes=["fit-image"],
                show_download_button=False,
                show_fullscreen_button=False,
            )
            info = gr.Markdown()
            out_txt = gr.Textbox(
                label="النص السامري (من اليمين إلى اليسار)",
                lines=CONFIG["text_lines"],
                rtl=True,
                show_copy_button=True,
                elem_classes=["sam"],
            )
            with gr.Accordion("تفاصيل كل حرف", open=False):
                details = gr.HTML()

    btn.click(run, [inp, conf, labels], [out_img, info, out_txt, details])

if __name__ == "__main__":
    demo.launch()
