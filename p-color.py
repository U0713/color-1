import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image, ImageColor, ImageDraw
from io import BytesIO
import base64

st.set_page_config(page_title="의상 실시간 컬러 체인저 & A4 출력", layout="wide")

# --- 색상 변환 헬퍼 함수 ---
def hsl_to_rgb(h, s, l):
    h_std = (h / 240.0) * 360.0
    s_std = (s / 240.0) * 100.0
    l_std = (l / 240.0) * 100.0
    try:
        return ImageColor.getrgb(f"hsl({h_std:.1f}, {s_std:.1f}%, {l_std:.1f}%)")
    except ValueError:
        return (255, 255, 255)

def rgb_to_hsl(r, g, b):
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r_n, g_n, b_n), min(r_n, g_n, b_n)
    l = (mx + mn) / 2.0
    if mx == mn:
        s = h = 0.0
    else:
        diff = mx - mn
        s = diff / (2.0 - mx - mn) if l > 0.5 else diff / (mx + mn)
        if mx == r_n:
            h = (g_n - b_n) / diff + (6.0 if g_n < b_n else 0.0)
        elif mx == g_n:
            h = (b_n - r_n) / diff + 2.0
        else:
            h = (r_n - g_n) / diff + 4.0
        h /= 6.0
    return int(h * 240), int(s * 240), int(l * 240)

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)

def rgb_to_hex(r, g, b):
    return f"{r:02X}{g:02X}{b:02X}"

# --- 1-A. 노란색 목폴라(polat_sweater.png) 마스크 추출 ---
@st.cache_data
def get_yellow_sweater_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([16, 115, 65])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 턱 위쪽 및 목 윗부분 확실히 제외
    mask[0:int(h * 0.112), :] = 0

    # 목폴라 윗단 안쪽 목 피부 보호 영역
    neck_protect = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        neck_protect,
        center=(w // 2, int(h * 0.115)),
        axes=(int(w * 0.185), int(h * 0.072)),
        angle=0, startAngle=0, endAngle=180,
        color=255, thickness=-1
    )
    mask[neck_protect == 255] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 하단부(허리~양쪽 소매 바깥 경계까지) 빈틈 없이 모두 채우기
    lower_start = int(h * 0.52)
    for y in range(lower_start, h):
        cols = np.where(mask[y, :] > 0)[0]
        if len(cols) > 0:
            left_edge = 0 if cols[0] < int(w * 0.08) else cols[0]
            right_edge = w - 1 if cols[-1] > int(w * 0.92) else cols[-1]
            mask[y, left_edge:right_edge + 1] = 255

    mask[int(h * 0.95):h, :] = 255
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask, img

# --- 1-B. 흰색 U넥 티셔츠(polat_sweater01.png) 의상 영역 마스크 추출 ---
@st.cache_data
def get_white_tshirt_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    is_white_fabric = (s_ch < 38) & (v_ch > 165)
    raw_mask = np.where(is_white_fabric, 255, 0).astype(np.uint8)
    raw_mask[0:int(h * 0.255), :] = 0

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[0:int(h * 0.27), :] = cv2.GC_BGD
    gc_mask[int(h * 0.27):int(h * 0.45), 0:int(w * 0.12)] = cv2.GC_PR_BGD
    gc_mask[int(h * 0.27):int(h * 0.45), int(w * 0.88):w] = cv2.GC_PR_BGD

    gc_mask[raw_mask == 255] = cv2.GC_PR_FGD
    gc_mask[int(h * 0.62):h, int(w * 0.10):int(w * 0.90)] = cv2.GC_FGD
    gc_mask[int(h * 0.32):h, 0:int(w * 0.14)] = cv2.GC_FGD
    gc_mask[int(h * 0.32):h, int(w * 0.86):w] = cv2.GC_FGD
    gc_mask[s_ch >= 38] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    scale = 0.5
    small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    small_gc = cv2.resize(gc_mask, (small_img.shape[1], small_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.grabCut(small_img, small_gc, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)

    shirt_mask = np.where((small_gc == cv2.GC_FGD) | (small_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    shirt_mask = cv2.resize(shirt_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    shirt_mask[s_ch >= 40] = 0
    shirt_mask[0:int(h * 0.255), :] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel_close)
    shirt_mask = cv2.GaussianBlur(shirt_mask, (5, 5), 0)

    return shirt_mask, img

# --- 2. 추출된 의상 영역에 새 색상 적용 (음영 무시/적용 선택 지원) ---
def apply_clothing_color(img, mask, target_rgb, base_norm=235.0, smooth_lower_gaps=False, use_shading=False):
    h, w, _ = img.shape
    img_float = img.astype(np.float32)
    b_t, g_t, r_t = target_rgb[2], target_rgb[1], target_rgb[0]

    colored_layer = np.zeros_like(img_float)

    if use_shading:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        shading = np.clip(v_channel / base_norm, 0.20, 1.05)

        if smooth_lower_gaps:
            lower_y = int(h * 0.55)
            lower_shading = shading[lower_y:, :].copy()
            deep_shadow = lower_shading < 0.80
            lower_shading[deep_shadow] = np.clip(lower_shading[deep_shadow] * 0.35 + 0.62, 0.84, 0.94)
            lower_shading = cv2.GaussianBlur(lower_shading, (9, 9), 0)
            shading[lower_y:, :] = lower_shading

        colored_layer[:, :, 0] = b_t * shading
        colored_layer[:, :, 1] = g_t * shading
        colored_layer[:, :, 2] = r_t * shading
    else:
        # 음영을 완전히 무시하고 선택한 순수 단색(Flat Color) 그대로 채움
        colored_layer[:, :, 0] = b_t
        colored_layer[:, :, 1] = g_t
        colored_layer[:, :, 2] = r_t

    colored_layer = np.clip(colored_layer, 0, 255)

    mask_norm = (mask / 255.0)[:, :, np.newaxis]
    final_img = img_float * (1.0 - mask_norm) + colored_layer * mask_norm
    return cv2.cvtColor(final_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

# --- 3. A4 용지(210mm x 297mm, 300 DPI) 규격 이미지 생성 ---
def create_a4_sheet(images_list, labels_list, hex_code, rgb_val, hsl_val, fit_mode="여백 포함 맞춤 (추천)", include_info=True):
    A4_W, A4_H = 2480, 3508
    a4_canvas = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
    draw = ImageDraw.Draw(a4_canvas)

    num_imgs = len(images_list)

    if num_imgs == 1:
        src_pil = Image.fromarray(images_list[0])
        src_w, src_h = src_pil.size

        if fit_mode == "여백 포함 맞춤 (추천)":
            margin_x = 180
            margin_top = 180
            margin_bottom = 320 if include_info else 180
            avail_w = A4_W - margin_x * 2
            avail_h = A4_H - margin_top - margin_bottom

            scale = min(avail_w / src_w, avail_h / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = src_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            paste_x = (A4_W - new_w) // 2
            paste_y = margin_top + (avail_h - new_h) // 2
            a4_canvas.paste(resized, (paste_x, paste_y))

            if include_info:
                box_x = paste_x
                box_y = paste_y + new_h + 50
                draw.rectangle([box_x, box_y, box_x + 110, box_y + 110], fill=rgb_val, outline=(80, 80, 80), width=3)
                info_text_1 = f"ITEM : {labels_list[0]}   |   COLOR : #{hex_code}   |   RGB : ({rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]})"
                info_text_2 = f"HSL (240) : ({hsl_val[0]}, {hsl_val[1]}, {hsl_val[2]})   |   PAPER : A4 (210 x 297 mm, 300 DPI)"
                draw.text((box_x + 140, box_y + 18), info_text_1, fill=(30, 30, 30), font_size=34)
                draw.text((box_x + 140, box_y + 68), info_text_2, fill=(90, 90, 90), font_size=28)
        else:
            scale = max(A4_W / src_w, A4_H / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = src_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            crop_x = (new_w - A4_W) // 2
            crop_y = (new_h - A4_H) // 2
            a4_canvas = resized.crop((crop_x, crop_y, crop_x + A4_W, crop_y + A4_H))

    else:
        margin_x = 140
        gap_x = 80
        margin_top = 340
        margin_bottom = 450 if include_info else 300

        slot_w = (A4_W - margin_x * 2 - gap_x) // 2
        slot_h = A4_H - margin_top - margin_bottom

        for idx, (img_np, lbl) in enumerate(zip(images_list, labels_list)):
            src_pil = Image.fromarray(img_np)
            src_w, src_h = src_pil.size
            scale = min(slot_w / src_w, slot_h / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = src_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            slot_x = margin_x + idx * (slot_w + gap_x)
            paste_x = slot_x + (slot_w - new_w) // 2
            paste_y = margin_top + (slot_h - new_h) // 2
            a4_canvas.paste(resized, (paste_x, paste_y))

            draw.text((paste_x, paste_y - 60), lbl, fill=(40, 40, 40), font_size=36)

        if include_info:
            box_x = margin_x
            box_y = A4_H - margin_bottom + 70
            draw.rectangle([box_x, box_y, box_x + 120, box_y + 120], fill=rgb_val, outline=(80, 80, 80), width=3)
            info_text_1 = f"COLOR CODE : #{hex_code}   |   RGB : ({rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]})"
            info_text_2 = f"HSL (240)  : ({hsl_val[0]}, {hsl_val[1]}, {hsl_val[2]})   |   PAPER : A4 (210 x 297 mm, 300 DPI)"
            draw.text((box_x + 150, box_y + 20), info_text_1, fill=(30, 30, 30), font_size=38)
            draw.text((box_x + 150, box_y + 74), info_text_2, fill=(90, 90, 90), font_size=30)

    return a4_canvas

def create_a4_pdf_bytes(a4_pil_image):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader

        pdf_buf = BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=A4)
        page_w, page_h = A4

        img_buf = BytesIO()
        a4_pil_image.save(img_buf, format="JPEG", quality=95)
        img_buf.seek(0)

        c.drawImage(ImageReader(img_buf), 0, 0, width=page_w, height=page_h)
        c.showPage()
        c.save()
        return pdf_buf.getvalue()
    except ImportError:
        pdf_buf = BytesIO()
        a4_pil_image.save(pdf_buf, format="PDF", resolution=300.0)
        return pdf_buf.getvalue()

# --- 세션 상태 관리 ---
if "prev_rgb" not in st.session_state:
    st.session_state.prev_rgb = (241, 162, 8)
if "curr_rgb" not in st.session_state:
    st.session_state.curr_rgb = (30, 43, 60)
if "curr_hsl" not in st.session_state:
    st.session_state.curr_hsl = rgb_to_hsl(30, 43, 60)
if "curr_hex" not in st.session_state:
    st.session_state.curr_hex = "1E2B3C"

def set_color_rgb(r, g, b):
    st.session_state.prev_rgb = st.session_state.curr_rgb
    st.session_state.curr_rgb = (r, g, b)
    st.session_state.curr_hsl = rgb_to_hsl(r, g, b)
    st.session_state.curr_hex = rgb_to_hex(r, g, b)

def on_standard_click(hex_code):
    r, g, b = hex_to_rgb(hex_code)
    set_color_rgb(r, g, b)

def on_picker_change():
    r, g, b = hex_to_rgb(st.session_state.spectrum_picker)
    set_color_rgb(r, g, b)

def on_lum_change():
    h, s, _ = st.session_state.curr_hsl
    l = st.session_state.lum_slider
    r, g, b = hsl_to_rgb(h, s, l)
    st.session_state.prev_rgb = st.session_state.curr_rgb
    st.session_state.curr_rgb = (r, g, b)
    st.session_state.curr_hsl = (h, s, l)
    st.session_state.curr_hex = rgb_to_hex(r, g, b)

def on_rgb_change():
    set_color_rgb(st.session_state.r_in, st.session_state.g_in, st.session_state.b_in)

def on_hsl_change():
    r, g, b = hsl_to_rgb(st.session_state.h_in, st.session_state.s_in, st.session_state.l_in)
    st.session_state.prev_rgb = st.session_state.curr_rgb
    st.session_state.curr_rgb = (r, g, b)
    st.session_state.curr_hsl = (st.session_state.h_in, st.session_state.s_in, st.session_state.l_in)
    st.session_state.curr_hex = rgb_to_hex(r, g, b)

def on_hex_change():
    clean = st.session_state.hex_in.strip().lstrip('#')
    if len(clean) == 6:
        try:
            r, g, b = hex_to_rgb(clean)
            set_color_rgb(r, g, b)
        except ValueError:
            pass

# --- CSS 스타일: 모바일 세로 화면에서도 가로 벌집 팔레트 강제 유지 & 테두리 제거 ---
st.markdown("""
<style>
/* 모바일 세로 화면에서도 사이드바 팔레트 컬럼이 세로로 줄바꿈되지 않도록 강제 가로 배치 */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 2px !important;
    margin-bottom: -10px !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    min-width: 0px !important;
    flex-shrink: 1 !important;
    padding: 0px !important;
}

/* 사이드바 팔레트 버튼 테두리/배경 완전 제거 (모바일/데스크톱 공통) */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    background-color: transparent !important;
    padding: 0px !important;
    min-height: 20px !important;
    height: 20px !important;
    width: 100% !important;
    min-width: 0px !important;
    margin: 0px !important;
    z-index: 2;
    position: relative;
}

section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button:hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button:focus,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button:active {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* 육각형 컬러 타일을 투명 버튼과 1:1 밀착 */
.hex-swatch {
    width: 100%;
    height: 20px;
    clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
    margin-top: -36px;
    pointer-events: none;
    position: relative;
    z-index: 1;
}

.spectrum-bar {
    width: 100%;
    height: 24px;
    border-radius: 4px;
    border: 1px solid #888;
    background: linear-gradient(to right, #ff0000 0%, #ffff00 17%, #00ff00 33%, #00ffff 50%, #0000ff 67%, #ff00ff 83%, #ff0000 100%);
    margin-bottom: 6px;
}
.lum-bar {
    width: 100%;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #888;
    background: linear-gradient(to right, #000000 0%, #808080 50%, #ffffff 100%);
    margin-bottom: -8px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# [왼쪽 사이드바] 색상 조절 UI
# =====================================================================
with st.sidebar:
    st.subheader("색")
    
    selection_mode = st.selectbox(
        "색상 선택 방식:",
        ["표준 (육각형 팔레트)", "사용자 지정 (스펙트럼 & 수치 입력)"],
        index=0
    )

    # 음영 무시/적용 토글 (기본값 False = 음영 무시하고 순수 단색으로 변경)
    use_shading = st.checkbox("옷 주름·음영 살리기 (체크 해제 시 단색 적용)", value=False)
    st.markdown("---")

    if selection_mode == "표준 (육각형 팔레트)":
        st.caption("색(C):")
        hex_honeycomb = [
            ["#003366", "#003399", "#000099", "#0000CC", "#000066", "#330066", "#4B0082"],
            ["#006666", "#006699", "#0066CC", "#0066FF", "#3333FF", "#6600CC", "#660099", "#800080"],
            ["#008080", "#0099CC", "#0099FF", "#3399FF", "#6666FF", "#9933FF", "#9900CC", "#990099", "#660033"],
            ["#009966", "#00CC99", "#00CCFF", "#66CCFF", "#9999FF", "#CC66FF", "#CC00FF", "#CC0099", "#990066", "#800040"],
            ["#006600", "#00CC00", "#00FF99", "#99FFFF", "#CCCCFF", "#FFFFFF", "#FFCCFF", "#FF66FF", "#FF00FF", "#CC0066", "#660022"],
            ["#003300", "#009900", "#66FF66", "#CCFFCC", "#FFFFCC", "#FFCCCC", "#FF99CC", "#FF33CC", "#CC0052", "#800020"],
            ["#333300", "#669900", "#99FF33", "#FFFF99", "#FFCC99", "#FF6699", "#FF0066", "#990033", "#4D0013"],
            ["#666600", "#99CC00", "#CCFF00", "#FFFF00", "#FF9966", "#FF3366", "#CC0033", "#800000"],
            ["#663300", "#996600", "#FFCC00", "#FF9900", "#FF6600", "#FF0000", "#990000"]
        ]

        for r_idx, row_colors in enumerate(hex_honeycomb):
            pad = (11 - len(row_colors)) * 0.5
            col_weights = ([pad] if pad > 0 else []) + [1.0] * len(row_colors) + ([pad] if pad > 0 else [])
            cols = st.columns(col_weights)
            start_col = 1 if pad > 0 else 0

            for c_idx, hex_c in enumerate(row_colors):
                with cols[start_col + c_idx]:
                    st.button(" ", key=f"hex_btn_{r_idx}_{c_idx}", help=hex_c,
                              on_click=on_standard_click, args=(hex_c,), use_container_width=True)
                    st.markdown(f'<div class="hex-swatch" style="background-color:{hex_c};"></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        gray_row = ["#FFFFFF", "#E6E6E6", "#CCCCCC", "#B3B3B3", "#999999", "#808080", "#666666", "#4D4D4D", "#333333", "#1A1A1A", "#000000"]
        g_cols = st.columns(len(gray_row))
        for g_idx, g_hex in enumerate(gray_row):
            with g_cols[g_idx]:
                st.button(" ", key=f"gray_btn_{g_idx}", help=g_hex,
                          on_click=on_standard_click, args=(g_hex,), use_container_width=True)
                st.markdown(f'<div class="hex-swatch" style="background-color:{g_hex};"></div>', unsafe_allow_html=True)

    else:
        st.caption("색(C):")
        st.markdown('<div class="spectrum-bar"></div>', unsafe_allow_html=True)
        
        c_pick1, c_pick2 = st.columns([1.2, 1])
        with c_pick1:
            st.color_picker(
                "스펙트럼 색상 선택",
                value=f"#{st.session_state.curr_hex}",
                key="spectrum_picker",
                on_change=on_picker_change
            )
        with c_pick2:
            st.caption("명도 조절")
            st.markdown('<div class="lum-bar"></div>', unsafe_allow_html=True)
            st.slider(
                "명도(L)", min_value=0, max_value=240,
                value=st.session_state.curr_hsl[2],
                key="lum_slider", on_change=on_lum_change,
                label_visibility="collapsed"
            )

        color_model = st.selectbox("색 모델(D):", ["RGB", "HSL"])

        if color_model == "RGB":
            st.number_input("빨강(R):", min_value=0, max_value=255, value=st.session_state.curr_rgb[0],
                            step=1, key="r_in", on_change=on_rgb_change)
            st.number_input("녹색(G):", min_value=0, max_value=255, value=st.session_state.curr_rgb[1],
                            step=1, key="g_in", on_change=on_rgb_change)
            st.number_input("파랑(B):", min_value=0, max_value=255, value=st.session_state.curr_rgb[2],
                            step=1, key="b_in", on_change=on_rgb_change)
        else:
            st.number_input("색상(H):", min_value=0, max_value=240, value=st.session_state.curr_hsl[0],
                            step=1, key="h_in", on_change=on_hsl_change)
            st.number_input("채도(S):", min_value=0, max_value=240, value=st.session_state.curr_hsl[1],
                            step=1, key="s_in", on_change=on_hsl_change)
            st.number_input("명도(L):", min_value=0, max_value=240, value=st.session_state.curr_hsl[2],
                            step=1, key="l_in", on_change=on_hsl_change)

        st.text_input("HTML 코드 (# 제외):", value=st.session_state.curr_hex, max_chars=6,
                      key="hex_in", on_change=on_hex_change)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    prev_hex = rgb_to_hex(*st.session_state.prev_rgb)
    curr_hex = st.session_state.curr_hex
    st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; width:120px; margin:0 auto;">
            <span style="font-size:12px; margin-bottom:4px;">새 색</span>
            <div style="width:100%; border:1px solid #888;">
                <div style="width:100%; height:34px; background-color:#{curr_hex};"></div>
                <div style="width:100%; height:34px; background-color:#{prev_hex};"></div>
            </div>
            <span style="font-size:12px; margin-top:4px;">현재 색</span>
        </div>
    """, unsafe_allow_html=True)

# =====================================================================
# [오른쪽 메인 영역] 목폴라 & U넥 티셔츠 동시 색상 변경 및 A4 출력
# =====================================================================
st.title("🧶 의상 실시간 컬러 체인저 & A4 출력")

mask_turtle, img_turtle = get_yellow_sweater_mask("polat_sweater.png")
mask_tshirt, img_tshirt = get_white_tshirt_mask("polat_sweater01.png")

if mask_turtle is None or mask_tshirt is None:
    st.error("폴더에 `polat_sweater.png`와 `polat_sweater01.png` 파일이 모두 있는지 확인해 주세요.")
    st.stop()

res_turtle = apply_clothing_color(
    img_turtle, mask_turtle, st.session_state.curr_rgb,
    base_norm=235.0, smooth_lower_gaps=True, use_shading=use_shading
)
res_tshirt = apply_clothing_color(
    img_tshirt, mask_tshirt, st.session_state.curr_rgb,
    base_norm=242.0, smooth_lower_gaps=False, use_shading=use_shading
)

col_preview, col_ctrl = st.columns([1.45, 1])

with col_ctrl:
    st.subheader("🖨️ A4 출력 대상 & 설정")
    
    print_target = st.radio(
        "A4 출력 이미지 선택",
        [
            "목폴라 + U넥 티셔츠 (2종 함께 A4 배치)",
            "목폴라 단독 (polat_sweater.png)",
            "U넥 티셔츠 단독 (polat_sweater01.png)"
        ],
        index=0
    )

    if print_target.startswith("목폴라 + U넥"):
        selected_imgs = [res_turtle, res_tshirt]
        selected_labels = ["TURTLENECK (polat_sweater)", "U-NECK TEE (polat_sweater01)"]
        fit_mode = "여백 포함 맞춤 (추천)"
    elif print_target.startswith("목폴라 단독"):
        selected_imgs = [res_turtle]
        selected_labels = ["TURTLENECK (polat_sweater)"]
        fit_mode = st.radio("A4 용지 배치 방식", ["여백 포함 맞춤 (추천)", "A4 용지 전체 꽉 채우기"], index=0)
    else:
        selected_imgs = [res_tshirt]
        selected_labels = ["U-NECK TEE (polat_sweater01)"]
        fit_mode = st.radio("A4 용지 배치 방식", ["여백 포함 맞춤 (추천)", "A4 용지 전체 꽉 채우기"], index=0)

    include_info = st.checkbox(
        "하단에 색상 코드(HEX/RGB/HSL) 스와치 표기",
        value=True,
        disabled=(len(selected_imgs) == 1 and fit_mode == "A4 용지 전체 꽉 채우기")
    )

    a4_pil = create_a4_sheet(
        selected_imgs,
        selected_labels,
        st.session_state.curr_hex,
        st.session_state.curr_rgb,
        st.session_state.curr_hsl,
        fit_mode=fit_mode,
        include_info=include_info
    )

    st.markdown("---")
    st.subheader("적용된 색상 정보")
    st.write(f"- **HEX 코드:** `#{st.session_state.curr_hex}`")
    st.write(f"- **RGB:** `{st.session_state.curr_rgb[0]}, {st.session_state.curr_rgb[1]}, {st.session_state.curr_rgb[2]}`")
    st.write(f"- **HSL (240 기준):** `{st.session_state.curr_hsl[0]}, {st.session_state.curr_hsl[1]}, {st.session_state.curr_hsl[2]}`")
    st.write("- **출력 규격:** `A4 (210mm × 297mm / 300 DPI)`")

    st.markdown("---")

    pdf_bytes = create_a4_pdf_bytes(a4_pil)
    st.download_button(
        label="📄 A4 인쇄용 PDF 다운로드 (.pdf)",
        data=pdf_bytes,
        file_name=f"garments_A4_{st.session_state.curr_hex}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    png_buf = BytesIO()
    a4_pil.save(png_buf, format="PNG", dpi=(300, 300))
    st.download_button(
        label="🖼️ A4 고해상도 이미지 다운로드 (.png)",
        data=png_buf.getvalue(),
        file_name=f"garments_A4_{st.session_state.curr_hex}.png",
        mime="image/png",
        use_container_width=True
    )

    c_dl1, c_dl2 = st.columns(2)
    buf1 = BytesIO()
    Image.fromarray(res_turtle).save(buf1, format="PNG")
    c_dl1.download_button(
        label="📥 목폴라 개별 PNG",
        data=buf1.getvalue(),
        file_name=f"turtleneck_{st.session_state.curr_hex}.png",
        mime="image/png",
        use_container_width=True
    )

    buf2 = BytesIO()
    Image.fromarray(res_tshirt).save(buf2, format="PNG")
    c_dl2.download_button(
        label="📥 U넥 티셔츠 개별 PNG",
        data=buf2.getvalue(),
        file_name=f"uneck_{st.session_state.curr_hex}.png",
        mime="image/png",
        use_container_width=True
    )

    preview_buf = BytesIO()
    a4_pil.resize((1240, 1754), Image.Resampling.LANCZOS).save(preview_buf, format="JPEG", quality=92)
    b64_img = base64.b64encode(preview_buf.getvalue()).decode()

    print_html = f"""
    <script>
    function printA4Image() {{
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
            <head>
                <title>A4 인쇄 - #{st.session_state.curr_hex}</title>
                <style>
                    @page {{
                        size: A4 portrait;
                        margin: 0;
                    }}
                    body {{
                        margin: 0;
                        padding: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }}
                    img {{
                        width: 210mm;
                        height: 297mm;
                        object-fit: contain;
                    }}
                </style>
            </head>
            <body>
                <img src="data:image/jpeg;base64,{b64_img}" onload="window.print(); window.close();" />
            </body>
            </html>
        `);
        printWindow.document.close();
    }}
    </script>
    <button onclick="printA4Image()" style="
        width: 100%;
        padding: 10px 16px;
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        margin-top: 6px;
    ">🖨️ 프린터로 바로 A4 인쇄하기</button>
    """
    components.html(print_html, height=58)

with col_preview:
    tab_a4, tab_both = st.tabs(["📄 A4 출력 미리보기", "👕 2종 실시간 비교 보기"])

    with tab_a4:
        a4_preview = a4_pil.resize((840, 1188), Image.Resampling.LANCZOS)
        st.image(a4_preview, use_container_width=True, caption=f"A4 출력 레이아웃 (210mm × 297mm) | #{st.session_state.curr_hex}")

    with tab_both:
        c_img1, c_img2 = st.columns(2)
        c_img1.image(res_turtle, use_container_width=True, caption=f"목폴라 (#{st.session_state.curr_hex})")
        c_img2.image(res_tshirt, use_container_width=True, caption=f"U넥 티셔츠 (#{st.session_state.curr_hex})")