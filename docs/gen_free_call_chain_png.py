"""Generate free() call chain diagram as PNG using Pillow."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1760, 2140  # 2x for clarity
SCALE = 2

def s(v):
    return int(v * SCALE)

img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# Try to load a good monospace font
font_paths = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
def load_font(size):
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()

font_title = load_font(s(18))
font_box = load_font(s(13))
font_sub = load_font(s(11))
font_note = load_font(s(11))
font_small = load_font(s(10))
font_decision = load_font(s(12))

# Colors
C_ENGINE = (0, 121, 107)       # teal for engine layer
C_PY = (21, 101, 192)          # blue
C_CPP = (230, 81, 0)           # orange
C_CUDA = (74, 20, 140)         # purple
C_RESERVE = (46, 125, 50)      # green
C_YES = (46, 125, 50)
C_NO = (198, 40, 40)
C_ARROW = (68, 68, 68)
C_DIAMOND_BG = (255, 243, 224)
C_DIAMOND_BORDER = (230, 81, 0)
C_DIAMOND2_BG = (232, 245, 233)
C_DIAMOND2_BORDER = (46, 125, 50)
C_DIAMOND3_BG = (243, 229, 245)
C_DIAMOND3_BORDER = (123, 31, 162)
C_NOPATH_BG = (255, 205, 210)
C_TEXT = (26, 26, 26)
C_TEXTSUB = (224, 224, 224)
C_NOTE = (85, 85, 85)
C_GRAY_LINE = (200, 200, 200)

CX = 440  # center x for the main flow

def rounded_rect(x, y, w, h, color, r=12):
    draw.rounded_rectangle([s(x), s(y), s(x+w), s(y+h)], radius=s(r//2), fill=color)

def text_center(x, y, txt, font, color=(255,255,255)):
    bbox = draw.textbbox((0,0), txt, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((s(x) - tw//2, s(y)), txt, fill=color, font=font)

def arrow_down(x, y1, y2, color=C_ARROW):
    sx, sy1, sy2 = s(x), s(y1), s(y2)
    draw.line([(sx, sy1), (sx, sy2-8)], fill=color, width=s(2))
    draw.polygon([(sx-6, sy2-10), (sx+6, sy2-10), (sx, sy2)], fill=color)

def arrow_right(x1, y, x2, color=C_ARROW):
    sx1, sy, sx2 = s(x1), s(y), s(x2)
    draw.line([(sx1, sy), (sx2-8, sy)], fill=color, width=s(2))
    draw.polygon([(sx2-10, sy-6), (sx2-10, sy+6), (sx2, sy)], fill=color)

def dashed_line(x1, y1, x2, y2, color=C_ARROW, dash=6, gap=4):
    """Draw a dashed line from (x1,y1) to (x2,y2) in logical coords."""
    import math
    sx1, sy1, sx2, sy2 = s(x1), s(y1), s(x2), s(y2)
    dx, dy = sx2 - sx1, sy2 - sy1
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return
    ux, uy = dx/length, dy/length
    pos = 0
    while pos < length:
        end = min(pos + s(dash), length)
        draw.line([(int(sx1+ux*pos), int(sy1+uy*pos)),
                   (int(sx1+ux*end), int(sy1+uy*end))], fill=color, width=s(2))
        pos = end + s(gap)

def diamond(cx, cy, hw, hh, bg, border):
    pts = [(s(cx), s(cy-hh)), (s(cx+hw), s(cy)), (s(cx), s(cy+hh)), (s(cx-hw), s(cy))]
    draw.polygon(pts, fill=bg, outline=border, width=s(2))

def draw_layer_divider(y, label_left, label_right=""):
    """Draw a horizontal dashed divider line with labels."""
    for xx in range(s(20), s(860), s(8)):
        draw.line([(xx, s(y)), (min(xx+s(4), s(860)), s(y))],
                  fill=C_GRAY_LINE, width=s(1))
    draw.text((s(22), s(y+2)), label_left, fill=C_NOTE, font=font_small)
    if label_right:
        bbox = draw.textbbox((0,0), label_right, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text((s(858) - tw, s(y+2)), label_right, fill=C_NOTE, font=font_small)

# ══════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════
text_center(440, 18, "KVCacheManager.free() Complete Call Chain", font_title, C_TEXT)

# ══════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════
rounded_rect(20, 50, 200, 150, (250, 250, 250), r=8)
draw.text((s(30), s(58)), "Legend", fill=C_TEXT, font=font_decision)
rounded_rect(30, 78, 14, 14, C_ENGINE, r=4)
draw.text((s(50), s(78)), "Engine layer (vLLM/SGLang)", fill=C_NOTE, font=font_small)
rounded_rect(30, 98, 14, 14, C_PY, r=4)
draw.text((s(50), s(98)), "kvcached Python layer", fill=C_NOTE, font=font_small)
rounded_rect(30, 118, 14, 14, C_CPP, r=4)
draw.text((s(50), s(118)), "kvcached C++ layer", fill=C_NOTE, font=font_small)
rounded_rect(30, 138, 14, 14, C_CUDA, r=4)
draw.text((s(50), s(138)), "CUDA driver", fill=C_NOTE, font=font_small)
rounded_rect(30, 158, 14, 14, C_RESERVE, r=4)
draw.text((s(50), s(158)), "Fast path (no unmap)", fill=C_NOTE, font=font_small)
rounded_rect(30, 178, 14, 14, C_NOPATH_BG, r=4)
draw.text((s(50), s(178)), "No unmap needed", fill=C_NOTE, font=font_small)

# ══════════════════════════════════════════════════════════════
# Engine Layer: vLLM / SGLang
# ══════════════════════════════════════════════════════════════
# vLLM box (left)
rounded_rect(240, 55, 190, 55, C_ENGINE)
text_center(335, 64, "vLLM BlockPool", font_box)
text_center(335, 82, ".free_blocks(blocks)", font_sub, C_TEXTSUB)
text_center(335, 100, "→ extract block_ids", font_sub, (180, 230, 220))

# SGLang box (right)
rounded_rect(450, 55, 190, 55, C_ENGINE)
text_center(545, 64, "SGLang Allocator", font_box)
text_center(545, 82, ".free(free_index)", font_sub, C_TEXTSUB)
text_center(545, 100, "→ tensor to list", font_sub, (180, 230, 220))

# File annotations
draw.text((s(240), s(112)), "patches.py (vLLM)", fill=C_NOTE, font=font_small)
draw.text((s(490), s(112)), "patches.py (SGLang)", fill=C_NOTE, font=font_small)

# Arrows from both boxes converge to ElasticBlockPool → KVCacheManager
# vLLM arrow
draw.line([(s(335), s(110)), (s(335), s(130)), (s(CX), s(130))], fill=C_ARROW, width=s(2))
# SGLang arrow
draw.line([(s(545), s(110)), (s(545), s(130)), (s(CX), s(130))], fill=C_ARROW, width=s(2))
# merged arrow down
arrow_down(CX, 130, 150)

draw_layer_divider(142, "Engine Integration", "autopatch")

# ══════════════════════════════════════════════════════════════
# kvcached Python Layer
# ══════════════════════════════════════════════════════════════

# ── Box 1: KVCacheManager.free ──
rounded_rect(265, 155, 350, 50, C_PY)
text_center(CX, 167, "KVCacheManager.free(indices)", font_box)
text_center(CX, 186, "kv_cache_manager.py", font_sub, C_TEXTSUB)

arrow_down(CX, 205, 235)

# ── Box 2: Group + free_batch ──
rounded_rect(265, 235, 350, 50, C_PY)
text_center(CX, 247, "Group indices by page_id", font_box)
text_center(CX, 266, "page.free_batch(idxs)", font_sub, C_TEXTSUB)

arrow_down(CX, 285, 320)

# ── Diamond 1: page.empty()? ──
diamond(CX, 360, 110, 45, C_DIAMOND_BG, C_DIAMOND_BORDER)
text_center(CX, 347, "page.empty()?", font_decision, C_TEXT)
text_center(CX, 365, "all blocks freed?", font_small, C_NOTE)

# NO → right
arrow_right(550, 360, 690, C_NO)
draw.text((s(560), s(343)), "NO", fill=C_NO, font=font_decision)
rounded_rect(690, 340, 160, 40, C_NOPATH_BG, r=6)
draw.rectangle([s(690), s(340), s(850), s(380)], outline=C_NO, width=s(1))
text_center(770, 352, "→ avail_pages", font_decision, (183, 28, 28))
draw.text((s(700), s(385)), "page stays mapped", fill=C_NOTE, font=font_small)

# YES → down
arrow_down(CX, 405, 435)
draw.text((s(CX+8), s(407)), "YES", fill=C_YES, font=font_decision)

# ── Box 3: free_pages ──
rounded_rect(265, 435, 350, 50, C_PY)
text_center(CX, 447, "PageAllocator.free_pages(page_ids)", font_box)
text_center(CX, 466, "page_allocator.py", font_sub, C_TEXTSUB)

arrow_down(CX, 485, 520)

# ── Diamond 2: reserved pool full? ──
diamond(CX, 565, 130, 50, C_DIAMOND2_BG, C_DIAMOND2_BORDER)
text_center(CX, 551, "reserved_pool", font_decision, C_TEXT)
text_center(CX, 569, "< max (10)?", font_decision, C_TEXT)

# YES → right (reserve, no unmap)
arrow_right(570, 565, 690, C_YES)
draw.text((s(578), s(548)), "YES", fill=C_YES, font=font_decision)
rounded_rect(690, 535, 160, 55, C_RESERVE, r=6)
text_center(770, 548, "reserved_page_list", font_box)
text_center(770, 567, "NO UNMAP", font_sub, C_TEXTSUB)
draw.text((s(700), s(597)), "keep mapped for fast reuse", fill=C_NOTE, font=font_small)

# NO → down
arrow_down(CX, 615, 650)
draw.text((s(CX+8), s(622)), "NO", fill=C_NO, font=font_decision)

# ── Box 4: _unmap_pages ──
rounded_rect(265, 650, 350, 50, C_PY)
text_center(CX, 662, "PageAllocator._unmap_pages(page_ids)", font_box)
text_center(CX, 681, "page_allocator.py — compute offsets", font_sub, C_TEXTSUB)

arrow_down(CX, 700, 735)

# ── Diamond 3: tp > 1? ──
diamond(CX, 768, 95, 38, C_DIAMOND3_BG, C_DIAMOND3_BORDER)
text_center(CX, 760, "tp > 1?", font_decision, C_TEXT)

# YES → left (broadcast)
# Draw L-shaped arrow: left from diamond, then down to C++ box
draw.line([(s(345), s(768)), (s(200), s(768))], fill=C_YES, width=s(2))
draw.text((s(315), s(750)), "YES", fill=C_YES, font=font_decision)
rounded_rect(55, 748, 145, 44, C_PY, r=6)
text_center(127, 758, "broadcast_unmap", font_box)
text_center(127, 777, "tp_ipc_util.py", font_sub, C_TEXTSUB)
# Arrow from broadcast box: down, then right to join C++ box
# Down from broadcast box bottom
dashed_line(127, 792, 127, 850, C_ARROW)
# Right to the C++ box left edge
dashed_line(127, 850, 265, 850, C_ARROW)
# arrowhead at the C++ box
sx_arr = s(265)
draw.polygon([(sx_arr-2, s(850)-6), (sx_arr-2, s(850)+6), (sx_arr+8, s(850))], fill=C_ARROW)

# NO → right, then down to C++ box
draw.line([(s(535), s(768)), (s(630), s(768))], fill=C_NO, width=s(2))
draw.text((s(543), s(750)), "NO", fill=C_NO, font=font_decision)
# down from 630 to C++ box
draw.line([(s(630), s(768)), (s(630), s(830))], fill=C_ARROW, width=s(2))
draw.polygon([(s(630)-6, s(830)-8), (s(630)+6, s(830)-8), (s(630), s(830))], fill=C_ARROW)

draw_layer_divider(815, "C++ Layer", "csrc/")

# ══════════════════════════════════════════════════════════════
# C++ Layer
# ══════════════════════════════════════════════════════════════

# ── Box 5: C++ unmap_from_kv_tensors ──
rounded_rect(265, 830, 410, 50, C_CPP)
text_center(470, 842, "FTensorAllocator::unmap_from_kv_tensors(offsets)", font_box)
text_center(470, 861, "allocator.cpp — iterate layers x offsets", font_sub, C_TEXTSUB)

arrow_down(470, 880, 915)

# ── Box 6: FTensor::unmap ──
rounded_rect(265, 915, 410, 50, C_CPP)
text_center(470, 927, "FTensor::unmap(offset)", font_box)
text_center(470, 946, "ftensor.cpp — release physical page, remap zero page", font_sub, C_TEXTSUB)

arrow_down(470, 965, 995)

draw_layer_divider(988, "CUDA Driver", "")

# ── Box 7: cuMemUnmap ──
rounded_rect(295, 1000, 350, 45, C_CUDA)
text_center(470, 1015, "cuMemUnmap()", font_box)
text_center(470, 1034, "CUDA Virtual Memory Management API", font_sub, C_TEXTSUB)

img.save("d:/proj/kvcached/docs/free_call_chain.png", "PNG", dpi=(288, 288))
print("PNG saved: d:/proj/kvcached/docs/free_call_chain.png")
print(f"Size: {img.size[0]}x{img.size[1]}")
