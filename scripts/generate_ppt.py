"""生成文档处理微服务架构分享 PPT"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── 颜色常量 ────────────────────────────────────────────────────
BG_DARK    = RGBColor(0x1A, 0x1A, 0x2E)   # 深色背景
BG_CARD    = RGBColor(0x16, 0x21, 0x3E)   # 卡片背景
ACCENT     = RGBColor(0x00, 0xD2, 0xFF)   # 主强调色（青色）
ACCENT2    = RGBColor(0x7C, 0x3A, 0xED)   # 次强调色（紫色）
ACCENT3    = RGBColor(0x10, 0xB9, 0x81)   # 绿色
ACCENT4    = RGBColor(0xF5, 0x9E, 0x0B)   # 橙色
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
GRAY       = RGBColor(0x9C, 0xA3, 0xAF)
LIGHT_GRAY = RGBColor(0xD1, 0xD5, 0xDB)
RED_SOFT   = RGBColor(0xEF, 0x44, 0x44)
GREEN_SOFT = RGBColor(0x22, 0xC5, 0x5E)


def set_slide_bg(slide, color):
    """设置幻灯片背景色"""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Microsoft YaHei"):
    """添加文本框"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_rich_textbox(slide, left, top, width, height, lines, font_name="Microsoft YaHei"):
    """添加多段落富文本框

    lines: [(text, font_size, color, bold, alignment), ...]
    """
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, font_size, color, bold, alignment) in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.font.name = font_name
        p.alignment = alignment
        p.space_after = Pt(4)
    return txBox


def add_card(slide, left, top, width, height, fill_color=BG_CARD):
    """添加圆角矩形卡片"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_arrow(slide, left, top, width, height, color=ACCENT):
    """添加箭头"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_down_arrow(slide, left, top, width, height, color=ACCENT):
    """添加向下箭头"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.DOWN_ARROW, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_circle(slide, left, top, size, fill_color, text="", font_size=14, font_color=WHITE):
    """添加圆形标记"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, left, top, size, size
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.bold = True
        p.font.name = "Microsoft YaHei"
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].space_before = Pt(0)
    return shape


# ── 创建 PPT ────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

W = prs.slide_width
H = prs.slide_height


# ════════════════════════════════════════════════════════════════
# Slide 1: 封面
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
set_slide_bg(slide, BG_DARK)

# 装饰线条
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(3.2), Inches(13.333), Pt(3))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT
shape.line.fill.background()

add_textbox(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.5),
            "文档处理微服务架构", font_size=44, color=WHITE, bold=True)
add_textbox(slide, Inches(1), Inches(2.5), Inches(11), Inches(0.8),
            "从单体到异步微服务的演进之路", font_size=24, color=ACCENT, bold=False)

add_textbox(slide, Inches(1), Inches(4.0), Inches(11), Inches(0.6),
            "预处理 → GLM-OCR 异步识别 → 后处理", font_size=20, color=GRAY)

add_textbox(slide, Inches(1), Inches(5.5), Inches(5), Inches(0.5),
            "2026.06", font_size=16, color=GRAY)
add_textbox(slide, Inches(1), Inches(6.0), Inches(5), Inches(0.5),
            "技术分享", font_size=16, color=GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 2: 目录
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(5), Inches(0.8),
            "目录", font_size=36, color=WHITE, bold=True)

toc_items = [
    ("01", "背景与问题", "为什么需要微服务化"),
    ("02", "整体架构设计", "三大微服务 + Pipeline 编排"),
    ("03", "异步处理架构", "Fire-and-Forget + Redis 聚合"),
    ("04", "预处理服务", "串行流水线：检测→矫正→矫正"),
    ("05", "Pipeline 编排服务", "端到端串联三个微服务"),
    ("06", "端口复用方案", "Uvicorn Workers vs TCP 负载均衡"),
    ("07", "进程管理", "Supervisor 配置与部署"),
    ("08", "开发路线图", "后续规划与优化方向"),
]

for i, (num, title, desc) in enumerate(toc_items):
    y = Inches(1.6) + Inches(i * 0.7)
    add_circle(slide, Inches(1.0), y, Inches(0.45), ACCENT if i % 2 == 0 else ACCENT2,
               text=num, font_size=12)
    add_textbox(slide, Inches(1.7), y - Pt(2), Inches(4), Inches(0.35),
                title, font_size=20, color=WHITE, bold=True)
    add_textbox(slide, Inches(1.7), y + Pt(18), Inches(4), Inches(0.3),
                desc, font_size=14, color=GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 3: 背景与问题
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(5), Inches(0.8),
            "01  背景与问题", font_size=32, color=WHITE, bold=True)

# 问题卡片
problems = [
    ("阻塞等待", "OCR 请求提交后阻塞等待 vLLM 返回，\n无法处理其他请求，吞吐量低"),
    ("无法独立部署", "预处理、OCR、后处理耦合在一起，\n无法按需扩缩容"),
    ("GPU 资源浪费", "vLLM 推理时 CPU 空闲，\n预处理时 GPU 空闲"),
    ("扩展困难", "新增预处理步骤需要改核心代码，\n风险高、迭代慢"),
]

for i, (title, desc) in enumerate(problems):
    x = Inches(0.8) + Inches(i * 3.1)
    card = add_card(slide, x, Inches(1.8), Inches(2.8), Inches(2.5))
    add_circle(slide, x + Inches(0.2), Inches(2.0), Inches(0.4), RED_SOFT,
               text=str(i + 1), font_size=14)
    add_textbox(slide, x + Inches(0.8), Inches(2.0), Inches(1.8), Inches(0.4),
                title, font_size=18, color=WHITE, bold=True)
    add_textbox(slide, x + Inches(0.2), Inches(2.6), Inches(2.4), Inches(1.5),
                desc, font_size=14, color=LIGHT_GRAY)

# 目标
add_card(slide, Inches(0.8), Inches(4.8), Inches(11.7), Inches(2.0))
add_textbox(slide, Inches(1.2), Inches(5.0), Inches(2), Inches(0.4),
            "目标", font_size=20, color=ACCENT, bold=True)

goals = [
    "异步非阻塞：提交即返回，vLLM 结果通过 Redis 聚合",
    "微服务化：预处理、OCR、后处理独立部署、独立扩缩容",
    "灵活组合：Pipeline 编排，可跳过任意步骤",
    "生产就绪：Supervisor 进程管理 + 端口复用",
]
for i, g in enumerate(goals):
    add_textbox(slide, Inches(1.2), Inches(5.5) + Inches(i * 0.35), Inches(10), Inches(0.35),
                f"  {g}", font_size=14, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 4: 整体架构设计
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(5), Inches(0.8),
            "02  整体架构设计", font_size=32, color=WHITE, bold=True)

# 调用方
add_card(slide, Inches(0.5), Inches(2.5), Inches(1.8), Inches(1.2), RGBColor(0x1E, 0x3A, 0x5F))
add_textbox(slide, Inches(0.5), Inches(2.8), Inches(1.8), Inches(0.6),
            "调用方", font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Pipeline 服务
add_arrow(slide, Inches(2.4), Inches(2.9), Inches(0.8), Inches(0.4), ACCENT)
add_card(slide, Inches(3.3), Inches(1.8), Inches(2.2), Inches(2.6), RGBColor(0x1E, 0x29, 0x3B))
add_textbox(slide, Inches(3.3), Inches(2.0), Inches(2.2), Inches(0.4),
            "Pipeline 服务", font_size=16, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(3.5), Inches(2.5), Inches(1.8), Inches(1.8),
            "编排调度\n\n预处理 → OCR\n→ 后处理\n\n端口: 8090",
            font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# 三个微服务
services = [
    ("预处理服务", "文档检测\n方向矫正\n扭曲矫正", "8001", ACCENT3),
    ("GLM-OCR 异步", "Fire-and-Forget\nRedis 聚合\nWebSocket", "8000", ACCENT2),
    ("后处理服务", "结果格式化\n文本清洗\n结构化输出", "8002", ACCENT4),
]

for i, (name, desc, port, color) in enumerate(services):
    x = Inches(6.2) + Inches(i * 2.4)
    add_arrow(slide, Inches(5.6), Inches(2.9) + Inches(i * 0.05), Inches(0.5), Inches(0.35), color)
    add_card(slide, x, Inches(1.8), Inches(2.1), Inches(2.6), RGBColor(0x1E, 0x29, 0x3B))
    add_textbox(slide, x, Inches(2.0), Inches(2.1), Inches(0.4),
                name, font_size=14, color=color, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.15), Inches(2.5), Inches(1.8), Inches(1.5),
                desc, font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x, Inches(3.9), Inches(2.1), Inches(0.3),
                f"端口: {port}", font_size=11, color=GRAY, alignment=PP_ALIGN.CENTER)

# vLLM
add_card(slide, Inches(6.2), Inches(5.0), Inches(2.1), Inches(1.0), RGBColor(0x2D, 0x1B, 0x4E))
add_textbox(slide, Inches(6.2), Inches(5.2), Inches(2.1), Inches(0.6),
            "vLLM 推理\n(GPU)", font_size=13, color=ACCENT2, bold=True, alignment=PP_ALIGN.CENTER)
add_down_arrow(slide, Inches(7.0), Inches(4.5), Inches(0.3), Inches(0.5), ACCENT2)

# Redis
add_card(slide, Inches(8.8), Inches(5.0), Inches(2.1), Inches(1.0), RGBColor(0x3B, 0x1C, 0x1C))
add_textbox(slide, Inches(8.8), Inches(5.2), Inches(2.1), Inches(0.6),
            "Redis\n结果聚合", font_size=13, color=RED_SOFT, bold=True, alignment=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════
# Slide 5: 异步处理架构
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(8), Inches(0.8),
            "03  异步处理架构：Fire-and-Forget", font_size=32, color=WHITE, bold=True)

# 传统同步模式
add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5), Inches(0.4),
            "传统同步模式（阻塞）", font_size=18, color=RED_SOFT, bold=True)

sync_steps = ["提交图片", "等待 vLLM", "等待 vLLM", "等待 vLLM", "返回结果"]
for i, step in enumerate(sync_steps):
    x = Inches(0.8) + Inches(i * 1.2)
    color = RED_SOFT if "等待" in step else GRAY
    add_card(slide, x, Inches(2.0), Inches(1.1), Inches(0.6))
    add_textbox(slide, x, Inches(2.1), Inches(1.1), Inches(0.4),
                step, font_size=11, color=color, alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(0.8), Inches(2.8), Inches(6), Inches(0.3),
            "问题：vLLM 推理慢（秒级），期间 CPU 完全空闲", font_size=13, color=RED_SOFT)

# 异步 Fire-and-Forget
add_textbox(slide, Inches(0.8), Inches(3.4), Inches(8), Inches(0.4),
            "异步 Fire-and-Forget 模式（非阻塞）", font_size=18, color=GREEN_SOFT, bold=True)

async_flow = [
    ("提交图片", ACCENT3),
    ("立即返回\ndoc_id", ACCENT3),
    ("处理下一张", ACCENT),
    ("处理下一张", ACCENT),
    ("Redis 聚合\n结果就绪", ACCENT2),
]
for i, (step, color) in enumerate(async_flow):
    x = Inches(0.8) + Inches(i * 1.2)
    add_card(slide, x, Inches(3.9), Inches(1.1), Inches(0.7))
    add_textbox(slide, x, Inches(4.0), Inches(1.1), Inches(0.6),
                step, font_size=11, color=color, alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(0.8), Inches(4.8), Inches(8), Inches(0.3),
            "优势：提交即返回，CPU 和 GPU 并行工作，吞吐量提升 N 倍", font_size=13, color=GREEN_SOFT)

# 核心机制
add_textbox(slide, Inches(0.8), Inches(5.5), Inches(5), Inches(0.4),
            "核心机制", font_size=18, color=WHITE, bold=True)

mechanisms = [
    ("RegionAggregator", "基于 Redis 的结果聚合器，追踪每个 region 的完成状态"),
    ("Semaphore 控流", "限制并发 vLLM 请求数，防止 OOM"),
    ("WebSocket 推送", "实时推送处理进度，前端无需轮询"),
]
for i, (title, desc) in enumerate(mechanisms):
    x = Inches(0.8) + Inches(i * 4.0)
    add_card(slide, x, Inches(6.0), Inches(3.7), Inches(1.0))
    add_textbox(slide, x + Inches(0.15), Inches(6.05), Inches(3.4), Inches(0.35),
                title, font_size=14, color=ACCENT, bold=True)
    add_textbox(slide, x + Inches(0.15), Inches(6.4), Inches(3.4), Inches(0.5),
                desc, font_size=12, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 6: 预处理服务
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(8), Inches(0.8),
            "04  预处理服务：串行流水线", font_size=32, color=WHITE, bold=True)

# 串行流程
steps = [
    ("文档检测", "检测文档边界\n提取文档区域", ACCENT3),
    ("方向矫正", "检测图像旋转\n自动旋转校正", ACCENT),
    ("扭曲矫正", "透视变换\n矫正扭曲变形", ACCENT2),
]

for i, (name, desc, color) in enumerate(steps):
    x = Inches(0.8) + Inches(i * 3.5)
    add_card(slide, x, Inches(1.6), Inches(2.8), Inches(2.0))
    add_circle(slide, x + Inches(0.15), Inches(1.8), Inches(0.35), color,
               text=str(i + 1), font_size=12)
    add_textbox(slide, x + Inches(0.6), Inches(1.8), Inches(2.0), Inches(0.35),
                name, font_size=18, color=color, bold=True)
    add_textbox(slide, x + Inches(0.15), Inches(2.4), Inches(2.5), Inches(1.0),
                desc, font_size=14, color=LIGHT_GRAY)
    if i < 2:
        add_arrow(slide, x + Inches(2.85), Inches(2.3), Inches(0.6), Inches(0.3), color)

# 关键设计
add_textbox(slide, Inches(0.8), Inches(4.2), Inches(5), Inches(0.4),
            "关键设计", font_size=20, color=WHITE, bold=True)

design_points = [
    ("串行执行", "三个步骤必须按顺序执行，后一步依赖前一步结果"),
    ("独立部署", "可单独部署为微服务，也可嵌入其他流程"),
    ("GPU 模型", "支持 PyTorch / ONNX / TensorRT 多框架加载"),
    ("灵活组合", "支持单步调用 /detect, /orient, /dewarp 或完整流水线 /preprocess"),
]
for i, (title, desc) in enumerate(design_points):
    y = Inches(4.8) + Inches(i * 0.6)
    add_textbox(slide, Inches(1.0), y, Inches(2.5), Inches(0.3),
                f"  {title}", font_size=14, color=ACCENT, bold=True)
    add_textbox(slide, Inches(3.5), y, Inches(8), Inches(0.3),
                desc, font_size=13, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 7: Pipeline 编排服务
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(8), Inches(0.8),
            "05  Pipeline 编排服务", font_size=32, color=WHITE, bold=True)

# 数据流
flow_items = [
    ("文件上传", "/process", GRAY),
    ("预处理", "preprocess:/preprocess", ACCENT3),
    ("OCR 提交", "glmocr:/parse/async", ACCENT2),
    ("轮询等待", "/parse/status/{id}", ACCENT2),
    ("获取结果", "/parse/result/{id}", ACCENT2),
    ("后处理", "postprocess:/postprocess", ACCENT4),
    ("返回结果", "ProcessResponse", ACCENT),
]

for i, (label, detail, color) in enumerate(flow_items):
    x = Inches(0.5) + Inches(i * 1.8)
    add_card(slide, x, Inches(1.6), Inches(1.6), Inches(1.4))
    add_textbox(slide, x, Inches(1.7), Inches(1.6), Inches(0.4),
                label, font_size=13, color=color, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x, Inches(2.2), Inches(1.6), Inches(0.6),
                detail, font_size=9, color=GRAY, alignment=PP_ALIGN.CENTER)
    if i < len(flow_items) - 1:
        add_arrow(slide, x + Inches(1.65), Inches(2.1), Inches(0.12), Inches(0.2), color)

# 代码结构
add_textbox(slide, Inches(0.8), Inches(3.5), Inches(5), Inches(0.4),
            "代码结构", font_size=20, color=WHITE, bold=True)

code_files = [
    ("pipeline/config.py", "配置管理，环境变量驱动", ACCENT),
    ("pipeline/client.py", "三个微服务的 HTTP 客户端封装", ACCENT3),
    ("pipeline/service.py", "核心编排逻辑（预处理→OCR→后处理）", ACCENT2),
    ("pipeline/server.py", "FastAPI 服务入口，暴露统一接口", ACCENT4),
]
for i, (file, desc, color) in enumerate(code_files):
    y = Inches(4.1) + Inches(i * 0.55)
    add_textbox(slide, Inches(1.0), y, Inches(3), Inches(0.3),
                file, font_size=14, color=color, bold=True)
    add_textbox(slide, Inches(4.2), y, Inches(7), Inches(0.3),
                desc, font_size=13, color=LIGHT_GRAY)

# API 接口
add_textbox(slide, Inches(0.8), Inches(6.0), Inches(5), Inches(0.4),
            "API 接口", font_size=20, color=WHITE, bold=True)

apis = [
    ("POST /process", "上传文件，端到端处理"),
    ("POST /process/base64", "输入 base64，跳过预处理"),
    ("GET  /health", "健康检查（含依赖服务状态）"),
]
for i, (endpoint, desc) in enumerate(apis):
    x = Inches(0.8) + Inches(i * 4.0)
    add_card(slide, x, Inches(6.5), Inches(3.7), Inches(0.7))
    add_textbox(slide, x + Inches(0.1), Inches(6.5), Inches(1.8), Inches(0.35),
                endpoint, font_size=12, color=ACCENT, bold=True)
    add_textbox(slide, x + Inches(0.1), Inches(6.85), Inches(3.5), Inches(0.3),
                desc, font_size=11, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 8: 端口复用方案对比
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(10), Inches(0.8),
            "06  端口复用方案对比", font_size=32, color=WHITE, bold=True)

# 表头
headers = ["维度", "Uvicorn --workers", "TCP 负载均衡 (HAProxy/socat)"]
col_widths = [Inches(2.5), Inches(4.5), Inches(5.0)]
col_x = [Inches(0.8)]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

for i, (header, x, w) in enumerate(zip(headers, col_x, col_widths)):
    add_card(slide, x, Inches(1.5), w - Inches(0.05), Inches(0.5), ACCENT2)
    add_textbox(slide, x, Inches(1.52), w - Inches(0.05), Inches(0.45),
                header, font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# 表格数据
rows = [
    ("代码修改量", "改 3 个 server.py", "0 行 Python 改", True),
    ("进程隔离", "master 挂了全挂", "每个进程独立", False),
    ("动态扩缩容", "改 worker 需重启", "scale 命令即生效", False),
    ("单独日志", "所有 worker 混在一起", "每个进程独立日志", False),
    ("后台任务", "worker 挂了 task 全丢", "只丢 1/N 的 task", False),
    ("GPU 显存", "每个 worker 各加载一份", "每个进程各加载一份", None),
    ("依赖", "内置在 uvicorn", "多一个 HAProxy/socat 进程", True),
    ("调用方感知", "一个端口，完全透明", "一个端口，完全透明", None),
]

for ri, (dim, uv, lb, winner) in enumerate(rows):
    y = Inches(2.1) + Inches(ri * 0.6)
    bg = BG_CARD if ri % 2 == 0 else BG_DARK

    # 维度
    add_card(slide, col_x[0], y, col_widths[0] - Inches(0.05), Inches(0.5), bg)
    add_textbox(slide, col_x[0] + Inches(0.1), y + Pt(2), col_widths[0] - Inches(0.2), Inches(0.45),
                dim, font_size=12, color=WHITE, bold=True)

    # Uvicorn
    uv_color = GREEN_SOFT if winner is True else (RED_SOFT if winner is False else LIGHT_GRAY)
    add_card(slide, col_x[1], y, col_widths[1] - Inches(0.05), Inches(0.5), bg)
    add_textbox(slide, col_x[1] + Inches(0.1), y + Pt(2), col_widths[1] - Inches(0.2), Inches(0.45),
                uv, font_size=12, color=uv_color)

    # 负载均衡
    lb_color = GREEN_SOFT if winner is False else (RED_SOFT if winner is True else LIGHT_GRAY)
    add_card(slide, col_x[2], y, col_widths[2] - Inches(0.05), Inches(0.5), bg)
    add_textbox(slide, col_x[2] + Inches(0.1), y + Pt(2), col_widths[2] - Inches(0.2), Inches(0.45),
                lb, font_size=12, color=lb_color)

# 结论
add_card(slide, Inches(0.8), Inches(6.0), Inches(11.7), Inches(1.0), RGBColor(0x1E, 0x3A, 0x2E))
add_textbox(slide, Inches(1.2), Inches(6.1), Inches(11), Inches(0.8),
            "结论：生产环境推荐 TCP 负载均衡方案 — 进程隔离、动态扩缩容、独立日志、任务安全",
            font_size=16, color=GREEN_SOFT, bold=True)


# ════════════════════════════════════════════════════════════════
# Slide 9: Supervisor 进程管理
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(8), Inches(0.8),
            "07  Supervisor 进程管理", font_size=32, color=WHITE, bold=True)

# 服务列表
svc_list = [
    ("预处理服务", "preprocess-service", "8001", "2 workers", ACCENT3, "priority=100"),
    ("GLM-OCR 异步", "glmocr-async-service", "8000", "4 workers", ACCENT2, "priority=200"),
    ("后处理服务", "postprocess-service", "8002", "2 workers", ACCENT4, "priority=300"),
    ("Pipeline 编排", "pipeline-service", "8090", "2 workers", ACCENT, "priority=400"),
]

for i, (name, prog, port, workers, color, pri) in enumerate(svc_list):
    y = Inches(1.6) + Inches(i * 1.2)
    add_card(slide, Inches(0.8), y, Inches(11.7), Inches(1.0))
    add_circle(slide, Inches(1.0), y + Inches(0.15), Inches(0.5), color, text=str(i + 1), font_size=14)
    add_textbox(slide, Inches(1.7), y + Inches(0.1), Inches(2.5), Inches(0.35),
                name, font_size=16, color=color, bold=True)
    add_textbox(slide, Inches(1.7), y + Inches(0.5), Inches(2.5), Inches(0.3),
                prog, font_size=12, color=GRAY)
    add_textbox(slide, Inches(4.5), y + Inches(0.1), Inches(1.5), Inches(0.35),
                f"端口: {port}", font_size=13, color=WHITE)
    add_textbox(slide, Inches(6.2), y + Inches(0.1), Inches(2), Inches(0.35),
                workers, font_size=13, color=WHITE)
    add_textbox(slide, Inches(8.5), y + Inches(0.1), Inches(2.5), Inches(0.35),
                pri, font_size=13, color=GRAY)

# 环境变量
add_textbox(slide, Inches(0.8), Inches(6.0), Inches(5), Inches(0.4),
            "环境变量配置", font_size=18, color=WHITE, bold=True)

env_vars = [
    "PREPROCESS_PORT=8001    PREPROCESS_HOST=127.0.0.1",
    "GLMOCR_PORT=8000        GLMOCR_HOST=127.0.0.1",
    "POSTPROCESS_PORT=8002   POSTPROCESS_HOST=127.0.0.1",
    "PIPELINE_PORT=8090      PIPELINE_HOST=127.0.0.1",
    "REDIS_URL=redis://127.0.0.1:6379/0",
]
for i, var in enumerate(env_vars):
    add_textbox(slide, Inches(1.0), Inches(6.4) + Inches(i * 0.22), Inches(11), Inches(0.22),
                var, font_size=11, color=ACCENT if i < 4 else RED_SOFT)


# ════════════════════════════════════════════════════════════════
# Slide 10: 开发路线图
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(8), Inches(0.8),
            "08  开发路线图", font_size=32, color=WHITE, bold=True)

# 时间线
phases = [
    ("Phase 1\n已完成", "异步架构搭建", [
        "GLM-OCR 异步服务",
        "Fire-and-Forget 模式",
        "Redis 结果聚合",
    ], GREEN_SOFT),
    ("Phase 2\n已完成", "微服务拆分", [
        "预处理服务（串行流水线）",
        "后处理服务",
        "Pipeline 编排服务",
    ], GREEN_SOFT),
    ("Phase 3\n已完成", "生产化部署", [
        "Supervisor 进程管理",
        "端口复用方案",
        "Mock 服务测试",
    ], ACCENT),
    ("Phase 4\n规划中", "优化与监控", [
        "TCP 负载均衡 (HAProxy)",
        "Prometheus + Grafana",
        "Docker Compose 部署",
    ], ACCENT4),
]

for i, (phase, title, items, color) in enumerate(phases):
    x = Inches(0.5) + Inches(i * 3.2)
    add_card(slide, x, Inches(1.6), Inches(2.9), Inches(5.0))
    add_textbox(slide, x, Inches(1.7), Inches(2.9), Inches(0.7),
                phase, font_size=14, color=color, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x, Inches(2.5), Inches(2.9), Inches(0.4),
                title, font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    for j, item in enumerate(items):
        add_textbox(slide, x + Inches(0.2), Inches(3.2) + Inches(j * 0.4), Inches(2.5), Inches(0.35),
                    f"  {item}", font_size=12, color=LIGHT_GRAY)

# 连接线
for i in range(3):
    x = Inches(3.5) + Inches(i * 3.2)
    add_arrow(slide, x, Inches(3.8), Inches(0.3), Inches(0.2), GRAY)


# ════════════════════════════════════════════════════════════════
# Slide 11: 总结
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(5), Inches(0.8),
            "总结", font_size=36, color=WHITE, bold=True)

# 核心成果
add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5), Inches(0.4),
            "核心成果", font_size=22, color=ACCENT, bold=True)

achievements = [
    "异步非阻塞架构 — 吞吐量从 1x 提升到 Nx",
    "三大微服务独立部署 — 按需扩缩容",
    "Pipeline 编排层 — 灵活组合、统一入口",
    "生产级进程管理 — Supervisor + 端口复用",
]
for i, text in enumerate(achievements):
    y = Inches(2.1) + Inches(i * 0.5)
    add_circle(slide, Inches(1.0), y + Pt(2), Inches(0.3), ACCENT3, text="", font_size=10)
    add_textbox(slide, Inches(1.5), y, Inches(10), Inches(0.4),
                text, font_size=16, color=LIGHT_GRAY)

# 关键技术选型
add_textbox(slide, Inches(0.8), Inches(4.3), Inches(5), Inches(0.4),
            "关键技术选型", font_size=22, color=ACCENT2, bold=True)

techs = [
    ("FastAPI + Uvicorn", "高性能异步 Web 框架"),
    ("Redis", "结果聚合 + 进度追踪"),
    ("httpx", "异步 HTTP 客户端，服务间通信"),
    ("Supervisor", "进程管理，自动重启"),
]
for i, (tech, desc) in enumerate(techs):
    y = Inches(4.9) + Inches(i * 0.45)
    add_textbox(slide, Inches(1.0), y, Inches(3), Inches(0.35),
                tech, font_size=14, color=ACCENT, bold=True)
    add_textbox(slide, Inches(4.2), y, Inches(7), Inches(0.35),
                desc, font_size=13, color=LIGHT_GRAY)

# 底部
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(6.8), Inches(13.333), Pt(2))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT
shape.line.fill.background()

add_textbox(slide, Inches(1), Inches(6.9), Inches(11), Inches(0.4),
            "谢谢！Q & A", font_size=20, color=GRAY, alignment=PP_ALIGN.CENTER)


# ── 保存 ────────────────────────────────────────────────────────
output_path = "/workspace/docs/document_pipeline_architecture.pptx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
prs.save(output_path)
print(f"PPT 已生成: {output_path}")
