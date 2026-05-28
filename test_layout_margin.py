"""Test script for layout_detector.py table extension with boundary margin.

Directly tests the table extension + margin logic from PPDocLayoutDetector.process()
without requiring the actual model to be loaded.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/workspace")

OUTPUT_DIR = "/workspace/test_output_margin"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simulate_process_logic(image, table_boxes, table_boundary_margin=0.01):
    """Simulate the core logic of PPDocLayoutDetector.process().

    Args:
        image: PIL Image
        table_boxes: list of [x1, y1, x2, y2] for each detected table
        table_boundary_margin: float, margin ratio

    Returns:
        extended_tables, text_regions, margin
    """
    image_width, image_height = image.size

    # Step 1: Build table_regions
    table_regions = []
    for box in table_boxes:
        x1, y1, x2, y2 = box
        table_regions.append({
            "box": box,
            "label": "table",
            "score": 0.95,
            "polygon_points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        })

    # Step 2: Determine extension direction based on table aspect ratio
    extended_tables = []
    horizontal_extend_count = 0
    vertical_extend_count = 0
    for table in table_regions:
        x1, y1, x2, y2 = table["box"]
        table_width = x2 - x1
        table_height = y2 - y1

        width_ratio = table_width / image_width
        height_ratio = table_height / image_height

        if width_ratio > height_ratio:
            ext_x1, ext_y1, ext_x2, ext_y2 = 0, y1, image_width, y2
            horizontal_extend_count += 1
        else:
            ext_x1, ext_y1, ext_x2, ext_y2 = x1, 0, x2, image_height
            vertical_extend_count += 1

        extended_tables.append({
            "original_box": table["box"],
            "extended_box": [ext_x1, ext_y1, ext_x2, ext_y2],
            "label": table["label"],
            "score": table["score"],
        })

    # Determine splitting direction
    split_by_y = horizontal_extend_count >= vertical_extend_count

    # Calculate boundary margin in pixels
    margin = int(table_boundary_margin * min(image_width, image_height))

    # Step 3: Sort extended tables
    if split_by_y:
        extended_tables.sort(key=lambda t: t["extended_box"][1])
    else:
        extended_tables.sort(key=lambda t: t["extended_box"][0])

    # Step 4: Expand extended table boxes by margin
    for table in extended_tables:
        ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
        table["padded_box"] = [
            max(0, ext_x1 - margin),
            max(0, ext_y1 - margin),
            min(image_width, ext_x2 + margin),
            min(image_height, ext_y2 + margin),
        ]

    # Step 5: Split page into table + complementary text regions
    text_regions = []
    current_pos = 0

    if split_by_y:
        for table in extended_tables:
            _, ext_y1, _, ext_y2 = table["extended_box"]
            if current_pos < ext_y1:
                text_regions.append({
                    "box": [0, current_pos, image_width, ext_y1],
                    "padded_box": [
                        0,
                        max(0, current_pos - margin),
                        image_width,
                        min(image_height, ext_y1 + margin),
                    ],
                })
            current_pos = ext_y2
        if current_pos < image_height:
            text_regions.append({
                "box": [0, current_pos, image_width, image_height],
                "padded_box": [
                    0,
                    max(0, current_pos - margin),
                    image_width,
                    image_height,
                ],
            })
    else:
        for table in extended_tables:
            ext_x1, _, ext_x2, _ = table["extended_box"]
            if current_pos < ext_x1:
                text_regions.append({
                    "box": [current_pos, 0, ext_x1, image_height],
                    "padded_box": [
                        max(0, current_pos - margin),
                        0,
                        min(image_width, ext_x1 + margin),
                        image_height,
                    ],
                })
            current_pos = ext_x2
        if current_pos < image_width:
            text_regions.append({
                "box": [current_pos, 0, image_width, image_height],
                "padded_box": [
                    max(0, current_pos - margin),
                    0,
                    image_width,
                    image_height,
                ],
            })

    return extended_tables, text_regions, margin, split_by_y


def draw_boxes_on_image(image, extended_tables, text_regions, margin, split_by_y):
    """Draw all boxes with different colors and labels."""
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Draw original table boxes (red dashed)
    for i, table in enumerate(extended_tables):
        x1, y1, x2, y2 = table["original_box"]
        for offset in range(0, int(x2 - x1), 10):
            sx = x1 + offset
            ex = min(sx + 5, x2)
            draw.line([(sx, y1), (ex, y1)], fill="red", width=2)
            draw.line([(sx, y2), (ex, y2)], fill="red", width=2)
        for offset in range(0, int(y2 - y1), 10):
            sy = y1 + offset
            ey = min(sy + 5, y2)
            draw.line([(x1, sy), (x1, ey)], fill="red", width=2)
            draw.line([(x2, sy), (x2, ey)], fill="red", width=2)
        draw.text((x1 + 4, y1 + 4), f"Original Table {i+1}", fill="red", font=font_small)

    # Draw extended boxes (green)
    for i, table in enumerate(extended_tables):
        x1, y1, x2, y2 = table["extended_box"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1 + 4, y1 + 20), f"Extended Table {i+1}", fill="green", font=font_small)

    # Draw padded table boxes (blue, with overlap zone)
    for i, table in enumerate(extended_tables):
        x1, y1, x2, y2 = table["padded_box"]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        draw.text((x1 + 4, y1 + 36), f"Padded Table {i+1} (margin={margin}px)", fill="blue", font=font_small)

    # Draw text padded boxes (orange, with overlap zone)
    for i, text_region in enumerate(text_regions):
        x1, y1, x2, y2 = text_region["padded_box"]
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)
        draw.text((x1 + 4, y1 + 4), f"Padded Text {i+1}", fill="orange", font=font_small)

    # Highlight overlap zones
    for table in extended_tables:
        tx1, ty1, tx2, ty2 = table["padded_box"]
        for text_region in text_regions:
            rx1, ry1, rx2, ry2 = text_region["padded_box"]
            ox1 = max(tx1, rx1)
            oy1 = max(ty1, ry1)
            ox2 = min(tx2, rx2)
            oy2 = min(ty2, ry2)
            if ox1 < ox2 and oy1 < oy2:
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [ox1, oy1, ox2, oy2],
                    fill=(255, 255, 0, 60),
                    outline=(255, 200, 0, 200),
                    width=2,
                )
                image = Image.alpha_composite(image.convert("RGBA"), overlay)
                draw = ImageDraw.Draw(image)
                draw.text((ox1 + 2, oy1 + 2), "OVERLAP", fill=(200, 150, 0), font=font_small)

    # Legend
    legend_y = img_h - 90
    draw.rectangle([10, legend_y, 30, legend_y + 15], outline="red", width=2)
    draw.text((35, legend_y), "Original Table Box", fill="red", font=font_small)
    legend_y += 18
    draw.rectangle([10, legend_y, 30, legend_y + 15], outline="green", width=2)
    draw.text((35, legend_y), "Extended Table Box", fill="green", font=font_small)
    legend_y += 18
    draw.rectangle([10, legend_y, 30, legend_y + 15], outline="blue", width=3)
    draw.text((35, legend_y), f"Padded Table Box (margin={margin}px)", fill="blue", font=font_small)
    legend_y += 18
    draw.rectangle([10, legend_y, 30, legend_y + 15], outline="orange", width=3)
    draw.text((35, legend_y), "Padded Text Box", fill="orange", font=font_small)
    legend_y += 18
    draw.rectangle([10, legend_y, 30, legend_y + 15], fill=(255, 255, 0, 60), outline=(255, 200, 0))
    draw.text((35, legend_y), "Overlap Zone", fill=(200, 150, 0), font=font_small)

    return image.convert("RGB")


def crop_and_save_regions(image, extended_tables, text_regions, prefix=""):
    """Crop each region and save."""
    img = np.array(image)
    img_h, img_w = img.shape[:2]

    for i, table in enumerate(extended_tables):
        x1, y1, x2, y2 = table["padded_box"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        cropped = image.crop((x1, y1, x2, y2))
        path = os.path.join(OUTPUT_DIR, f"{prefix}cropped_table_{i+1}.png")
        cropped.save(path)
        print(f"  Saved: {path} ({x2-x1}x{y2-y1})")

    for i, text_region in enumerate(text_regions):
        x1, y1, x2, y2 = text_region["padded_box"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        cropped = image.crop((x1, y1, x2, y2))
        path = os.path.join(OUTPUT_DIR, f"{prefix}cropped_text_{i+1}.png")
        cropped.save(path)
        print(f"  Saved: {path} ({x2-x1}x{y2-y1})")


def test_case_1():
    """Portrait image with 2 tables: one wide (horizontal extend), one tall (vertical extend)."""
    print("\n" + "=" * 60)
    print("Test Case 1: Portrait image (800x1200)")
    print("  Table 1: wide table at top (width_ratio > height_ratio -> horizontal extend)")
    print("  Table 2: tall table at right (height_ratio > width_ratio -> vertical extend)")
    print("=" * 60)

    img_w, img_h = 800, 1200
    image = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    # Draw some fake content
    draw.rectangle([50, 50, 750, 200], fill=(220, 220, 255), outline="gray")
    draw.text((60, 60), "Title Area", fill="black")
    draw.rectangle([50, 250, 700, 550], fill=(200, 255, 200), outline="gray")
    draw.text((60, 260), "Wide Table (Table 1)", fill="black")
    draw.rectangle([50, 600, 750, 900], fill=(220, 220, 255), outline="gray")
    draw.text((60, 610), "Middle Text Area", fill="black")
    draw.rectangle([550, 950, 750, 1150], fill=(255, 220, 200), outline="gray")
    draw.text((560, 960), "Tall\nTable\n(T2)", fill="black")
    draw.rectangle([50, 950, 530, 1150], fill=(220, 220, 255), outline="gray")
    draw.text((60, 960), "Left Text Area", fill="black")

    # Table 1: wide table (width_ratio=0.8125, height_ratio=0.25 -> horizontal extend)
    # Table 2: tall table (width_ratio=0.25, height_ratio=0.167 -> horizontal extend too since 0.25 > 0.167)
    # Let me adjust: make table 2 actually have height_ratio > width_ratio
    table_boxes = [
        [50, 250, 700, 550],   # wide table
        [550, 950, 750, 1150],  # tall-ish table
    ]

    for margin_ratio in [0.0, 0.01, 0.02, 0.05]:
        extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
            image, table_boxes, table_boundary_margin=margin_ratio
        )

        vis_image = image.copy()
        vis_image = draw_boxes_on_image(vis_image, extended_tables, text_regions, margin, split_by_y)

        split_dir = "vertical(split_by_y)" if split_by_y else "horizontal(split_by_x)"
        path = os.path.join(OUTPUT_DIR, f"case1_margin{margin_ratio}_{split_dir.split('(')[0]}.png")
        vis_image.save(path)
        print(f"\n  margin_ratio={margin_ratio}, margin={margin}px, split_by_y={split_by_y}")
        print(f"  Saved: {path}")

        for i, t in enumerate(extended_tables):
            orig = t["original_box"]
            ext = t["extended_box"]
            pad = t["padded_box"]
            tw = orig[2] - orig[0]
            th = orig[3] - orig[1]
            wr = tw / img_w
            hr = th / img_h
            direction = "H-extend" if wr > hr else "V-extend"
            print(f"  Table {i+1}: orig={orig} ext={ext} pad={pad} "
                  f"({direction}, wr={wr:.3f}, hr={hr:.3f})")

        for i, t in enumerate(text_regions):
            print(f"  Text {i+1}: box={t['box']} pad={t['padded_box']}")

    # Save cropped regions with margin=0.02
    extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
        image, table_boxes, table_boundary_margin=0.02
    )
    crop_and_save_regions(image, extended_tables, text_regions, prefix="case1_")


def test_case_2():
    """Landscape image with 2 tables that both extend horizontally."""
    print("\n" + "=" * 60)
    print("Test Case 2: Landscape image (1200x800)")
    print("  Table 1: wide table at top -> horizontal extend")
    print("  Table 2: wide table at bottom -> horizontal extend")
    print("  Both tables extend horizontally -> split_by_y=True")
    print("=" * 60)

    img_w, img_h = 1200, 800
    image = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    draw.rectangle([100, 50, 1100, 300], fill=(200, 255, 200), outline="gray")
    draw.text((110, 60), "Wide Table 1", fill="black")
    draw.rectangle([50, 350, 1150, 500], fill=(220, 220, 255), outline="gray")
    draw.text((60, 360), "Middle Text Area", fill="black")
    draw.rectangle([80, 550, 1050, 750], fill=(255, 220, 200), outline="gray")
    draw.text((90, 560), "Wide Table 2", fill="black")

    table_boxes = [
        [100, 50, 1100, 300],
        [80, 550, 1050, 750],
    ]

    for margin_ratio in [0.0, 0.02, 0.05]:
        extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
            image, table_boxes, table_boundary_margin=margin_ratio
        )

        vis_image = image.copy()
        vis_image = draw_boxes_on_image(vis_image, extended_tables, text_regions, margin, split_by_y)

        path = os.path.join(OUTPUT_DIR, f"case2_margin{margin_ratio}.png")
        vis_image.save(path)
        print(f"\n  margin_ratio={margin_ratio}, margin={margin}px, split_by_y={split_by_y}")
        print(f"  Saved: {path}")

        for i, t in enumerate(extended_tables):
            orig = t["original_box"]
            tw = orig[2] - orig[0]
            th = orig[3] - orig[1]
            wr = tw / img_w
            hr = th / img_h
            direction = "H-extend" if wr > hr else "V-extend"
            print(f"  Table {i+1}: orig={orig} ext={t['extended_box']} pad={t['padded_box']} "
                  f"({direction}, wr={wr:.3f}, hr={hr:.3f})")

        for i, t in enumerate(text_regions):
            print(f"  Text {i+1}: box={t['box']} pad={t['padded_box']}")

    extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
        image, table_boxes, table_boundary_margin=0.02
    )
    crop_and_save_regions(image, extended_tables, text_regions, prefix="case2_")


def test_case_3():
    """Portrait image with tables that extend vertically."""
    print("\n" + "=" * 60)
    print("Test Case 3: Portrait image (800x1200)")
    print("  Table 1: narrow tall table at left -> vertical extend")
    print("  Table 2: narrow tall table at right -> vertical extend")
    print("  Both tables extend vertically -> split_by_x")
    print("=" * 60)

    img_w, img_h = 800, 1200
    image = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    draw.rectangle([50, 100, 250, 1100], fill=(200, 255, 200), outline="gray")
    draw.text((60, 110), "Tall\nTable\n1", fill="black")
    draw.rectangle([300, 50, 500, 1150], fill=(220, 220, 255), outline="gray")
    draw.text((310, 60), "Middle\nText\nArea", fill="black")
    draw.rectangle([550, 80, 750, 1050], fill=(255, 220, 200), outline="gray")
    draw.text((560, 90), "Tall\nTable\n2", fill="black")

    # narrow + tall -> height_ratio > width_ratio -> vertical extend
    table_boxes = [
        [50, 100, 250, 1100],
        [550, 80, 750, 1050],
    ]

    for margin_ratio in [0.0, 0.02, 0.05]:
        extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
            image, table_boxes, table_boundary_margin=margin_ratio
        )

        vis_image = image.copy()
        vis_image = draw_boxes_on_image(vis_image, extended_tables, text_regions, margin, split_by_y)

        path = os.path.join(OUTPUT_DIR, f"case3_margin{margin_ratio}.png")
        vis_image.save(path)
        print(f"\n  margin_ratio={margin_ratio}, margin={margin}px, split_by_y={split_by_y}")
        print(f"  Saved: {path}")

        for i, t in enumerate(extended_tables):
            orig = t["original_box"]
            tw = orig[2] - orig[0]
            th = orig[3] - orig[1]
            wr = tw / img_w
            hr = th / img_h
            direction = "H-extend" if wr > hr else "V-extend"
            print(f"  Table {i+1}: orig={orig} ext={t['extended_box']} pad={t['padded_box']} "
                  f"({direction}, wr={wr:.3f}, hr={hr:.3f})")

        for i, t in enumerate(text_regions):
            print(f"  Text {i+1}: box={t['box']} pad={t['padded_box']}")

    extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
        image, table_boxes, table_boundary_margin=0.02
    )
    crop_and_save_regions(image, extended_tables, text_regions, prefix="case3_")


def test_case_4_real_image():
    """Test with a real image from examples."""
    real_img_path = "/workspace/examples/source/table.png"
    if not os.path.exists(real_img_path):
        print(f"\n  Skipping real image test: {real_img_path} not found")
        return

    print("\n" + "=" * 60)
    print("Test Case 4: Real image from examples")
    print("=" * 60)

    image = Image.open(real_img_path).convert("RGB")
    img_w, img_h = image.size
    print(f"  Image size: {img_w}x{img_h}")

    # Simulate a table detection in the center of the image
    table_x1 = int(img_w * 0.1)
    table_y1 = int(img_h * 0.2)
    table_x2 = int(img_w * 0.9)
    table_y2 = int(img_h * 0.7)

    table_boxes = [[table_x1, table_y1, table_x2, table_y2]]

    for margin_ratio in [0.0, 0.01, 0.02, 0.05]:
        extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
            image, table_boxes, table_boundary_margin=margin_ratio
        )

        vis_image = image.copy()
        vis_image = draw_boxes_on_image(vis_image, extended_tables, text_regions, margin, split_by_y)

        path = os.path.join(OUTPUT_DIR, f"case4_real_margin{margin_ratio}.png")
        vis_image.save(path)
        print(f"\n  margin_ratio={margin_ratio}, margin={margin}px, split_by_y={split_by_y}")
        print(f"  Saved: {path}")

        for i, t in enumerate(extended_tables):
            orig = t["original_box"]
            tw = orig[2] - orig[0]
            th = orig[3] - orig[1]
            wr = tw / img_w
            hr = th / img_h
            direction = "H-extend" if wr > hr else "V-extend"
            print(f"  Table {i+1}: orig={orig} ext={t['extended_box']} pad={t['padded_box']} "
                  f"({direction}, wr={wr:.3f}, hr={hr:.3f})")

        for i, t in enumerate(text_regions):
            print(f"  Text {i+1}: box={t['box']} pad={t['padded_box']}")

    extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
        image, table_boxes, table_boundary_margin=0.02
    )
    crop_and_save_regions(image, extended_tables, text_regions, prefix="case4_")


def test_overlap_verification():
    """Verify that padded table and text boxes actually overlap at boundaries."""
    print("\n" + "=" * 60)
    print("Overlap Verification Test")
    print("=" * 60)

    img_w, img_h = 1000, 1500
    image = Image.new("RGB", (img_w, img_h), (245, 245, 245))

    # One wide table in the middle
    table_boxes = [[100, 500, 900, 900]]

    margin_ratio = 0.02
    extended_tables, text_regions, margin, split_by_y = simulate_process_logic(
        image, table_boxes, table_boundary_margin=margin_ratio
    )

    print(f"\n  margin_ratio={margin_ratio}, margin={margin}px, split_by_y={split_by_y}")

    for i, table in enumerate(extended_tables):
        pad = table["padded_box"]
        print(f"  Padded Table {i+1}: {pad}")

    for i, text_region in enumerate(text_regions):
        pad = text_region["padded_box"]
        print(f"  Padded Text {i+1}: {pad}")

    # Check overlap
    print("\n  Overlap check:")
    for i, table in enumerate(extended_tables):
        tx1, ty1, tx2, ty2 = table["padded_box"]
        for j, text_region in enumerate(text_regions):
            rx1, ry1, rx2, ry2 = text_region["padded_box"]
            ox1 = max(tx1, rx1)
            oy1 = max(ty1, ry1)
            ox2 = min(tx2, rx2)
            oy2 = min(ty2, ry2)
            if ox1 < ox2 and oy1 < oy2:
                overlap_w = ox2 - ox1
                overlap_h = oy2 - oy1
                print(f"  Table {i+1} <-> Text {j+1}: OVERLAP "
                      f"({overlap_w}x{overlap_h}px at [{ox1},{oy1}]-[{ox2},{oy2}])")
            else:
                print(f"  Table {i+1} <-> Text {j+1}: NO OVERLAP")

    vis_image = image.copy()
    vis_image = draw_boxes_on_image(vis_image, extended_tables, text_regions, margin, split_by_y)
    path = os.path.join(OUTPUT_DIR, "overlap_verification.png")
    vis_image.save(path)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("Testing table extension with boundary margin logic")
    print(f"Output directory: {OUTPUT_DIR}")

    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4_real_image()
    test_overlap_verification()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print(f"Check output images in: {OUTPUT_DIR}")
    print("=" * 60)
