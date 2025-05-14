# -*- coding: utf-8 -*-            
# @Time : 2025/5/14 15:09
# @Autor: joanzhong
# @FileName: image_utils.py
# @Software: IntelliJ IDEA
import os
import textwrap
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont


def get_available_fonts(directory):
    """ 获取指定目录及其所有子目录下的所有字体文件名（不带扩展名）。 """
    fonts = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # # 获取文件的扩展名
            # name, ext = os.path.splitext(filename)
            # # 检查文件扩展名是否为字体文件
            # if ext.lower() in ['.ttf', '.otf']:
            #     fonts.append(root + "/" + filename)  # 添加不带扩展名的字体名
            if filename.__contains__('Times New Roman.ttf'):
                fonts.append(root + "/" + filename)
    return fonts


def determine_grid_dimensions(num_images):
    """确定能容纳指定数量图片的最接近方形的网格尺寸。"""
    if num_images <= 0:
        return 0, 0
    if num_images == 1:
        return 1, 1
    if num_images == 2:
        return 1, 2
    elif num_images == 3:
        return 1, 3
    elif num_images == 4:
        return 2, 2
    # 对于 5 和 6，2x3 网格是最小的矩形网格
    elif num_images == 5:
        return 2, 3  # 返回能容纳5张图的最小规则网格
    elif num_images == 6:
        return 2, 3
    # 对于 7 和 8，2x4 网格
    elif num_images == 7:
        return 2, 4
    elif num_images == 8:
        return 2, 4
    # 对于 9，3x3 网格
    elif num_images == 9:
        return 3, 3
    # 对于 10, 11, 12，3x4 网格
    elif num_images >= 10 and num_images <= 12:
        return 3, 4
    # 其他情况，尝试保持每行最多 4 列
    else:
        cols = 4
        rows = math.ceil(num_images / cols)
        # 如果这样导致最后一行太空，尝试更接近方形
        if rows * cols - num_images >= cols:  # 如果最后一行空了超过一整行
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)
        return rows, cols


def create_rounded_corner_mask(size, radius):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0) + size, radius=radius, fill=255)
    return mask


def apply_rounded_corners(image, radius):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    rounded_mask = create_rounded_corner_mask(image.size, radius)
    fitted_image = ImageOps.fit(image, rounded_mask.size, Image.Resampling.LANCZOS)
    fitted_image.putalpha(rounded_mask)
    return fitted_image


def apply_shadow(image, shadow_radius=10, shadow_offset=(5, 5), shadow_color=(0, 0, 0, 100), corner_radius=20):
    shadow_width = image.width + shadow_offset[0] * 2
    shadow_height = image.height + shadow_offset[1] * 2
    shadow_base = Image.new("RGBA", (shadow_width, shadow_height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_base)
    shadow_rect = (
        shadow_offset[0],
        shadow_offset[1],
        image.width + shadow_offset[0],
        image.height + shadow_offset[1]
    )
    shadow_draw.rounded_rectangle(shadow_rect, radius=corner_radius, fill=shadow_color)
    shadow = shadow_base.filter(ImageFilter.GaussianBlur(shadow_radius))
    combined = Image.new("RGBA", (shadow.width, shadow.height), (0, 0, 0, 0))
    combined.paste(shadow, (0, 0), shadow)
    combined.paste(image, shadow_offset, image)
    return combined


def arrange_images(image_data, captions, font_dir, img_size=(256, 256), corner_radius=20, shadow_radius=10,
                   shadow_offset=(5, 5), padding=20, caption_height_estimate=50):
    available_fonts = get_available_fonts(font_dir)
    if not available_fonts:
        print("Warning: No valid font files found in the specified directory. Using default.")
    font_path = available_fonts[0] if available_fonts else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    num_images = len(image_data)
    if num_images == 0:
        return Image.new("RGBA", (padding * 2, padding * 2), "white")

    num_rows, max_cols_per_row = determine_grid_dimensions(num_images)
    img_width, img_height = img_size

    cell_width = img_width + shadow_offset[0] * 2
    cell_height = img_height + shadow_offset[1] * 2

    actual_max_cols = min(max_cols_per_row, num_images)

    canvas_width = actual_max_cols * cell_width + (actual_max_cols + 1) * padding
    canvas_height = num_rows * (cell_height + caption_height_estimate) + (num_rows + 1) * padding

    canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, 16)

    y_offset = padding
    for row in range(num_rows):
        images_in_this_row = min(actual_max_cols, num_images - row * actual_max_cols)
        total_row_width = images_in_this_row * cell_width + (images_in_this_row - 1) * padding
        x_offset = (canvas_width - total_row_width) // 2

        for col in range(images_in_this_row):
            image_index = row * actual_max_cols + col
            img = image_data[image_index]
            caption = captions[image_index]

            try:
                img = img.convert('RGBA')  # Ensure RGBA mode
                img.thumbnail(img_size, Image.Resampling.LANCZOS)
                base_img = Image.new('RGBA', img_size, (255, 255, 255, 0))
                paste_x = (img_size[0] - img.width) // 2
                paste_y = (img_size[1] - img.height) // 2
                base_img.paste(img, (paste_x, paste_y))
                rounded_img = apply_rounded_corners(base_img, corner_radius)
                final_img_with_shadow = apply_shadow(rounded_img, shadow_radius, shadow_offset,
                                                     corner_radius=corner_radius)
                canvas.paste(final_img_with_shadow, (x_offset, y_offset), final_img_with_shadow)

                wrapped_caption = "\n".join(textwrap.wrap(caption, width=30))
                text_y = y_offset + cell_height + padding
                for line in wrapped_caption.split('\n'):
                    text_width, text_height = draw.textbbox((0, 0), line, font=font)[2:]
                    text_x = x_offset + (cell_width - text_width) // 2
                    draw.text((text_x, text_y), line, fill="black", font=font)
                    text_y += text_height

                x_offset += cell_width + padding

            except FileNotFoundError:
                print(f"Warning: Image data is corrupt.")
                continue

        y_offset += cell_height + caption_height_estimate + padding

    return canvas
